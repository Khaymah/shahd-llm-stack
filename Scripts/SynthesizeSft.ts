// Synthetic SFT generation (Plan: Brain-Strengthening §2.3, §8): a LOCAL open-weights teacher served
// over an OpenAI-compatible endpoint writes conversations in Shahd's exact format (think -> tool ->
// answer, multi-turn), grounded in the permissive code of the Foundry (or a local --Corpus directory)
// and verified (executed tests, executed tools, dependency checks), then gated (validation, token
// budget, dedup, eval decontamination). Accepted conversations are appended to --Out as JSONL, one per
// line, the moment they are accepted; a <Out>.stats.json with per-generator acceptance is written at the
// end (also after --MaxMinutes, SIGTERM or a dead teacher). Import with `bun run sft:import <file>`.
//
//   SHAHD_TEACHER_URL=http://127.0.0.1:8081 SHAHD_TEACHER_MODEL=qwen2.5-coder-1.5b-instruct \
//     bun run sft:synthesize --Count=200 --Out=Runs/Sft/Synthetic.jsonl --MaxMinutes=60
//   bun run sft:synthesize --Corpus=../my-repo --CorpusLicense=MIT --Count=50     # no database needed
//   bun run sft:synthesize --Shard=3/8 --Count=400 --Out=Runs/Sft/shard-3.jsonl    # one CI shard
//
// Flags: --Count (accepted items for this shard) · --Mix=grounded:0.4,textbook:0.2,tools:0.25,multiturn:0.15
// · --Slices=ts,js,html,css,go,sql · --Budget=512 (tokens per rendered conversation) · --Shard=k/N
// (0-based) · --Out · --Corpus=<dir> [--CorpusLicense=<SPDX>] · --Seed · --MaxMinutes (0 = no limit)
// · --Tokenizer=<Data/Tokenizer.*.json> (exact token budgets; default: chars/3.6 estimate) · --Spec
// · --GroundingLimit · --TeacherTimeoutSec · --Append (keep an existing --Out instead of truncating).
//
// SECURITY: the textbook generator EXECUTES teacher-written code through Brain/Eval/CodeExecutor.ts,
// which is a resource-capped runner, NOT a sandbox. Run it on an ephemeral CI runner (the
// synthesize-sft workflow) or pass --Mix without textbook on a machine you care about.

import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { RunAgainstTestsAsync } from "../Brain/Eval/EvalHarness.ts";
import { TrainingDecontaminationDocs } from "../Brain/Eval/Suite/SuiteEvalDocs.ts";
import { BytePairEncoder, LoadBpeModel } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { SynthAcceptance } from "../Brain/Sft/Synthetic/SynthAcceptance.ts";
import { CreateDocumentMeasure, CreateTextMeasure, DefaultSynthBudget } from "../Brain/Sft/Synthetic/SynthBudget.ts";
import { DefaultSynthMix, ParseMix, ParseSlices } from "../Brain/Sft/Synthetic/SynthMix.ts";
import { DescribeGeneratorStats, SerializeSynthRecord } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { ParseShard } from "../Brain/Sft/Synthetic/SynthSampling.ts";
import { GroupGrounding, RunSynthesis } from "../Brain/Sft/Synthetic/SynthesisRunner.ts";
import { SynthToolReplay } from "../Brain/Sft/Synthetic/SynthTools.ts";
import type { GroundingDoc, SynthContext } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { CodeSlices } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { LoadDataSpec } from "../Foundry/DataSpec.ts";
import { DefaultGroundingLimit, LoadCorpusGrounding, LoadStoreGrounding, UnknownCorpusLicense } from "../Foundry/SftGrounding.ts";
import { CreateOpenAiTeacher, DefaultTeacherTimeoutMs, ResolveTeacherModel } from "../Foundry/TeacherClient.ts";
import { DataSpecPath, ResolveFoundryStores, TeacherModel, TeacherUrl } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag, ReadFloatArg, ReadIntArg } from "./ScriptArgs.ts";
import { ChatSpecials } from "./TrainingCorpus.ts";

const MsPerSecond = 1000;
const MsPerMinute = 60_000;
const MinBudget = 64;

const Count = ReadIntArg("--Count", 100, { Min: 1 });
const Mix = ParseMix(ReadArg("--Mix=", DefaultSynthMix));
const Slices = ParseSlices(ReadArg("--Slices=", CodeSlices.join(",")));
const Budget = ReadIntArg("--Budget", DefaultSynthBudget, { Min: MinBudget });
const Shard = ParseShard(ReadArg("--Shard=", "0/1"));
const Out = ReadArg("--Out=", "Runs/Sft/Synthetic.jsonl");
const Corpus = ReadArg("--Corpus=", "");
const CorpusLicense = ReadArg("--CorpusLicense=", UnknownCorpusLicense);
const Seed = ReadIntArg("--Seed", 1, { Min: 0 });
const MaxMinutes = ReadFloatArg("--MaxMinutes", 0, { Min: 0 });
const TokenizerPath = ReadArg("--Tokenizer=", "");
const GroundingLimit = ReadIntArg("--GroundingLimit", DefaultGroundingLimit, { Min: 1 });
const TimeoutSec = ReadIntArg("--TeacherTimeoutSec", DefaultTeacherTimeoutMs / MsPerSecond, { Min: 1 });
const Append = ReadFlag("--Append");
const StatsPath = `${Out.replace(/\.jsonl$/i, "")}.stats.json`;

const Started = Date.now();
const Deadline = MaxMinutes > 0 ? Started + MaxMinutes * MsPerMinute : null;

// SIGTERM (CI cancel / timeout) and Ctrl-C finish the CURRENT item, then stop and write the stats.
let StopRequested = false;
for (const Signal of ["SIGTERM", "SIGINT"] as const) {
  process.on(Signal, () => {
    if (StopRequested) process.exit(1); // a second signal: give up immediately
    StopRequested = true;
    console.warn(`[synth] ${Signal}: finishing the current item, then stopping`);
  });
}

async function LoadGrounding(): Promise<GroundingDoc[]> {
  const Spec = LoadDataSpec(DataSpecPath());
  if (Corpus !== "") {
    if (CorpusLicense === UnknownCorpusLicense) console.warn(`[synth] --Corpus without --CorpusLicense: files without a license header are ${UnknownCorpusLicense} and their conversations will be Rejected at import`);
    return LoadCorpusGrounding(Corpus, Spec, CorpusLicense);
  }
  if (process.env["DATABASE_URL"] === undefined) {
    console.warn("[synth] no DATABASE_URL and no --Corpus: the grounded generator has no documents and is skipped");
    return [];
  }
  const Stores = ResolveFoundryStores();
  try {
    return await LoadStoreGrounding(Stores.Kind("code"), Spec, GroundingLimit);
  } finally {
    await Stores.Close();
  }
}

const Tokenizer = TokenizerPath === "" ? null : new SpecialTokenizer(new BytePairEncoder({ Merges: LoadBpeModel(TokenizerPath).Merges }), [...ChatSpecials]);
const Url = TeacherUrl();
const Model = await ResolveTeacherModel(Url, TeacherModel());
const Teacher = CreateOpenAiTeacher({ Url, Model, TimeoutMs: TimeoutSec * MsPerSecond, Backoff: { OnRetry: (Attempt, DelayMs, Reason) => console.warn(`[synth] teacher retry ${Attempt} in ${DelayMs} ms: ${Reason}`) } });
const Grounding = await LoadGrounding();
const Tools = SynthToolReplay();
const Context: SynthContext = {
  Teacher,
  Budget,
  MeasureText: CreateTextMeasure(Tokenizer),
  Slices,
  Grounding: GroupGrounding(Grounding, Shard, Slices),
  Tools,
  Verify: (Problem, Candidate) => RunAgainstTestsAsync(Problem, Candidate),
};
const Gate = new SynthAcceptance({ EvalDocs: TrainingDecontaminationDocs(), Measure: CreateDocumentMeasure(Tokenizer), Budget, Replay: Tools });
const ShardDocs = [...Context.Grounding.values()].reduce((Sum, Docs) => Sum + Docs.length, 0);
console.log(`[synth] teacher ${Model} @ ${Url}; shard ${Shard.Index}/${Shard.Total}; grounding ${ShardDocs} docs in this shard (of ${Grounding.length}); budget ${Budget} tokens (${Tokenizer === null ? "estimated" : `tokenizer ${TokenizerPath}`})${Deadline === null ? "" : `; stops after ${MaxMinutes} min`}`);

mkdirSync(dirname(Out), { recursive: true });
if (!Append) writeFileSync(Out, "");
const Stats = await RunSynthesis({
  Count,
  Mix,
  Shard,
  Seed,
  Context,
  Gate,
  Sink: (Record) => appendFileSync(Out, `${SerializeSynthRecord(Record)}\n`),
  DeadlineMs: Deadline,
  ShouldStop: () => StopRequested,
  Log: (Line) => console.log(Line),
});
writeFileSync(StatsPath, `${JSON.stringify(Stats, null, 2)}\n`);
const Accepted = Object.values(Stats.Generators).reduce((Sum, G) => Sum + G.Accepted, 0);
for (const Line of DescribeGeneratorStats(Stats.Generators)) console.log(`[synth] ${Line}`);
console.log(`[synth] stopped (${Stats.StopReason}) after ${(Stats.ElapsedMs / MsPerMinute).toFixed(1)} min: ${Accepted} conversations -> ${Out} (stats ${StatsPath})`);
process.exit(Stats.StopReason === "teacher-down" || Accepted === 0 ? 1 : 0);
