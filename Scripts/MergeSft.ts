// Merge the shard outputs of the synthesize-sft workflow into ONE dataset: every *.jsonl under --In is
// concatenated (shard order, then file order), exact duplicates across shards dropped by the recomputed
// conversation hash, and the per-shard *.stats.json summed into one summary (accepted / attempted and
// rejection reasons per generator, plus the merged set's mean tokens). Writes
//   <Out>/SyntheticSft.jsonl · <Out>/Stats.json · <Out>/README.md
// and appends the README to $GITHUB_STEP_SUMMARY when set.
//
//   bun run sft:merge --In=Runs/Sft/shards --Out=Runs/Sft/Merged

import { appendFileSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { SynthRecord, SynthStats } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { DedupSynthRecords, DescribeGeneratorStats, ParseSynthRecords, SerializeSynthRecord, SumGeneratorStats, SynthStatsSchema } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { ReadArg } from "./ScriptArgs.ts";

const In = ReadArg("--In=", "Runs/Sft/shards");
const Out = ReadArg("--Out=", "Runs/Sft/Merged");

function WalkFiles(Dir: string): string[] {
  const Found: string[] = [];
  for (const Name of readdirSync(Dir).sort()) {
    const Full = join(Dir, Name);
    if (statSync(Full).isDirectory()) Found.push(...WalkFiles(Full));
    else Found.push(Full);
  }
  return Found;
}

const Files = WalkFiles(In);
const All: SynthRecord[] = [];
let Malformed = 0;
for (const File of Files.filter((Path) => Path.endsWith(".jsonl"))) {
  const Parsed = ParseSynthRecords(readFileSync(File, "utf8"));
  All.push(...Parsed.Records);
  Malformed += Parsed.Invalid.length;
}
const ShardStats: SynthStats[] = [];
for (const File of Files.filter((Path) => Path.endsWith(".stats.json"))) {
  const Parsed = SynthStatsSchema.safeParse(JSON.parse(readFileSync(File, "utf8")));
  if (Parsed.success) ShardStats.push(Parsed.data);
  else console.warn(`sft:merge: ${File} is not a stats file — skipped`);
}

const Merged = DedupSynthRecords(All);
const Generators = SumGeneratorStats(ShardStats);
const MeanTokens: Record<string, number> = {};
const PerGenerator: Record<string, number> = {};
for (const Record of Merged.Records) {
  PerGenerator[Record.Generator] = (PerGenerator[Record.Generator] ?? 0) + 1;
  MeanTokens[Record.Generator] = (MeanTokens[Record.Generator] ?? 0) + Record.Tokens;
}
for (const Name of Object.keys(MeanTokens)) MeanTokens[Name] = Math.round(MeanTokens[Name] / PerGenerator[Name]);
const Teachers = [...new Set(Merged.Records.map((Record) => Record.Teacher))];
const Summary = {
  Shards: ShardStats.map((Stats) => ({ Shard: Stats.Shard, StopReason: Stats.StopReason, ElapsedMs: Stats.ElapsedMs })),
  Teachers,
  Records: Merged.Records.length,
  CrossShardDuplicates: Merged.Duplicates,
  MalformedLines: Malformed,
  PerGenerator,
  MeanTokens,
  Generators,
};

mkdirSync(Out, { recursive: true });
writeFileSync(join(Out, "SyntheticSft.jsonl"), Merged.Records.map((Record) => `${SerializeSynthRecord(Record)}\n`).join(""));
writeFileSync(join(Out, "Stats.json"), `${JSON.stringify(Summary, null, 2)}\n`);
const Readme = [
  "# SyntheticSft",
  "",
  `${Merged.Records.length} conversations from ${ShardStats.length} shard(s); teacher(s): ${Teachers.join(", ") || "none"}; ${Merged.Duplicates} cross-shard duplicates dropped; ${Malformed} malformed lines skipped.`,
  "",
  "| generator | stored | mean tokens |",
  "|---|---|---|",
  ...Object.keys(PerGenerator).sort().map((Name) => `| ${Name} | ${PerGenerator[Name]} | ${MeanTokens[Name]} |`),
  "",
  "Acceptance per generator (all shards):",
  "",
  ...DescribeGeneratorStats(Generators).map((Line) => `- ${Line}`),
  "",
  "Import before SFT: `bun run sft:import SyntheticSft.jsonl`",
  "",
].join("\n");
writeFileSync(join(Out, "README.md"), Readme);
const StepSummary = process.env["GITHUB_STEP_SUMMARY"];
if (StepSummary !== undefined && StepSummary !== "") appendFileSync(StepSummary, Readme);
console.log(Readme);
