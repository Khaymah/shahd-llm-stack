// Train a byte-level BPE model on the CLEAN Foundry corpus (the Filtered tier). Byte-level tokenizer
// => full character coverage (never crashes on an unseen char) and better code coverage than the
// 83-char seed model. Honest scope: still a tiny model on CPU — this fixes coverage + robustness and
// improves code-likeness, not natural-language chat. Config-driven so the run can be sized to a time
// budget; --Measure prints per-step wall-time and exits (calibrate before a long run).
//
//   bun run Scripts/TrainOnFoundry.ts --Measure                 # time a few steps, then stop
//   bun run Scripts/TrainOnFoundry.ts --Steps=2000 --Save=Checkpoints/Foundry.ckpt
//   bun run Scripts/TrainOnFoundry.ts --Spec=Data/Spec.json --Steps=20000   # recipe v2 (see MODEL-SCALING)
//
// Recipe v2 knobs: --Spec samples the mix PER SLICE of the Data Spec (context, epoch cap, WSD schedule
// default from it); --Tokenizer reuses a persisted tokenizer (default Data/Tokenizer.v1.json when it
// exists; "fresh" = learn merges from this corpus); --AllowEpochs=<n> overrides the epoch guard.

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { WsdDecayStart } from "../Brain/Config/ValidateConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import { DedupedIndices } from "../Brain/Data/NearDedup.ts";
import { AssignValidationSides } from "../Brain/Data/DocumentSplit.ts";
import { Decontaminate } from "../Brain/Data/Decontamination.ts";
import { TrainingDecontaminationDocs } from "../Brain/Eval/Suite/SuiteEvalDocs.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { CreateTrainWorkerPool } from "../Brain/Training/WorkerPool.ts";
import { TrainLoop } from "../Brain/Training/TrainLoop.ts";
import { Logger } from "../Brain/Logging/Logger.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { BuildCheckpoint, WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { CheckpointFile, ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag, ReadFloatArg } from "./ScriptArgs.ts";
import { AssembleSpecMix, CheckEpochs, ChooseTokenizerFile, DecideResume, DescribeMix, PickMerges, PretrainSources, PretrainSpecials, ResolveSpec, SpecTotalMb, StepsForEpochs, StopFilePredicate } from "./TrainingCorpus.ts";
import { LoadBpeModel } from "../Brain/Tokenizer/Internal/BpeModelFile.ts";

// Specials reserved above the BPE vocab: EOS + the FIM sentinels + the chat/tool control tokens —
// all reserved unconditionally so a later run (or resume of one) doesn't shift the vocab / token ids.
// The chat/tool specials never appear in the pretraining stream (EncodeBase + no chat data), so their
// embeddings stay at init here; reserving them is what lets a chat SFT run WARM-START from this base
// (TrainSftChat --From) with an IDENTICAL tokenizer/vocab, and the SFT stage trains them.
// ⚠ This changes the vocab offset: base checkpoints saved before the reservation no longer resume
// (the guard below rejects them cleanly — retrain, which the bigger-model plan does anyway).
// The layout itself lives in TrainingCorpus.ts (PretrainSpecials) — shared with TrainBpeMerges, whose
// default merge count is sized so that 256 + merges + these specials = the spec's VocabSize.
const BaseSpecials = [...PretrainSpecials];
const NumSpecials = BaseSpecials.length;

// --Spec (opt-in here; Data/Spec.json): the per-slice mix, the context default, the epoch cap and the
// WSD default all come from the spec. Without it the historical code/knowledge-MB path runs unchanged.
const SpecChoice = ResolveSpec(ReadArg("--Spec=", "none"), false);
const Spec = SpecChoice?.Spec ?? null;
const CorpusMbArg = ReadArg("--CorpusMb=", "");
const CorpusMb = Number(CorpusMbArg === "" ? "3" : CorpusMbArg);
const NumMerges = Number(ReadArg("--Merges=", "256")); // vocab = 256 + merges (only when no tokenizer file is used)
const EmbedDim = Number(ReadArg("--EmbedDim=", "128"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const NumHeads = Number(ReadArg("--Heads=", "4"));
const BlockSize = Number(ReadArg("--Block=", Spec !== null ? String(Spec.Training.ContextTokens) : "128"));
const BatchSize = Number(ReadArg("--Batch=", "16"));
const Workers = Number(ReadArg("--Workers=", "0")); // sequence-parallel worker threads; 0 = sequential
let Steps = Number(ReadArg("--Steps=", "2000")); // replaced by --Epochs below when given
const SavePath = ReadArg("--Save=", CheckpointFile());
const Measure = ReadFlag("--Measure");
// Storage precision: F32 halves tape/weight memory and feeds the 8-lane f32 kernels. Resumed runs
// inherit the CHECKPOINT's precision below (changing width mid-run would reinterpret the weights).
const Precision = ReadArg("--Precision=", "F64") === "F32" ? "F32" as const : "F64" as const;
// Persisted tokenizer: explicit --Tokenizer=<file> > Data/Tokenizer.v1.json when present > learn merges
// from this corpus ("fresh" forces that). A resume always keeps the checkpoint's merges (PickMerges).
const TokenizerFile = ChooseTokenizerFile(ReadArg("--Tokenizer=", ""));
// Epoch guard override: an explicit cap that replaces Spec.Training.MaxEpochs for this run.
const AllowEpochsArg = ReadArg("--AllowEpochs=", "");
const AllowEpochs = AllowEpochsArg === "" ? null : ReadFloatArg("--AllowEpochs", 0, { Min: Number.MIN_VALUE });
// LR schedule for a FRESH run (a resume inherits the checkpoint's): Wsd by default under --Spec (a long
// run that can be extended without restarting), Cosine otherwise (the historical default).
const ScheduleKind = ReadArg("--Schedule=", Spec !== null ? "Wsd" : "Cosine");
if (ScheduleKind !== "Wsd" && ScheduleKind !== "Cosine" && ScheduleKind !== "Constant") throw new Error(`--Schedule must be Wsd, Cosine or Constant (got '${ScheduleKind}')`);
const DecayFraction = ReadFloatArg("--DecayFraction", 0.1, { Min: Number.MIN_VALUE });

// Build the pretraining corpus from the SEPARATED kind tables, each up to its own byte budget: code
// (--CodeMb, default = --CorpusMb for back-compat) + optional general knowledge (--KnowledgeMb). This
// is how a base model is composed as pure-code or code+language, with the mix controlled per kind.
const CodeMb = Number(ReadArg("--CodeMb=", String(CorpusMb)));
const KnowledgeMb = Number(ReadArg("--KnowledgeMb=", "0"));
// ByTier reads in ascending id order, so the same budget always selects the same documents (a resumed
// run sees the same corpus). PartIds stays index-aligned with Parts: the id drives the train/val split.
const Stores = ResolveFoundryStores();
const Parts: string[] = [];
const PartIds: string[] = [];
async function AddKind(Which: "code" | "knowledge", BudgetMb: number): Promise<void> {
  if (BudgetMb <= 0) return;
  const Budget = Math.round(BudgetMb * 1e6);
  const Docs = await Stores.Kind(Which).ByTier("Filtered", Math.max(200, Math.ceil(BudgetMb * 400))); // over-read, stop on bytes
  let Bytes = 0;
  let Count = 0;
  for (const Doc of Docs) {
    Parts.push(Doc.Content);
    PartIds.push(Doc.Id);
    Bytes += Doc.Bytes;
    Count++;
    if (Bytes >= Budget) break;
  }
  console.log(`corpus[${Which}]: ${Count} docs, ${(Bytes / 1e6).toFixed(2)}MB`);
}
if (Spec !== null && SpecChoice !== null) {
  // Recipe v2: the mix PER SLICE of the spec (each slice's Langs -> documents, shares from the weights).
  // Total = --CorpusMb when given, else everything the spec asks for (the sum of its TargetMb).
  const TotalMb = CorpusMbArg !== "" ? CorpusMb : SpecTotalMb(Spec);
  const Mix = await AssembleSpecMix(Spec, PretrainSources(Stores), TotalMb, false);
  for (const Line of DescribeMix(Mix)) console.log(`corpus[slice]: ${Line}`);
  for (const Doc of Mix.Docs) {
    Parts.push(Doc.Content);
    PartIds.push(Doc.Id);
  }
  console.log(`corpus: spec ${SpecChoice.Path} (${Spec.Version}), ${Mix.Mb}MB of ${TotalMb}MB requested`);
} else {
  await AddKind("code", CodeMb);
  await AddKind("knowledge", KnowledgeMb);
}
await Stores.Close();
console.log(`corpus: ${Parts.length} docs total, ${(Parts.join("\n\n").length / 1e6).toFixed(2)}MB`);
// Near-duplicate removal (MinHash): the Foundry Filtered tier dedups only EXACT bytes, so near-dups
// (reformatted / renamed copies) survive and would otherwise (a) leak across the train/val split and
// (b) over-weight repeated code. Apply it here, right before training. O(n^2) — fine at current scale;
// an LSH-banded variant is the scale path.
const KeepIdx = new Set(DedupedIndices(Parts, 0.8));
const DedupedDocs = Parts.filter((_P, I) => KeepIdx.has(I));
const DedupedIds = PartIds.filter((_P, I) => KeepIdx.has(I));
console.log(`near-dedup: ${Parts.length} -> ${DedupedDocs.length} docs`);
// Eval decontamination: drop every document sharing a 13-gram with the fixed eval suite or the coding
// problem set, so the suite stays an honest held-out measurement (same rule as TrainOnCorpus/BuildCorpus).
const Decon = Decontaminate(DedupedDocs, TrainingDecontaminationDocs());
const Docs = Decon.Kept.map((I) => DedupedDocs[I]);
const DocIds = Decon.Kept.map((I) => DedupedIds[I]);
console.log(`decontamination: removed ${Decon.Removed.length} docs overlapping the eval suite -> ${Docs.length} docs`);
// Guard: an empty/misconfigured Filtered tier would otherwise train on nothing (a degenerate vocab
// still passes Zod) and silently produce a meaningless model.
if (Docs.length < 2) {
  throw new Error(`TrainOnFoundry: need >= 2 Filtered documents after near-dedup + decontamination (got ${Docs.length}) — collect/curate more data first`);
}
const CorpusText = Docs.join("\n\n");

// --Epochs=<n>: derive Steps from the data instead of guessing it (the inverse of the epoch guard,
// and what the train-model workflow passes). It needs the training-side token count, so the merges
// are resolved early - the tokenizer file, else fresh merges that the real tokenizer below reuses.
const EpochsArg = ReadArg("--Epochs=", "");
let EarlyMerges: [number, number][] | null = null;
if (EpochsArg !== "" && !Measure) {
  const Epochs = ReadFloatArg("--Epochs", 0, { Min: Number.MIN_VALUE });
  EarlyMerges = TokenizerFile !== null ? LoadBpeModel(TokenizerFile.Path).Merges : TrainBpe(CorpusText, NumMerges).Merges;
  const EarlyTok = new SpecialTokenizer(new BytePairEncoder({ Merges: EarlyMerges }), BaseSpecials);
  const EarlyVal = AssignValidationSides(DocIds);
  let TrainingTokens = 0;
  for (let I = 0; I < Docs.length; I++) if (!EarlyVal[I]) TrainingTokens += EarlyTok.EncodeBase(Docs[I]).length + 1; // + EOS
  Steps = StepsForEpochs(Epochs, TrainingTokens, BatchSize, BlockSize);
  console.log(`--Epochs=${Epochs}: ${TrainingTokens.toLocaleString("en-US")} training tokens at batch ${BatchSize} x context ${BlockSize} -> --Steps=${Steps}`);
}

const EffectiveSteps = Measure ? 8 : Steps;
const CkptName = ReadArg("--Name=", "foundry");
const DbUrl = process.env["DATABASE_URL"];
const CkptStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresCheckpointStore(DbUrl) : null;
const Fresh = ReadFlag("--Fresh");
const ResumeFlag = ReadFlag("--Resume"); // explicit "train it more" — extend even past the old MaxSteps

// RESUME: if a checkpoint of the same name + matching architecture exists (e.g. a stopped or crashed
// run), continue from where it left off — reuse its tokenizer, weights, optimizer, and RNG. Otherwise
// start fresh. This is what makes a long run crash-safe together with the periodic saves below.
// Auto-resumes a same-name/same-Steps run (crash recovery); with --Resume it also EXTENDS a finished
// model to a higher Steps (the dashboard "Resume training" flow) instead of retraining fresh. A
// same-name model that is NOT resumed is refused (overwrite guard; --Fresh overwrites deliberately).
let Resume: { Ckpt: Checkpoint; Step: number; Merges: [number, number][] } | null = null;
if (CkptStore !== null && !Measure && !Fresh) {
  const Decision = DecideResume({
    Existing: await CkptStore.Load(CkptName),
    Name: CkptName,
    Steps: EffectiveSteps,
    ResumeFlag,
    RequireSpecials: false,
    What: "model",
    // VocabSize must also match: EOS + the reserved FIM/chat/tool specials add NumSpecials above the BPE
    // vocab (256 + merges). A checkpoint predating that offset is a different architecture — rejected
    // here so we start fresh instead of crashing later in ApplyCheckpoint's VocabSize shape check.
    ArchMatches: (Ckpt, State) => {
      const M = Ckpt.Config.Model;
      return M.EmbedDim === EmbedDim && M.NumLayers === NumLayers && M.NumHeads === NumHeads && M.BlockSize === BlockSize && M.VocabSize === 256 + State.Merges.length + NumSpecials;
    },
  });
  if (Decision.Refusal !== null) {
    console.error(Decision.Refusal);
    process.exit(1);
  }
  if (Decision.Resume !== null) Resume = { Ckpt: Decision.Resume.Ckpt, Step: Decision.Resume.Step, Merges: Decision.Resume.State.Merges };
}

// Tokenizer: the checkpoint's merges when resuming (an explicit --Tokenizer that disagrees is refused),
// else the persisted tokenizer file, else fresh BPE on the corpus. Wrapped with an EOS special so
// document boundaries are a hard token in the training stream.
const T0 = Date.now();
let Choice: ReturnType<typeof PickMerges>;
try {
  Choice = PickMerges(Resume !== null ? { Merges: Resume.Merges, From: "checkpoint", Name: CkptName } : null, TokenizerFile, () => EarlyMerges ?? TrainBpe(CorpusText, NumMerges).Merges);
} catch (Refused) {
  console.error((Refused as Error).message);
  if (CkptStore !== null) await CkptStore.Close();
  process.exit(1);
}
const Bpe = { Merges: Choice.Merges };
const BaseTok = new BytePairEncoder(Bpe);
const Tokenizer = new SpecialTokenizer(BaseTok, BaseSpecials);
const EosId = Tokenizer.Id(SpecialTokens.Eos);
console.log(`bpe: ${Bpe.Merges.length} merges -> vocab ${Tokenizer.VocabSize} (${Choice.Origin}; ${((Date.now() - T0) / 1000).toFixed(1)}s)`);

// Wsd needs the warmup to end before its decay window (validated); on a tiny run clamp it there.
const BaseWarmup = Math.max(1, Math.min(40, Math.floor(EffectiveSteps / 4)));
const WarmupSteps = ScheduleKind === "Wsd" ? Math.min(BaseWarmup, WsdDecayStart(EffectiveSteps, DecayFraction)) : BaseWarmup;
const EvalInterval = Measure ? EffectiveSteps + 1 : Math.max(1, Math.min(100, EffectiveSteps));
// RESUME INHERITANCE: a resumed run continues the CHECKPOINT's training semantics — batch size,
// optimizer, schedule shape — so a stale form/CLI value cannot change the math mid-run (a resumed
// batch-8 run once silently continued at batch 16). Only Steps (the extension target) and the
// operational knobs (workers, eval/checkpoint cadence) come from THIS invocation.
const Inherited = Resume !== null ? Resume.Ckpt.Config : null;
const Config = LoadConfig({
  Overrides: {
    Model: Inherited !== null ? { ...Inherited.Model } : { VocabSize: Tokenizer.VocabSize, EmbedDim, NumLayers, NumHeads, BlockSize, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize: Inherited !== null ? Inherited.Training.BatchSize : BatchSize, Workers, EvalInterval, EvalIterations: 10, CheckpointInterval: EffectiveSteps },
    Schedule: Inherited !== null ? { ...Inherited.Schedule, MaxSteps: EffectiveSteps } : { Kind: ScheduleKind, WarmupSteps, MaxSteps: EffectiveSteps, MinLrRatio: 0.1, DecayFraction },
    Optimizer: Inherited !== null ? { ...Inherited.Optimizer } : { Kind: "AdamW", LearningRate: 0.003 },
    Compute: Inherited !== null ? { ...Inherited.Compute } : { Precision },
  },
  UseCli: false,
  UseEnv: false,
});
if (Inherited !== null) console.log(`resume: inheriting training semantics from the checkpoint (batch ${Inherited.Training.BatchSize}, ${Inherited.Optimizer.Kind} lr ${Inherited.Optimizer.LearningRate})`);

const ComputeChoice = ActivateFromConfig(Config);
const Rng = CreateRngStreams(Config.Training.Seed);
// Document-level split with EOS between docs — no positional train/val leak. Membership is a property
// of the DOCUMENT (its id hashes into a 0..99 bucket; buckets < ValPercent are validation), not of a
// shuffle, so a resumed run — or a later run over a grown corpus — keeps every already-seen document
// on the same side: nothing that was validation ever leaks into training.
// EncodeBase (not Encode): pretraining docs are untrusted; EOS is added structurally below, so raw
// text is never special-split for control tokens.
// The bucket rule itself lives in Brain/Data/DocumentSplit.ts — shared with the eval suite's frozen
// held-out set, so "validation" is the same documents in both places.
// Both sides must be non-empty (Docs.length >= 2 is guaranteed above); AssignValidationSides moves the
// single most-borderline document across when the hash put everything on one side.
const IsVal = AssignValidationSides(DocIds);
function EncodeSide(WantVal: boolean): number[] {
  const Out: number[] = [];
  for (let I = 0; I < Docs.length; I++) {
    if (IsVal[I] !== WantVal) continue;
    for (const Id of Tokenizer.EncodeBase(Docs[I])) Out.push(Id);
    Out.push(EosId);
  }
  return Out;
}
const Train = EncodeSide(false);
const Val = EncodeSide(true);
const ValDocs = IsVal.filter((V) => V).length;
console.log(`split: ${Train.length} train + ${Val.length} val tokens (document-level by id hash: ${Docs.length - ValDocs} train / ${ValDocs} val docs)`);
// EPOCH GUARD (the memorisation check): the unique tokens are the TRAINING side; every step draws
// Batch x Context of them. Over the spec's MaxEpochs (or --AllowEpochs) the run is refused before any
// weight is touched, with the largest Steps that fits. Without --Spec it only reports.
if (!Measure) {
  const Epochs = CheckEpochs({ UniqueTokens: Train.length, Steps: EffectiveSteps, Batch: Config.Training.BatchSize, Context: Config.Model.BlockSize, MaxEpochs: Spec?.Training.MaxEpochs ?? null, AllowEpochs });
  if (!Epochs.Ok) {
    console.error(`epoch guard: ${Epochs.Message}`);
    if (CkptStore !== null) await CkptStore.Close();
    process.exit(1);
  }
  console.log(`epoch guard: ${Epochs.Message}`);
}
// Separate RNG streams: the train loader draws from DataRng (checkpointed, so resume continues the
// exact batch sequence); validation draws from its own stream, so the NUMBER of eval draws can never
// shift the training batch sequence. The val stream is derived from the run seed (reproducible) and
// restarts on resume — eval batches are a measurement, not training state.
const ValStreamSeedOffset = 0x5641; // "VA": any fixed offset; MixSeed decorrelates it from the train streams
const ValRng = CreateRngStreams(Config.Training.Seed + ValStreamSeedOffset).DataRng;
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, ValRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
let StartStep = 0;
if (Resume !== null) {
  ApplyCheckpoint(Resume.Ckpt, Model, Optimizer, Rng); // restore weights + optimizer + RNG state
  StartStep = Resume.Step;
}
// Sequence-parallel pool (after ApplyCheckpoint so the shared weights start from the restored
// values; ApplyCheckpoint itself writes in place, so resume order would work either way).
const Pool = Config.Training.Workers > 0 ? await CreateTrainWorkerPool(Model, Config) : null;
const Params = Model.Parameters().reduce((A, P) => A + P.Size, 0);
console.log(`model: ${Params.toLocaleString()} params (emb=${EmbedDim} L=${NumLayers} ctx=${BlockSize}); backend ${ComputeChoice.Chosen}${Pool !== null ? `; ${Pool.WorkerCount} workers` : ""}; ${Resume !== null ? `resuming from step ${StartStep}` : "fresh"} -> ${EffectiveSteps} steps`);

const GlobalStart = Date.now();
const RunLogger = new Logger(null, true);
// EVERY step is logged with its own duration (StepMs) — the per-step cost is the number being
// tuned, and hiding 49 of every 50 steps hid exactly the variance that matters.
let LastElapsedMs = 0;
const OnStep = (Step: number, Loss: number, ElapsedMs: number): void => {
  const StepMs = ElapsedMs - LastElapsedMs;
  LastElapsedMs = ElapsedMs;
  console.log(JSON.stringify({ Step, TrainLoss: Math.round(Loss * 1e4) / 1e4, StepMs: Math.round(StepMs), ElapsedMs }));
};

if (Measure) {
  TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger, OnStep, undefined, Pool);
  const PerStep = (Date.now() - GlobalStart) / EffectiveSteps;
  console.log(`per-step wall time: ${PerStep.toFixed(0)}ms -> ~${((PerStep * 2000) / 60000).toFixed(1)} min for 2000 steps`);
  console.log("measure-only: not saving.");
  Pool?.Dispose();
  process.exit(0);
}

// Train in chunks, saving a checkpoint after each — a crash loses AT MOST CheckEvery steps, and
// re-running resumes from the last saved checkpoint (nothing done is thrown away). A graceful pause
// (--StopFile appears — the dashboard's Stop button) saves at the EXACT step instead: TrainLoop polls
// the file once per step and returns early, we checkpoint at that step and exit 0.
const ShouldStop = StopFilePredicate(ReadArg("--StopFile=", ""));
const CheckEvery = Math.max(50, Math.floor(EffectiveSteps / 20)); // ~20 saves across the run
function BuildAt(Step: number): Checkpoint {
  return BuildCheckpoint(Model, Optimizer, Rng, { FinalStep: EffectiveSteps, Step, Corpus: "foundry-filtered" }, { Kind: "Bpe", Merges: Bpe.Merges, Specials: BaseSpecials });
}
for (let Start = StartStep; Start < EffectiveSteps; Start += CheckEvery) {
  const End = Math.min(Start + CheckEvery, EffectiveSteps);
  const Reached = TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger, OnStep, { StartStep: Start, EndStep: End, StartMs: GlobalStart }, Pool, ShouldStop);
  const Ckpt = BuildAt(Reached);
  if (CkptStore !== null) {
    await CkptStore.Save(CkptName, Ckpt, new Date().toISOString());
    console.log(`checkpoint saved at step ${Reached}/${EffectiveSteps}`);
  } else {
    WriteCheckpointObject(SavePath, Ckpt);
  }
  if (Reached < End) {
    console.log(`paused by user at step ${Reached}/${EffectiveSteps} — checkpoint saved; press Train again to resume from exactly here`);
    if (CkptStore !== null) await CkptStore.Close();
    Pool?.Dispose();
    process.exit(0);
  }
}
if (CkptStore !== null) {
  await CkptStore.Close();
  console.log(`saved to Postgres checkpoint "${CkptName}" (trained to step ${EffectiveSteps})`);
} else {
  console.log(`saved ${SavePath} (no DATABASE_URL — file store)`);
}

const Sample = Generate(Model, Tokenizer.Encode("export function "), 160, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("\n--- sample ---\n" + Tokenizer.Decode(Sample));
Pool?.Dispose();
process.exit(0);
