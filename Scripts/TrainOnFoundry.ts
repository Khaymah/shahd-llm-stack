// Train a byte-level BPE model on the CLEAN Foundry corpus (the Filtered tier). Byte-level tokenizer
// => full character coverage (never crashes on an unseen char) and better code coverage than the
// 83-char seed model. Honest scope: still a tiny model on CPU — this fixes coverage + robustness and
// improves code-likeness, not natural-language chat. Config-driven so the run can be sized to a time
// budget; --Measure prints per-step wall-time and exits (calibrate before a long run).
//
//   bun run Scripts/TrainOnFoundry.ts --Measure                 # time a few steps, then stop
//   bun run Scripts/TrainOnFoundry.ts --Steps=2000 --Save=Checkpoints/Foundry.ckpt

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import { FimTokens } from "../Brain/Data/FimReformat.ts";
import { ChatTokenList } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokenList } from "../Brain/Serving/ToolProtocol.ts";
import { DedupedIndices } from "../Brain/Data/NearDedup.ts";
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
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";

// Specials reserved above the BPE vocab: EOS + the FIM sentinels + the chat/tool control tokens —
// all reserved unconditionally so a later run (or resume of one) doesn't shift the vocab / token ids.
// The chat/tool specials never appear in the pretraining stream (EncodeBase + no chat data), so their
// embeddings stay at init here; reserving them is what lets a chat SFT run WARM-START from this base
// (TrainSftChat --From) with an IDENTICAL tokenizer/vocab, and the SFT stage trains them.
// ⚠ This changes the vocab offset: base checkpoints saved before the reservation no longer resume
// (the guard below rejects them cleanly — retrain, which the bigger-model plan does anyway).
const FimSpecials = Object.values(FimTokens);
const BaseSpecials = [SpecialTokens.Eos, ...FimSpecials, ...ChatTokenList, ...ToolTokenList];
const NumSpecials = BaseSpecials.length;

const CorpusMb = Number(ReadArg("--CorpusMb=", "3"));
const NumMerges = Number(ReadArg("--Merges=", "256")); // vocab = 256 + merges
const EmbedDim = Number(ReadArg("--EmbedDim=", "128"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const NumHeads = Number(ReadArg("--Heads=", "4"));
const BlockSize = Number(ReadArg("--Block=", "128"));
const BatchSize = Number(ReadArg("--Batch=", "16"));
const Workers = Number(ReadArg("--Workers=", "0")); // sequence-parallel worker threads; 0 = sequential
const Steps = Number(ReadArg("--Steps=", "2000"));
const SavePath = ReadArg("--Save=", CheckpointFile());
const Measure = ReadFlag("--Measure");
// Storage precision: F32 halves tape/weight memory and feeds the 8-lane f32 kernels. Resumed runs
// inherit the CHECKPOINT's precision below (changing width mid-run would reinterpret the weights).
const Precision = ReadArg("--Precision=", "F64") === "F32" ? "F32" as const : "F64" as const;

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
await AddKind("code", CodeMb);
await AddKind("knowledge", KnowledgeMb);
await Stores.Close();
console.log(`corpus: ${Parts.length} docs total, ${(Parts.join("\n\n").length / 1e6).toFixed(2)}MB`);
// Near-duplicate removal (MinHash): the Foundry Filtered tier dedups only EXACT bytes, so near-dups
// (reformatted / renamed copies) survive and would otherwise (a) leak across the train/val split and
// (b) over-weight repeated code. Apply it here, right before training. O(n^2) — fine at current scale;
// an LSH-banded variant is the scale path.
const KeepIdx = new Set(DedupedIndices(Parts, 0.8));
const Docs = Parts.filter((_P, I) => KeepIdx.has(I));
const DocIds = PartIds.filter((_P, I) => KeepIdx.has(I));
console.log(`near-dedup: ${Parts.length} -> ${Docs.length} docs`);
// Guard: an empty/misconfigured Filtered tier would otherwise train on nothing (a degenerate vocab
// still passes Zod) and silently produce a meaningless model.
if (Docs.length < 2) {
  throw new Error(`TrainOnFoundry: need >= 2 Filtered documents after near-dedup (got ${Docs.length}) — collect/curate more data first`);
}
const CorpusText = Docs.join("\n\n");

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
// model to a higher Steps (the dashboard "Resume training" flow) instead of retraining fresh.
type BpeTokenizerState = { Kind: string; Merges: [number, number][] };
let Resume: { Ckpt: Checkpoint; Step: number; Merges: [number, number][] } | null = null;
if (CkptStore !== null && !Measure && !Fresh) {
  const Existing = await CkptStore.Load(CkptName);
  const State = (Existing?.TokenizerState ?? null) as BpeTokenizerState | null;
  if (Existing !== null && State !== null && State.Kind === "Bpe" && Array.isArray(State.Merges)) {
    const M = Existing.Config.Model;
    // VocabSize must also match: EOS + the reserved FIM sentinels add NumSpecials above the BPE vocab
    // (256 + merges). A checkpoint predating that offset is a different architecture — reject it here
    // so we start fresh instead of crashing later in ApplyCheckpoint's VocabSize shape check.
    const ArchMatch = M.EmbedDim === EmbedDim && M.NumLayers === NumLayers && M.NumHeads === NumHeads && M.BlockSize === BlockSize && M.VocabSize === 256 + State.Merges.length + NumSpecials;
    const Match = ArchMatch && (ResumeFlag || Existing.Config.Schedule.MaxSteps === EffectiveSteps);
    const DoneStep = Number((Existing.Meta as Record<string, unknown>)["Step"] ?? 0);
    if (Match && DoneStep < EffectiveSteps) Resume = { Ckpt: Existing, Step: DoneStep, Merges: State.Merges };
  }
  // OVERWRITE GUARD: a same-name checkpoint that is NOT being resumed would be silently replaced by
  // this run's periodic saves — a finished model died that way once. Fail fast instead; overwriting
  // must be said out loud (--Fresh), never implied by reusing a name.
  if (Existing !== null && Resume === null) {
    const At = Number((Existing.Meta as Record<string, unknown>)["Step"] ?? 0);
    console.error(
      ResumeFlag
        ? `cannot resume "${CkptName}": the saved model (step ${At}) has a different architecture/tokenizer or is already at/past the requested ${EffectiveSteps} steps — pick a new name or raise Steps`
        : `model "${CkptName}" already exists (step ${At}) — pick a NEW name, enable Resume to extend it, or pass --Fresh to overwrite it intentionally`,
    );
    process.exit(1);
  }
}

// Tokenizer: reuse the checkpoint's merges when resuming, else train fresh BPE on the corpus. Wrapped
// with an EOS special so document boundaries are a hard token in the training stream.
const T0 = Date.now();
const Bpe = Resume !== null ? { Merges: Resume.Merges } : TrainBpe(CorpusText, NumMerges);
const BaseTok = new BytePairEncoder(Bpe);
const Tokenizer = new SpecialTokenizer(BaseTok, BaseSpecials);
const EosId = Tokenizer.Id(SpecialTokens.Eos);
console.log(`bpe: ${Bpe.Merges.length} merges -> vocab ${Tokenizer.VocabSize}${Resume !== null ? " (reused from checkpoint)" : `; trained in ${((Date.now() - T0) / 1000).toFixed(1)}s`}`);

const WarmupSteps = Math.max(1, Math.min(40, Math.floor(EffectiveSteps / 4)));
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
    Schedule: Inherited !== null ? { ...Inherited.Schedule, MaxSteps: EffectiveSteps } : { Kind: "Cosine", WarmupSteps, MaxSteps: EffectiveSteps, MinLrRatio: 0.1 },
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
const ValPercent = 10;
const SplitBuckets = 100;
function SplitBucket(Id: string): number {
  return Number.parseInt(createHash("sha256").update(Id).digest("hex").slice(0, 8), 16) % SplitBuckets;
}
const Buckets = DocIds.map(SplitBucket);
const IsVal = Buckets.map((B) => B < ValPercent);
// Both sides must be non-empty (Docs.length >= 2 is guaranteed above). On a tiny corpus where the hash
// put everything on one side, move the single most-borderline document across — chosen by (bucket, id),
// so the fix-up is itself deterministic.
const ByBucket = Docs.map((_D, I) => I).sort((A, B) => Buckets[A] - Buckets[B] || (DocIds[A] < DocIds[B] ? -1 : DocIds[A] > DocIds[B] ? 1 : 0));
if (!IsVal.includes(true)) IsVal[ByBucket[0]] = true;
if (!IsVal.includes(false)) IsVal[ByBucket[ByBucket.length - 1]] = false;
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
const StopFile = ReadArg("--StopFile=", "");
const ShouldStop = StopFile === "" ? undefined : (): boolean => existsSync(StopFile);
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
