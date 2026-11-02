// SFT worker: train a CHAT model (Phase 8). Builds a SpecialTokenizer (byte-level BPE + chat/tool
// special tokens), assembles an owned instruction corpus (persona + arithmetic-as-tool-call +
// thinking + tool-use exemplars + code-language-ID over the Filtered corpus), and trains with MASKED
// cross-entropy (loss only on assistant tokens) via the existing SftForwardBackward. Saves a
// checkpoint tagged Format:"chat" whose tokenizer state records the specials, so LoadRunnableModel
// rebuilds the SpecialTokenizer and serving routes it through the agent loop + reasoning trace.
// Spawned by the dashboard's "Train Chat" panel; emits JSON progress lines it parses. Honest scope:
// teaches FORMAT + tool-calling + thinking scaffold, not fluency/knowledge (scale-bound).
//
//   bun run Scripts/TrainSftChat.ts --Name=Chat --Steps=600
//   bun run Scripts/TrainSftChat.ts --Name=ChatV2 --From=BaseV2 --Spec=Data/Spec.json   # recipe v2
//
// Held-out discipline: code and conversation documents on the VALIDATION side of the shared document
// split (Brain/Data/DocumentSplit.ts) are excluded, so SFT never trains on what held-out BPB measures.
// --Tokenizer reuses a persisted tokenizer for a from-scratch run (default Data/Tokenizer.v1.json when it
// exists); a resume or --From keeps the checkpoint's own merges. --Spec sets the context default.

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer, ComputeLr } from "../Brain/Optim/OptimBarrel.ts";
import { ClipGradsChecked } from "../Brain/Training/RunTrainingSteps.ts";
import { SftForwardBackward } from "../Brain/Sft/SftStep.ts";
import { RenderForTraining, RenderChat } from "../Brain/Sft/ChatTemplate.ts";
import type { TrainingSequence } from "../Brain/Sft/ChatTemplate.ts";
import { BuildOwnedConversations, OwnedSystemPrompt } from "../Brain/Sft/OwnedSftData.ts";
import type { CodeSample } from "../Brain/Sft/OwnedSftData.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { BuildCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint, ApplyCheckpointWeights } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import type { TierDocument } from "../Foundry/DocumentStore.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { CreateTrainWorkerPool } from "../Brain/Training/WorkerPool.ts";
import { StripThinking } from "../Brain/Reasoning/ThinkingMode.ts";
import { Decontaminate } from "../Brain/Data/Decontamination.ts";
import { TrainingDecontaminationDocs } from "../Brain/Eval/Suite/SuiteEvalDocs.ts";
import { PartitionByDocumentSplit, TrainingSideReadSize } from "../Brain/Data/DocumentSplit.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";
import { ChatSpecials, CheckpointStep, ChooseTokenizerFile, DecideResume, PickMerges, ReadBpeState, ResolveSpec, StopFilePredicate } from "./TrainingCorpus.ts";
import type { BpeTokenizerState } from "./TrainingCorpus.ts";

const Name = ReadArg("--Name=", "Chat");
const Steps = Number(ReadArg("--Steps=", "600"));
const NumMerges = Number(ReadArg("--Merges=", "700"));
const EmbedDim = Number(ReadArg("--EmbedDim=", "192"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const NumHeads = Number(ReadArg("--Heads=", "4"));
// --Spec (opt-in): the context default comes from Spec.Training.ContextTokens (match the base's).
const Spec = ResolveSpec(ReadArg("--Spec=", "none"), false)?.Spec ?? null;
const BlockSize = Number(ReadArg("--Block=", Spec !== null ? String(Spec.Training.ContextTokens) : "256"));
const BatchSize = Number(ReadArg("--Batch=", "8"));
const Workers = Number(ReadArg("--Workers=", "0")); // sequence-parallel worker threads; 0 = sequential
const Lr = Number(ReadArg("--Lr=", "0.002"));
const ResumeFlag = ReadFlag("--Resume"); // explicit "train it more" — extend an existing chat model
const Fresh = ReadFlag("--Fresh"); // explicit overwrite of a same-name model (bypasses the guard below)
const From = ReadArg("--From=", ""); // warm-start a NEW chat run from a pretrained BASE checkpoint's weights
// Storage precision (see TrainOnFoundry): inherited from the checkpoint on resume and from the
// BASE on --From (its weights are stored in that width's distribution).
const Precision = ReadArg("--Precision=", "F64") === "F32" ? "F32" as const : "F64" as const;
// Persisted tokenizer for a from-scratch run: explicit file > Data/Tokenizer.v1.json when present >
// merges learned from the SFT corpus ("fresh"). Resume / --From always keep the checkpoint's merges.
const TokenizerFile = ChooseTokenizerFile(ReadArg("--Tokenizer=", ""));

// 1) SFT data from the SEPARATED kind tables, each capped independently: code (for the owned
// language-ID task) from documents_code, real dialogue from documents_conversation. Knowledge
// (Wikipedia) is NOT dialogue — it belongs to pretraining, not chat SFT.
const CodeSamples = Number(ReadArg("--CodeSamples=", "4000"));
const ConvCount = Number(ReadArg("--ConvCount=", "4000"));
// Owned-mix balance. The original 200/120/1500 skewed the distribution ~7:1 toward code-language-ID,
// and the trained model answered arithmetic questions with a language name — the behaviors that need
// TOOL CALLS and THINKING must carry comparable weight to the ones that answer directly.
const ArithmeticCount = Number(ReadArg("--Arithmetic=", "2000"));
const ThinkingCount = Number(ReadArg("--Thinking=", "800"));
const CodeConvs = Number(ReadArg("--CodeConvs=", "800"));
// Multi-turn stitched conversations — the turn-transition coverage single exchanges can't teach.
// Weighted high (~25% of a default mix) so "answer the NEXT question after a finished exchange" is
// a first-class behavior: 1500 (~7% of rendered sequences next to a large real-dialogue mix) moved
// second-turn behavior but didn't lock it in; the other behaviors are long since saturated on a
// mature checkpoint, so the heavier dose costs them nothing.
const MultiTurn = Number(ReadArg("--MultiTurn=", "3000"));
const Stores = ResolveFoundryStores();
// Only the TRAINING side of the document split: over-read so ~the requested count survives the 10%
// hold-out, then cap. Same ascending-id read as before, so the selection stays deterministic.
async function TrainingSide(Kind: "code" | "conversation", Wanted: number): Promise<{ Docs: TierDocument[]; HeldOut: number }> {
  const Split = PartitionByDocumentSplit(await Stores.Kind(Kind).ByTier("Filtered", TrainingSideReadSize(Wanted)));
  return { Docs: Split.Train.slice(0, Wanted), HeldOut: Split.Validation.length };
}
const CodeSide = await TrainingSide("code", CodeSamples);
const ConvSide = await TrainingSide("conversation", ConvCount);
const CodeDocs = CodeSide.Docs;
const ConvDocs = ConvSide.Docs;
console.log(`held out (validation side of the document split, never trained): ${CodeSide.HeldOut} code + ${ConvSide.HeldOut} conversation docs`);
const Samples: CodeSample[] = CodeDocs.map((D) => ({ Lang: D.Lang, Content: D.Content }));
const Rng = CreateRngStreams(1234);
const Conversations = BuildOwnedConversations(Samples, Rng.DataRng, { ArithmeticCount, ThinkingCount, PersonaRepeats: 25, MaxCodeConversations: CodeConvs, MultiTurnCount: MultiTurn });

// Parse each collected conversation doc ("User: …\n\nAssistant: …") into an SFT example so the chat
// model learns from real dialogue — the link that makes "collect conversation data -> the model talks".
// Same UNIFIED system prompt as the owned mix: a third near-identical wording ("helpful assistant")
// would split a tiny model's behavior across prompts — the exact failure class of the typescript
// incident. Real dialogue carries NO synthetic <|think|> span (we don't fabricate reasoning for text
// we didn't author); the owned mix supplies the thinking distribution.
let AddedGeneral = 0;
for (const Doc of ConvDocs) {
  const Match = /^User: ([\s\S]*?)\n\nAssistant: ([\s\S]*)$/.exec(Doc.Content);
  if (Match === null) continue;
  Conversations.push([{ Role: "System", Content: OwnedSystemPrompt }, { Role: "User", Content: Match[1]! }, { Role: "Assistant", Content: Match[2]! }]);
  AddedGeneral++;
}
await Stores.Close();
// Eval decontamination: drop every conversation (owned or collected) sharing a 13-gram with the fixed
// eval suite or the coding problem set, BEFORE the tokenizer is trained or anything is rendered.
const Decon = Decontaminate(Conversations.map((C) => C.map((M) => M.Content).join("\n")), TrainingDecontaminationDocs());
for (const Index of [...Decon.Removed].reverse()) Conversations.splice(Index, 1);
console.log(`decontamination: removed ${Decon.Removed.length} conversations overlapping the eval suite`);
console.log(`sft: ${Conversations.length} conversations (${CodeDocs.length} code samples + ${AddedGeneral} real dialogues from documents_conversation)`);

// RESUME/EXTEND: continue an existing chat checkpoint of this name — reuse its EXACT tokenizer (merges
// + specials), weights, optimizer, and RNG — instead of retraining fresh. Auto on a same-name/same-
// Steps run (crash recovery); with --Resume it also EXTENDS a finished model to a higher Steps (the
// dashboard "train it more" flow). Requires a matching architecture, else it starts fresh.
const DbUrl = process.env["DATABASE_URL"];
const CkptStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresCheckpointStore(DbUrl) : null;
type ChatTokenizerState = { Merges: [number, number][]; Specials: string[] };
let Resume: { Ckpt: Checkpoint; Step: number } | null = null;
let ResumeState: ChatTokenizerState | null = null;
if (CkptStore !== null && !Fresh) {
  // Same contract as TrainOnFoundry (DecideResume): resume a matching same-name run, else refuse to
  // overwrite it unless --Fresh.
  const Decision = DecideResume({
    Existing: await CkptStore.Load(Name),
    Name,
    Steps,
    ResumeFlag,
    RequireSpecials: true,
    What: "chat model",
    ArchMatches: (Ckpt) => {
      const M = Ckpt.Config.Model;
      return M.EmbedDim === EmbedDim && M.NumLayers === NumLayers && M.NumHeads === NumHeads && M.BlockSize === BlockSize;
    },
  });
  if (Decision.Refusal !== null) {
    console.error(Decision.Refusal);
    process.exit(1);
  }
  if (Decision.Resume !== null) {
    Resume = { Ckpt: Decision.Resume.Ckpt, Step: Decision.Resume.Step };
    ResumeState = ChatState(Decision.Resume.State);
  }
}

/** A BPE state with its specials (DecideResume/RequireSpecials and the --From check guarantee them). */
function ChatState(State: BpeTokenizerState): ChatTokenizerState {
  return { Merges: State.Merges, Specials: State.Specials ?? [] };
}

// WARM START (--From): begin a NEW chat run from a pretrained BASE checkpoint's weights — the
// pretrain→SFT bridge that gives the chat model a language foundation instead of random init.
// The base's tokenizer (merges + specials, which now reserve the chat/tool control tokens) is reused
// VERBATIM so every token id lands on the embedding row the base trained; only the weights transfer —
// optimizer/schedule/RNG are fresh (ApplyCheckpointWeights). Resume takes precedence: a resumed run
// continues from its OWN weights and --From is ignored with a warning.
type WarmStart = { Ckpt: Checkpoint; State: ChatTokenizerState };
let Warm: WarmStart | null = null;
if (From !== "") {
  if (Resume !== null) {
    console.warn(`--From=${From} ignored: resuming "${Name}" continues from its own weights`);
  } else if (CkptStore === null) {
    console.error(`--From requires DATABASE_URL (the base checkpoint lives in Postgres)`);
    process.exit(1);
  } else {
    const BaseCkpt = await CkptStore.Load(From);
    if (BaseCkpt === null) {
      console.error(`--From: base model "${From}" not found`);
      process.exit(1);
    }
    const Base = ReadBpeState(BaseCkpt);
    if (Base === null || Base.Specials === null) {
      console.error(`--From: base model "${From}" has no reusable BPE tokenizer state`);
      process.exit(1);
    }
    const State = ChatState(Base);
    const Missing = ChatSpecials.filter((T) => !State.Specials.includes(T));
    if (Missing.length > 0) {
      console.error(`--From: base "${From}" vocab lacks the chat/tool specials (${Missing.join(", ")}) — retrain the base (TrainOnFoundry now reserves them)`);
      process.exit(1);
    }
    const M = BaseCkpt.Config.Model;
    if (M.EmbedDim !== EmbedDim || M.NumLayers !== NumLayers || M.NumHeads !== NumHeads || M.BlockSize !== BlockSize) {
      console.error(`--From: base "${From}" is emb${M.EmbedDim} L${M.NumLayers} h${M.NumHeads} ctx${M.BlockSize} but this run requested emb${EmbedDim} L${NumLayers} h${NumHeads} ctx${BlockSize} — architectures must match exactly`);
      process.exit(1);
    }
    Warm = { Ckpt: BaseCkpt, State };
  }
}

// 2) SpecialTokenizer = byte-level BPE + chat/tool special tokens. Reuse the checkpoint's exact merges
// + specials when resuming or warm-starting (so token ids align with the trained embeddings; an explicit
// --Tokenizer that disagrees is refused); else the persisted tokenizer file; else fresh BPE on the SFT
// corpus.
const Inherit = ResumeState ?? Warm?.State ?? null;
const Specials = Inherit !== null ? Inherit.Specials : [...ChatSpecials];
const CorpusText = Conversations.flatMap((C) => C.map((M) => M.Content)).join("\n");
let Choice: ReturnType<typeof PickMerges>;
try {
  const InheritedFrom = Inherit === null ? null : { Merges: Inherit.Merges, From: ResumeState !== null ? "checkpoint" as const : "base" as const, Name: ResumeState !== null ? Name : From };
  Choice = PickMerges(InheritedFrom, TokenizerFile, () => TrainBpe(CorpusText, NumMerges).Merges);
} catch (Refused) {
  console.error((Refused as Error).message);
  if (CkptStore !== null) await CkptStore.Close();
  process.exit(1);
}
const Bpe = { Merges: Choice.Merges };
const Tokenizer = new SpecialTokenizer(new BytePairEncoder(Bpe), Specials);
if (Warm !== null && Warm.Ckpt.Config.Model.VocabSize !== Tokenizer.VocabSize) {
  console.error(`--From: base "${From}" VocabSize ${Warm.Ckpt.Config.Model.VocabSize} != rebuilt tokenizer vocab ${Tokenizer.VocabSize} (corrupt/inconsistent tokenizer state)`);
  process.exit(1);
}
console.log(`tokenizer: ${Bpe.Merges.length} merges + ${Specials.length} specials -> vocab ${Tokenizer.VocabSize} (${Choice.Origin})`);

// 3) Pre-render every conversation with the loss mask; keep only those that fit the context AND have
// at least one trainable (assistant) token. Report how many were dropped (no silent truncation).
const Rendered: TrainingSequence[] = [];
let Dropped = 0;
for (const Conversation of Conversations) {
  const Seq = RenderForTraining(Conversation, Tokenizer);
  if (Seq.Ids.length <= BlockSize + 1 && Seq.LossMask.includes(true)) Rendered.push(Seq);
  else Dropped++;
}
console.log(`training sequences: ${Rendered.length} fit context ${BlockSize} (${Dropped} dropped as too long)`);
if (Rendered.length === 0) throw new Error("no SFT sequences fit the context — raise --Block");

// 4) Model at the special-token vocab + a fine-tune-style schedule.
// RESUME INHERITANCE — same contract as TrainOnFoundry: a resumed run keeps the checkpoint's
// training semantics (batch/optimizer/schedule shape); only Steps and operational knobs are new.
const Warmup = Math.max(1, Math.min(40, Math.floor(Steps / 5)));
const Inherited = Resume !== null ? Resume.Ckpt.Config : null;
const Config = LoadConfig({
  Overrides: {
    // Warm start adopts the BASE's full Model block (not just the CLI dims): the weights were trained
    // under its non-shape fields too (MlpKind/NormKind/PositionScheme/ratios), and rebuilding them
    // from CLI defaults would silently reinterpret the loaded tensors.
    Model: Inherited !== null ? { ...Inherited.Model } : Warm !== null ? { ...Warm.Ckpt.Config.Model } : { VocabSize: Tokenizer.VocabSize, EmbedDim, NumLayers, NumHeads, BlockSize, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize: Inherited !== null ? Inherited.Training.BatchSize : BatchSize, Workers },
    Schedule: Inherited !== null ? { ...Inherited.Schedule, MaxSteps: Steps } : { Kind: "Cosine", WarmupSteps: Warmup, MaxSteps: Steps, MinLrRatio: 0.1 },
    Optimizer: Inherited !== null ? { ...Inherited.Optimizer } : { Kind: "AdamW", LearningRate: Lr },
    Compute: Inherited !== null ? { ...Inherited.Compute } : Warm !== null ? { ...Warm.Ckpt.Config.Compute } : { Precision },
  },
  UseCli: false,
  UseEnv: false,
});
if (Inherited !== null) console.log(`resume: inheriting training semantics from the checkpoint (batch ${Inherited.Training.BatchSize}, ${Inherited.Optimizer.Kind} lr ${Inherited.Optimizer.LearningRate})`);
const ComputeChoice = ActivateFromConfig(Config);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
let StartStep = 0;
if (Resume !== null) {
  ApplyCheckpoint(Resume.Ckpt, Model, Optimizer, Rng); // restore weights + optimizer + RNG state
  StartStep = Resume.Step;
} else if (Warm !== null) {
  ApplyCheckpointWeights(Warm.Ckpt, Model); // weights only — fresh AdamW/schedule/RNG for the SFT stage
  console.log(`warm start: applied base "${From}" weights (pretrained to step ${CheckpointStep(Warm.Ckpt)})`);
}
// Sequence-parallel pool (after ApplyCheckpoint so the shared weights start from the restored
// values). Batch size comes from the RESOLVED config — on resume it is the checkpoint's, not the CLI's.
const EffBatch = Config.Training.BatchSize;
// Report how many passes over the SFT set this run makes (sequence epochs; SFT revisits examples by
// design, so this is reported, not enforced — the pretraining epoch guard lives in TrainOnFoundry).
console.log(`sft epochs ~ ${((Steps * EffBatch) / Rendered.length).toFixed(2)} (${Steps} steps x batch ${EffBatch} over ${Rendered.length} sequences)`);
const Pool = Config.Training.Workers > 0 ? await CreateTrainWorkerPool(Model, Config, "sft") : null;
const Params = Model.Parameters().reduce((A, P) => A + P.Size, 0);
console.log(`model: ${Params.toLocaleString()} params (emb=${EmbedDim} L=${NumLayers} ctx=${BlockSize}); backend ${ComputeChoice.Chosen}${Pool !== null ? `; ${Pool.WorkerCount} workers` : ""}; ${Resume !== null ? `resuming from step ${StartStep}` : Warm !== null ? `warm from base "${From}"` : "fresh"} -> ${Steps} SFT steps`);

// 5) SFT loop: masked forward/backward over a batch, divide accumulated grad by batch, clip, step.
// (CkptStore was opened above for resume detection; reused here for periodic + final saves.)
const GlobalStart = Date.now();
// Every step is logged with its own duration (StepMs) — same contract as TrainOnFoundry.
let LastElapsedMs = 0;
function BuildAt(Step: number): Checkpoint {
  // Provenance: a warm-started model records which base its weights came from.
  return BuildCheckpoint(Model, Optimizer, Rng, { FinalStep: Steps, Step, Corpus: "sft-owned", Format: "chat", ...(Warm !== null ? { From } : {}) }, { Kind: "Bpe", Merges: Bpe.Merges, Specials });
}
const CheckEvery = Math.max(50, Math.floor(Steps / 20));
// Graceful pause (--StopFile appears — the dashboard's Stop button): checked once per step so the
// checkpoint lands on the EXACT step the user stopped at, not the last periodic save.
const ShouldStop = StopFilePredicate(ReadArg("--StopFile=", ""));
for (let Step = StartStep; Step < Steps; Step++) {
  const CurrentLr = ComputeLr(Step, Config);
  let MeanLoss: number;
  if (Pool !== null) {
    // The main thread draws the batch with the SAME RNG calls as the sequential path (identical
    // data stream); the pool runs the sequences, reduces, and applies the 1/batch scaling.
    const Batch: TrainingSequence[] = [];
    for (let B = 0; B < EffBatch; B++) Batch.push(Rendered[Math.floor(Rng.DataRng.NextFloat() * Rendered.length)]!);
    MeanLoss = Pool.AccumulateSft(Batch);
  } else {
    Optimizer.ZeroGrad();
    let Loss = 0;
    for (let B = 0; B < EffBatch; B++) {
      const Seq = Rendered[Math.floor(Rng.DataRng.NextFloat() * Rendered.length)]!;
      Loss += SftForwardBackward(Model, Seq);
    }
    const Inv = 1 / EffBatch;
    for (const P of Optimizer.Params) for (let I = 0; I < P.Size; I++) P.Grad[I] *= Inv;
    MeanLoss = Loss * Inv;
  }
  ClipGradsChecked(Optimizer.Params, Config.Optimizer.GradClipNorm, Step);
  Optimizer.Step(CurrentLr);
  const ElapsedMs = Date.now() - GlobalStart;
  const StepMs = ElapsedMs - LastElapsedMs;
  LastElapsedMs = ElapsedMs;
  console.log(JSON.stringify({ Step, TrainLoss: Math.round(MeanLoss * 1e4) / 1e4, StepMs: Math.round(StepMs), ElapsedMs }));
  if (CkptStore !== null && (Step + 1) % CheckEvery === 0) {
    await CkptStore.Save(Name, BuildAt(Step + 1), new Date().toISOString());
    console.log(`checkpoint "${Name}" saved at step ${Step + 1}/${Steps}`);
  }
  if (ShouldStop !== undefined && Step + 1 < Steps && ShouldStop()) {
    if (CkptStore !== null) {
      await CkptStore.Save(Name, BuildAt(Step + 1), new Date().toISOString());
      await CkptStore.Close();
    }
    console.log(`paused by user at step ${Step + 1}/${Steps} — checkpoint "${Name}" saved; press Train again to resume from exactly here`);
    Pool?.Dispose();
    process.exit(0);
  }
}
if (CkptStore !== null) {
  await CkptStore.Save(Name, BuildAt(Steps), new Date().toISOString());
  await CkptStore.Close();
  console.log(`saved chat checkpoint "${Name}" (Format=chat) to Postgres`);
} else {
  console.log("no DATABASE_URL — chat checkpoint not persisted (set DATABASE_URL to save).");
}

// 6) Probe: does it produce the format? (garbled at this scale — the point is the mechanism works.)
for (const Probe of ["hi", "What is 7 + 5?"]) {
  // The probe MUST use the exact training system prompt — a tiny model keys behavior on the prompt
  // prefix, and probing with a different wording measures nothing (train/serve prompt parity).
  const Prompt = RenderChat([{ Role: "System", Content: OwnedSystemPrompt }, { Role: "User", Content: Probe }], true);
  const Out = Generate(Model, Tokenizer.Encode(Prompt), 40, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
  const Text = Tokenizer.Decode(Out.slice(Tokenizer.Encode(Prompt).length));
  console.log(`[probe] "${Probe}" -> ${JSON.stringify(StripThinking(Text).slice(0, 100))}`);
}
Pool?.Dispose();
process.exit(0);
