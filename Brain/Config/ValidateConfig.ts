// The single Zod schema that validates a fully-merged Shahd config. Rule #5: validation lives
// in one place. The schema carries NO defaults — Constants.ts is the sole source of defaults
// (so a value is never silently defaulted in two places). Cross-field invariants live in the
// superRefine below (this is where REVIEW.md's L4 head-divisibility guard is enforced).

import { z } from "zod";

// z.number() already rejects NaN; .finite() additionally rejects ±Infinity, which would otherwise
// pass .positive()/.min(0) and poison every downstream multiply (an Infinity LR or Epsilon is a
// NaN loss one step later). .int() implies finiteness, so integer fields need no extra guard.
const PositiveInt = z.number().int().positive();
const PositiveFinite = z.number().finite().positive();
const UnitInterval = z.number().min(0).max(1);
// Adam moment decay: must be STRICTLY inside (0, 1). Beta = 1 makes the bias correction
// 1 - Beta^t = 0 (division by zero -> Infinity/NaN update); Beta = 0 disables the moment entirely.
const AdamBeta = z.number().gt(0, "must be > 0").lt(1, "must be < 1 (Beta = 1 divides by zero in the bias correction)");

// Dropout is part of the config contract (and checkpoints carry a Dropout RNG stream), but no op in
// Nn/ or Ops/ consumes it yet. Accepting a non-zero value would silently train WITHOUT dropout, so
// the schema refuses anything but 0 until it is implemented.
export const DropoutNotImplementedMessage = "Dropout is reserved and not implemented yet (must be 0)";
// The explicit `: boolean` stops TS from inferring a type predicate (Value is 0), which would narrow
// the inferred config type to the literal 0 — the field stays a number knob for when it is built.
const ReservedDropout = z.number().refine((Value): boolean => Value === 0, { message: DropoutNotImplementedMessage });

export const ModelSchema = z.object({
  EmbedDim: PositiveInt, // hidden width (a.k.a. NEMB)
  NumLayers: PositiveInt, // transformer blocks (depth)
  NumHeads: PositiveInt, // attention heads; must divide EmbedDim (see superRefine)
  BlockSize: PositiveInt, // context length in tokens
  VocabSize: PositiveInt, // token vocabulary size (must match the tokenizer)
  MlpRatio: PositiveFinite, // MLP hidden = EmbedDim * MlpRatio (usually 4)
  LayerNormEps: PositiveFinite,
  Dropout: ReservedDropout, // reserved knob: only 0 is accepted until dropout is implemented
  WeightTying: z.boolean(), // tie token embedding with the LM head
  InitScale: PositiveFinite, // stddev for randn weight init
  UseScaledResidualInit: z.boolean(), // scale residual-projection init by 1/sqrt(2*NumLayers)
  // Architecture staples (Phase-2 lock). Defaults keep Phase-1 behavior; modern stack opts in.
  PositionScheme: z.enum(["Learned", "Rope"]), // learned absolute vs rotary positions
  NormKind: z.enum(["LayerNorm", "RmsNorm"]),
  MlpKind: z.enum(["Relu", "SwiGlu", "GeGlu"]),
  KvHeads: z.number().int().positive().optional(), // GQA: shared K/V heads; omit => = NumHeads (MHA)
});

export const OptimizerSchema = z.object({
  Kind: z.enum(["Adam", "AdamW"]),
  LearningRate: PositiveFinite,
  Beta1: AdamBeta,
  Beta2: AdamBeta,
  Epsilon: PositiveFinite,
  WeightDecay: z.number().finite().min(0), // decoupled decay; used only by AdamW
  GradClipNorm: PositiveFinite, // global-norm clip threshold
});

export const ScheduleSchema = z.object({
  Kind: z.enum(["Cosine", "Constant"]),
  WarmupSteps: z.number().int().min(0),
  MaxSteps: PositiveInt,
  MinLrRatio: UnitInterval, // floor LR as a fraction of the peak, at the end of decay
});

export const TrainingSchema = z.object({
  BatchSize: PositiveInt, // gradient-accumulation count (sequences per optimizer step)
  Seed: z.number().int().nonnegative(),
  EvalInterval: PositiveInt,
  EvalIterations: PositiveInt,
  CheckpointInterval: PositiveInt,
  // Sequence-level parallelism: batch sequences fan out across this many JS worker threads, each
  // running the UNMODIFIED ForwardBackward on shared weights (SharedArrayBuffer) with private grad
  // buffers reduced on the main thread in fixed order. 0 = the sequential path (exact, default).
  Workers: z.number().int().nonnegative(),
});

export const TokenizerSchema = z.object({
  Kind: z.enum(["Char", "Bpe"]),
  MergesPath: z.string().optional(), // required for Bpe; path to trained merges
});

// SAFETY (absolute priority, in one controllable place): a deterministic external content
// filter for the serving/generation boundary. Level tunes strictness; Off disables it entirely.
export const SafetySchema = z.object({
  Enabled: z.boolean(),
  Level: z.enum(["Off", "Standard", "Strict"]),
});

// LIMITS (performance/resource guardrails): hard bounds enforced at generation time so bad
// input or a runaway loop cannot blow past expected resource use.
export const LimitsSchema = z.object({
  MaxNewTokens: PositiveInt,
  MaxContextTokens: PositiveInt,
});

// TOOLS (agent capability gate — the ONE controllable place for tool safety, mirroring Safety).
// Dangerous surfaces (filesystem, code execution, network) are OFF/read-only by default and can
// only be widened here. File tools are additionally confined to WorkspaceRoot at runtime.
export const ToolsSchema = z.object({
  FileAccess: z.enum(["Off", "ReadOnly", "ReadWrite"]), // filesystem tool capability
  ExecEnabled: z.boolean(), // register the code-execution (run_code) tool
  WorkspaceRoot: z.string(), // confinement root for every file tool (traversal is refused)
  WebSearchEnabled: z.boolean(), // when false, web_search is a clearly-labeled offline stub
  MaxToolSteps: PositiveInt, // agent-loop step budget
  MaxFileBytes: PositiveInt, // per-file read/write byte cap
});

// COMPUTE (performance): selects the numeric backend for the matmul hot path. "Ts" with "F64" is
// the exact default (inline, gradient-checkable). "GoFfi" routes to the in-process Go kernel;
// "F32" is the mixed-precision path and the GPU prerequisite. FallbackToCpu drops to CPU when the
// chosen backend is unavailable, so a machine without a working GPU/FFI still runs.
export const ComputeSchema = z.object({
  Backend: z.enum(["Ts", "GoFfi", "Gpu"]),
  Precision: z.enum(["F64", "F32"]),
  FallbackToCpu: z.boolean(),
});

// DATA (the Foundry): toggles for the tiered dataset layer. WebEnabled gates network ingestion
// (permissive sources are training-eligible; general web is stored isolated for inspection only).
export const DataSchema = z.object({
  WebEnabled: z.boolean(),
  EmbeddingDim: PositiveInt,
});

export const ShahdConfigSchema = z
  .object({
    Model: ModelSchema,
    Optimizer: OptimizerSchema,
    Schedule: ScheduleSchema,
    Training: TrainingSchema,
    Tokenizer: TokenizerSchema,
    Safety: SafetySchema,
    Limits: LimitsSchema,
    Tools: ToolsSchema,
    Compute: ComputeSchema,
    Data: DataSchema,
  })
  .superRefine((Config, Ctx) => {
    // L4 guard: attention scale = 1/sqrt(HeadDim) is only correct when heads evenly split EmbedDim.
    if (Config.Model.EmbedDim % Config.Model.NumHeads !== 0) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Model", "NumHeads"],
        message: `EmbedDim (${Config.Model.EmbedDim}) must be divisible by NumHeads (${Config.Model.NumHeads}).`,
      });
    }
    if (Config.Model.PositionScheme === "Rope" && (Config.Model.EmbedDim / Config.Model.NumHeads) % 2 !== 0) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Model", "PositionScheme"],
        message: `RoPE requires an even head dim; EmbedDim/NumHeads = ${Config.Model.EmbedDim / Config.Model.NumHeads} is odd.`,
      });
    }
    if (Config.Model.KvHeads !== undefined && Config.Model.NumHeads % Config.Model.KvHeads !== 0) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Model", "KvHeads"],
        message: `NumHeads (${Config.Model.NumHeads}) must be divisible by KvHeads (${Config.Model.KvHeads}).`,
      });
    }
    if (Config.Schedule.WarmupSteps > Config.Schedule.MaxSteps) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Schedule", "WarmupSteps"],
        message: `WarmupSteps (${Config.Schedule.WarmupSteps}) cannot exceed MaxSteps (${Config.Schedule.MaxSteps}).`,
      });
    }
    if (Config.Tokenizer.Kind === "Bpe" && Config.Tokenizer.MergesPath === undefined) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Tokenizer", "MergesPath"],
        message: `MergesPath is required when Tokenizer.Kind is "Bpe".`,
      });
    }
  });

/** Validate an already-merged raw config object. Throws a ZodError on any violation. */
export function ValidateConfig(Raw: unknown): z.infer<typeof ShahdConfigSchema> {
  return ShahdConfigSchema.parse(Raw);
}
