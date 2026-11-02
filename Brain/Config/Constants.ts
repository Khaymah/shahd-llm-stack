// THE single source of default settings (rule #5). Every model/training knob has exactly one
// home here, heavily commented. Overrides (JSON presets, env, CLI) merge ONTO this base in
// LoadConfig; the Zod schema (ValidateConfig) carries no defaults so nothing is defaulted twice.
//
// This default is a small, fast Phase-1 config good for CPU smoke-training on a modest corpus.
// Larger runs live in Configs/*.Config.json presets.

import type { ShahdConfig } from "./ConfigTypes.ts";

export const DefaultConfig: ShahdConfig = {
  Model: {
    EmbedDim: 128, // hidden width
    NumLayers: 4, // depth
    NumHeads: 4, // 128 / 4 = 32-dim heads
    BlockSize: 128, // context length (tokens)
    VocabSize: 256, // byte-level default; char/BPE runs override to match their tokenizer
    MlpRatio: 4, // MLP hidden = 512
    LayerNormEps: 1e-5,
    Dropout: 0, // added as a real knob in Phase 2/3; 0 keeps Phase-1 deterministic
    WeightTying: true, // tie token embedding <-> LM head (decided at Phase 1 per ARCHITECTURE)
    InitScale: 0.02,
    UseScaledResidualInit: true, // 1/sqrt(2*NumLayers) on residual-projection weights
    PositionScheme: "Learned", // "Learned" | "Rope" — default keeps Phase-1 behavior
    NormKind: "LayerNorm", // "LayerNorm" | "RmsNorm"
    MlpKind: "Relu", // "Relu" | "SwiGlu" | "GeGlu"
  },
  Optimizer: {
    Kind: "AdamW",
    LearningRate: 3e-4,
    Beta1: 0.9,
    Beta2: 0.999,
    Epsilon: 1e-8,
    WeightDecay: 0.01,
    GradClipNorm: 1.0,
  },
  Schedule: {
    Kind: "Cosine",
    WarmupSteps: 100,
    MaxSteps: 5000,
    MinLrRatio: 0.1,
    // Wsd only: the last 10% of MaxSteps decays peak -> MinLrRatio * peak; the 90% before it holds the
    // peak. Short enough that most of the run trains at full LR, long enough to anneal (WSD/MiniCPM
    // report 10–20%). Ignored by Cosine/Constant.
    DecayFraction: 0.1,
  },
  Training: {
    BatchSize: 16, // gradient-accumulation count
    Seed: 1,
    EvalInterval: 250,
    EvalIterations: 50,
    CheckpointInterval: 1000,
    Workers: 0, // sequence-parallel worker threads; 0 = sequential (exact, bit-stable default)
  },
  Tokenizer: {
    Kind: "Char",
  },
  Safety: {
    Enabled: true, // safety on by default (absolute priority)
    Level: "Standard", // "Off" | "Standard" | "Strict" — controllable strength
  },
  Limits: {
    MaxNewTokens: 1024, // hard cap on tokens per generation
    MaxContextTokens: 4096, // hard cap on prompt/context tokens fed to the model
  },
  Tools: {
    FileAccess: "ReadOnly", // read files (confined to WorkspaceRoot); no writes until widened
    ExecEnabled: false, // code execution OFF by default (absolute safety) — opt in explicitly
    WorkspaceRoot: ".shahd-workspace", // a DEDICATED sandbox dir (not "." = the whole CWD, which would expose every file/secret in the tree by default); widen deliberately per deployment
    WebSearchEnabled: false, // offline by default; web_search returns a labeled stub
    MaxToolSteps: 6, // agent-loop budget
    MaxFileBytes: 262144, // 256 KiB read/write cap
  },
  Compute: {
    Backend: "GoFfi", // "Ts" | "GoFfi" | "Gpu" — GoFfi is the fast owned CPU kernel (falls back to Ts if the DLL is missing)
    Precision: "F64", // "F64" (exact, gradient-checkable) | "F32" (mixed-precision, GPU-prep)
    FallbackToCpu: true, // drop to CPU if the chosen backend is unavailable
  },
  Data: {
    WebEnabled: true, // web ingestion enabled (the network fetch provider is built separately)
    EmbeddingDim: 256, // Foundry embedding dimension
  },
};
