// Types for the interactive Foundry dashboard (M9): the settings a "Learn" run accepts and the
// progress events it streams. LearnFn is injected into the dashboard handler so the handler stays
// testable (a mock in tests, the real provider-backed runner at serving time).

import type { RepoLevel } from "./RepoQuality.ts";
import type { DataKind } from "./DataKinds.ts";

export type LearnSettings = {
  Source: "github" | "local" | "both" | "oasst" | "oasst2" | "wikipedia" | "gsm8k" | "wikidump" | "folder" | "stackexchange";
  Query: string; // GitHub search query — or a language filter (oasst/wikipedia/wikidump) or split (gsm8k: train|test|all)
  Repos: string[]; // local repo roots — or, for the "folder" source, the folders to ingest recursively
  Kind?: DataKind; // "folder" source: which documents_<kind> table its files land in (default "books")
  License?: string; // "folder" source: license recorded on every ingested file (default "public-domain")
  MinLevel: RepoLevel;
  MaxRepos: number; // how many repos to process this run
  MaxFilesPerRepo: number;
  MaxBytesPerRepo: number;
  MaxContentBytes: number; // per-file size cap (larger files are dropped)
  SkipLearned: boolean; // don't re-learn repos already in the store
};

export type LearnEvent =
  | { kind: "start"; query: string; source: string; repos: number } // repos = MaxRepos, for the progress bar denominator
  | { kind: "scanning"; label: string } // a "working" status shown during silent gaps (searching / downloading a repo)
  | { kind: "repo"; repo: string; level: string; files: number; bytes: number; ingested: boolean; reason: string | null }
  | { kind: "repo-progress"; repo: string; filesDone: number; filesTotal: number } // per-repo file ingestion
  // ingested = successful upserts; new = rows that did not exist before; duplicate = content-hash dedup
  // hits. Reporting new vs duplicate is what makes a re-collected bounded source read honestly ("0 new ·
  // 14355 duplicate") instead of the misleading "ingested 14355". semantics explains why (bounded source
  // exhausted vs streaming can grow).
  // collected = this source's LIFETIME new-doc total (collection ledger); exhausted = a bounded source
  // confirmed fully collected (re-running only dedups).
  // Terminal event of a learn run - emitted exactly once. `failedDocs` counts documents that failed
  // to store; `failed` is true when the run was aborted (failure budget, provider error, or an
  // unexpected runner error) and `message` carries the first cause; `stopped` is true after Stop.
  | {
      kind: "done";
      ingested: number;
      new?: number;
      duplicate?: number;
      failedDocs?: number;
      semantics?: "bounded" | "streaming";
      collected?: number;
      exhausted?: boolean;
      failed?: boolean;
      stopped?: boolean;
      message?: string;
    }
  | { kind: "error"; message: string };

export type LearnFn = (Settings: LearnSettings, OnEvent: (Event: LearnEvent) => void, Signal?: AbortSignal) => Promise<void>;

// Model TRAINING (distinct from Learn/data-collection): turn the collected Postgres corpus into a
// trained model checkpoint. Runs as a subprocess so it never blocks the dashboard event loop.
export type TrainSettings = {
  Kind: "pretrain" | "chat"; // pretrain a base model (TrainOnFoundry) or SFT a chat model (TrainSftChat)
  Name: string; // checkpoint name — train/keep multiple models side by side
  Resume: boolean; // continue/EXTEND an existing checkpoint of this name (weights+optimizer+RNG) — the
  // "train it more" flow. When false, a completed same-name run retrains fresh (unchanged behavior).
  Steps: number;
  CorpusMb: number;
  EmbedDim: number;
  NumLayers: number;
  NumHeads: number;
  BlockSize: number;
  Merges: number; // vocab = 256 + Merges
  BatchSize: number;
  // Storage precision for the run: F32 halves memory + uses the 8-lane f32 kernels; F64 is the
  // exact default. Resume/warm-start inherit the checkpoint's precision server-side regardless.
  Precision?: "F64" | "F32";
  // Warm start (chat only): name of a pretrained BASE checkpoint whose weights seed the SFT run
  // (TrainSftChat --From). Absent/empty = SFT from random init (the old behavior).
  From?: string;
  // Sequence-parallel worker threads (Config.Training.Workers; 0/absent = sequential) — used by BOTH
  // pretrain and chat/SFT since the pool learned variable-length sequences (T5a).
  Workers?: number;
  // Per-kind data amounts (the mix): pretrain reads CorpusMb of code + KnowledgeMb of knowledge; chat
  // SFT reads CodeSamples code docs + ConvCount real dialogues. Set any to 0 for a pure-kind model.
  KnowledgeMb: number;
  ConvCount: number;
  CodeSamples: number;
  MultiTurn?: number; // stitched 2-3 exchange conversations (the second-message fix); ~15-25% of the mix
};

export type TrainEvent =
  | { kind: "train-start"; steps: number }
  | { kind: "train-info"; text: string } // corpus / bpe / model setup lines
  | { kind: "train-progress"; step: number; steps: number; trainLoss: number; valLoss?: number; elapsedMs?: number; stepMs?: number }
  | { kind: "train-done"; savedTo: string }
  // paused: the run stopped GRACEFULLY with a checkpoint saved at the exact step — the server
  // hot-reloads that checkpoint into the chat model (same as train-done) so "pause -> try it in
  // chat -> resume" needs no manual model re-pick. A plain error/hard kill leaves the model as-is.
  | { kind: "train-error"; message: string; paused?: boolean };

export type TrainFn = (Settings: TrainSettings, OnEvent: (Event: TrainEvent) => void, Signal?: AbortSignal) => Promise<void>;
