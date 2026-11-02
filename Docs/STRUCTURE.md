# Project Structure

Why each thing lives where it does. The guiding split:

- **`Brain/`** is the owned neural model — pure, self-contained, no external-service dependencies.
- **`Foundry/`** is data infrastructure that *feeds* the Brain (database, web ingestion, dashboard).
  It may depend on `Brain/`; the Brain never depends on it. This is why the DB/web dependencies
  (`drizzle`, `postgres`) live only under `Foundry/`.
- Everything else is entry points (`Scripts/`), CI gates (`Tools/`), tests, config, data, and docs.

## Top level

| Path | Purpose |
|---|---|
| `Brain/` | The owned model and everything it needs at run/train time (see below). |
| `Foundry/` | Data-curation infrastructure: tiering, embeddings, store (in-memory + Postgres), web ingestion, dashboard. |
| `Scripts/` | Runnable entry points (train, sample, demos, foundry, migrations). One concern per file. |
| `Tools/` | CI gate scripts (file-length, naming) + the shared file walker. |
| `Tests/` | `*.Test.ts` suites, one per subsystem. |
| `Configs/` | JSON config presets (Phase0/Phase1/Phase2 sizes). |
| `Corpus/` | The hand-authored permissive seed corpus (`Seed/` + `Manifest.json`). Generated builds are gitignored. |
| `GoKernels/` | Go compute kernels (subprocess worker + cgo FFI matmul). Built artifacts gitignored. |
| `Docs/` | This file, the roadmap, architecture, conventions, data-foundry guide. |
| `App/` | The frozen Phase-0 reference model (`nano-gpt.ts`), kept as an oracle; gitignored. |

## `Brain/` — the owned model

Layered low → high; imports only ever point downward.

| Directory | Files (purpose) |
|---|---|
| `Config/` | `Constants.ts` (the single source of defaults), `ValidateConfig.ts` (one Zod schema + cross-field invariants), `ConfigTypes.ts` (types inferred from the schema), `DeriveConfig.ts` (computed values: head dim, scales), `LoadConfig.ts` (merge → validate → derive → hash → freeze). |
| `Random/` | `SeededRng.ts` — deterministic RNG with named streams (init/data/dropout/sampling). |
| `Tensor/` | `Tensor.ts` (flat Float64 data+grad), `TensorFactories.ts` (zeros/randn/…), `Tape.ts` (autograd on/off). |
| `Autograd/` | `Backward.ts` (topological reverse pass), `GradCheck.ts` (finite-difference oracle). |
| `Ops/` | Differentiable primitives — matmul, add, bias, scale, transpose, softmax, cross-entropy (masked/unmasked), layer/RMS norm, GELU/SiLU, rotary embeddings, mul; `OpsBarrel.ts` is the named export surface. |
| `Nn/` | `Shahd.ts` (the transformer), `MultiHeadAttention.ts`, `Mlp.ts`, `NormLayer.ts`, `Block.ts`, `Embedding.ts`, `InitPolicy.ts`, `NnBarrel.ts`. |
| `Optim/` | `Adam.ts`, `AdamW.ts`, `GradClip.ts`, `LrSchedule.ts`, `CreateOptimizer.ts`, `OptimBarrel.ts`. |
| `Tokenizer/` | Char + byte-level BPE, code-aware pretokenization, atomic special tokens, types. |
| `Data/` | **Pure** corpus preparation reused by both training and the Foundry: `LicenseManifest`, `NearDedup` (MinHash), `QualityFilter`, `FimReformat`, `Decontamination`, `CorpusBuilder`, plus `DataLoader`/`TrainValSplit`/`ShardedCorpusReader`. Kept in Brain because it is dependency-free transformation, not infrastructure. |
| `Training/` | `TrainingStep.ts`, `GradAccumulation.ts`, `EvalLoop.ts` (bits-per-byte), `TrainLoop.ts`. |
| `Sampling/` | `Sampler.ts` (temp/top-k/top-p), `Generate.ts`, `KvCache.ts`, `CachedForward.ts`. |
| `Checkpoint/` | Self-describing checkpoint format, writer, reader. |
| `Safety/` | `HarmfulPatterns.ts`, `ContentFilter.ts`, `SafetyPolicy.ts`, `GuardedGenerate.ts` — the controllable safety gate. |
| `Sft/` | `ChatTemplate.ts` (loss masking), `TaskTaxonomy.ts`, `ToolUseExamples.ts`, `SftStep.ts`. |
| `Eval/` `Rl/` | `PassAtK`, resource-capped `CodeExecutor` (not a sandbox), `EvalHarness`; `RejectionSampling` (RLVR). |
| `Serving/` | `ToolProtocol`, `Tools/` (the rich tool system + central capability gate), `AgentLoop`, `ChatSession`, `Compaction`, `InferenceServer`. |
| `Reasoning/` | `SpeculativeDecode`, `SelfConsistency`, `TreeOfThoughts`, `ThinkingMode`. |
| `ComputeBackend/` | The numeric backend seam: `ComputeBackend` (interface), `TsBackend`/`TsBackendF32`, `GoBackend`/`GoFfiBackend`, `BackendSelector` (runtime toggle), `WebGpuMatMul`/`AsyncCompute`. |
| `Logging/` | `Logger.ts` — structured run logs. |

## `Foundry/` — data infrastructure

| File | Purpose |
|---|---|
| `DocumentRecord.ts` | The tiered document type (tier/origin/license/quality/embedding/provenance). |
| `Tiering.ts` | Classify into Filtered / Raw / Rejected (reuses `Brain/Data` filters). |
| `Embedding.ts` | Owned hashing embedding + cosine (for near-dup / find-similar). |
| `DocumentStore.ts` | The persistence interface. |
| `InMemoryDocumentStore.ts` | Dependency-free store (tests + local runs). |
| `PostgresDocumentStore.ts` `FoundrySchema.ts` | Postgres store via Drizzle (portable JSONB embedding; pgvector optional). |
| `DataKinds.ts` `FoundryStores.ts` | Per-kind separation: each kind in its own `documents_<kind>` table, sharing one connection pool. |
| `Ingest.ts` | Classify → hash → embed → upsert (reports New vs Duplicate); plus training-text export of the Filtered tier. |
| `WebSource.ts` `HttpBackoff.ts` `HtmlToText.ts` | Ingestion orchestrator + shared HTTP resilience (backoff/circuit-breaker) + HTML→text. |
| `GitHubProvider.ts` `GitHubRepoProvider.ts` `LocalRepoProvider.ts` `WebSearchProvider.ts` | Code/web sources: GitHub files/whole-repos, local repos, general web (isolated Raw). |
| `OasstProvider.ts` `GsmProvider.ts` `WikipediaProvider.ts` `LocalFolderProvider.ts` `HfParquetProvider.ts` | Dialogue / reasoning / knowledge / books sources; `HfParquetProvider` reads HF parquet by byte-range (Wikipedia dumps, Stack Exchange). |
| `CollectionState.ts` `InMemoryCollectionStateStore.ts` `PostgresCollectionStateStore.ts` | The collection ledger: lifetime/exhausted/resume-cursor per source. |
| `QualityReport.ts` | Aggregate stats for inspection. |
| `Dashboard.ts` `DashboardHtml.ts` `DashboardScript.ts` | Bun-served control plane (collect → train → chat) over any store. |
| `FoundryBarrel.ts` | The public export surface. |

## Conventions

Filenames are PascalCase and named for their responsibility. Each subsystem exposes an explicit
`…Barrel.ts` (no `index.ts`). No grab-bag files. See [CONVENTIONS.md](CONVENTIONS.md) for the full,
CI-enforced rules and [ARCHITECTURE.md](ARCHITECTURE.md) for how the layers fit together.
