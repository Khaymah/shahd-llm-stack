# Data Foundry

The Foundry is the tiered, inspectable dataset layer and the collection engine that fills it. Every
document is classified into a tier and kept with full provenance, so the corpus is auditable and
improvable:

- **Filtered** — permissive/approved license + passed the quality filter → eligible for training.
- **Raw** — general web / unverified → kept for inspection only, never trained until licensed.
- **Rejected** — non-permissive or low-quality → kept with the reason.

It runs two ways behind one `DocumentStore` interface, so ingestion, reports, export, and the
dashboard are identical either way:

- **in-memory** — no setup; used by tests and quick local runs.
- **Postgres** — persistent, dashboard-backed. Portable: the embedding is stored as **JSONB** and
  similarity is ranked in-app (cosine), so it runs on **any** Postgres with no extension required.

## Data kinds (physically separate tables)

Each data kind lives in its **own** table (`documents_<kind>`, identical schema) so types stay
separated and a model can be trained pure or as a controlled mix. Kinds in use: `code`,
`conversation`, `knowledge`, `books`, `instruction` (`web` is reserved). A run routes its source to
exactly one kind table.

## Storage setup

```bash
docker compose up -d              # start Postgres (see docker-compose.yml)
cp .env.example .env              # DATABASE_URL=postgres://shahd:shahd@localhost:5432/shahd
bun run foundry:migrate           # idempotent: every documents_<kind> table + indexes (+ pg_trgm when available)
bun run Scripts/MigrateKinds.ts --Apply   # only for a DB created before per-kind tables: copies legacy rows over
```

Schema (`Foundry/FoundrySchema.ts`): id (content hash), tier, origin, source, license, lang, content,
bytes, quality_score, content_hash, `embedding jsonb`, reject_reason, provenance, ingested_at. The
collection ledger lives in a small `collection_state` table (self-created on first use). Trained
checkpoints keep only metadata + sha256 in the `checkpoints` table; the payload is a `.ckpt` file under
`FOUNDRY_CHECKPOINT_DIR` (default `Checkpoints/`).

## The collection engine

The control panel (`bun run foundry:dashboard`, http://localhost:8090) is a three-stage pipeline:
**Collect → Train → Chat**. Collection is built around a few principles:

- **Honest accounting.** Ingestion reports `New` vs `Duplicate` (content-hash dedup), never counting a
  re-stored duplicate as fresh data. The Overview shows a **collection ledger** — lifetime collected,
  state, and resume cursor per source.
- **Source semantics.** A source is `bounded` (a fixed dataset — a full collect exhausts it) or
  `streaming` (can keep producing fresh data every run). The UI uses this to explain a re-run that
  added nothing vs. one worth repeating to grow.
- **Stateful, resumable.** `collection_state` persists a per-source cursor. Streaming parquet sources
  resume shard-by-shard across runs; a bounded source is marked complete only when it ran dry *before*
  its item cap (not merely truncated by it).
- **Resilient fetching.** A shared HTTP layer adds exponential backoff (honoring `Retry-After`) and a
  circuit breaker, so a rate-limited endpoint stops cleanly instead of hammering.

## Sources

All providers are behind injected fetchers (testable offline, no hard API dependency). Origin drives
tiering: `owned` (local, trained regardless of license) / `web-permissive` (license-checked) /
`curated` (an approved dataset, license recorded) reach Filtered; `web-general` is always isolated to
Raw (inspect-only) because its license is unverified.

| Source | Kind | License | Semantics | Notes |
| --- | --- | --- | --- | --- |
| GitHub repos (`CreateGitHubRepoProvider`) | code | permissive (SPDX-verified) | streaming | whole-repo ingest; multiple `;`-separated queries per run grow past the 1000-results-per-query cap |
| Local repos (`CreateLocalRepoProvider`) | code | owned | bounded | our own repos on disk (code-file filtered) |
| Local folder (`CreateLocalFolderProvider`) | any (chosen) | chosen | bounded | ingest **every** text file under a folder (books/articles/…), of any type; streams to bound memory; strips Project Gutenberg boilerplate |
| OASST / OASST2 (`CreateOasstProvider`) | conversation | Apache-2.0 | bounded | curated human dialogue, multilingual |
| Stack Exchange (`StackExchangeSource`) | conversation | CC-BY-SA | streaming | paired Q&A → User/Assistant turns (HF parquet) |
| Wikipedia — live (`CreateWikipediaProvider`) | knowledge | CC-BY-SA | streaming | random-article API, capped per run |
| Wikipedia — dumps (`WikiDumpSource`) | knowledge | CC-BY-SA | streaming | bulk parquet, read via HTTP **byte-range** (footer + needed row groups only), resumed by shard cursor |
| GSM8K (`CreateGsmProvider`) | instruction | MIT | bounded | math word problems stored in the canonical `<\|think\|>…<answer>…</answer>` reasoning shape so they feed the SFT template and the STaR verifier |
| Web search (`CreateWebSearchProvider`) | web | unverified | — | isolated Raw tier only |

The generic **HF parquet provider** (`CreateHfParquetProvider`) is the reusable engine behind the
parquet sources: it discovers a dataset's shards from the HF tree API and reads them via byte-range
with `hyparquet`, so new HF datasets plug in as configs (like `WikiDumpSource` / `StackExchangeSource`)
rather than new providers.

## CLI ingestion (no dashboard)

```bash
# GitHub is free unauthenticated (rate-limited); a token raises the limit.
GITHUB_TOKEN=... bun run foundry:ingest-web \
  --Query="language:typescript license:mit stars:>500" --Store=postgres
bun run foundry:ingest-repos          # ingest our own local repos
bun run foundry                       # ingest the seed corpus + print the quality report
```
