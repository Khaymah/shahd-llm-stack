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

## Quality gates — three layers

Every candidate document passes three gates before it can reach the `Filtered` (training) tier:

1. **Structural** (`Foundry/CodeFileFilter.ts`, `Brain/Data/QualityFilter.ts`): size window (200 B – 50 KB by
   default; the spec can override), per-language extension allowlist, and rejection of lockfiles,
   snapshots, i18n/locale tables, minified/generated files, string/data tables, imports-only files and
   files that declare nothing. `ScoreCodeQuality` adds a comment-ratio band and a tests/examples bonus.
2. **Exact dedup across licenses** (`Foundry/Ingest.ts`, `content_hash` index): the same bytes are stored
   once in the training tier regardless of which repo or license they arrived under (`Duplicate` in the
   ingest stats). A `Rejected`/`Raw` copy never blocks a later permissive copy.
3. **Educational value** (`Brain/Data/EducationalScore.ts`): a logistic-regression score over 23 lexical
   features (comment/doc coverage, identifier length, functions per KB, nesting, line length, literal
   ratios, TODO/console.log density, type annotations, error handling, test signals, language). The
   default weights in `Data/EduWeights.json` are a hand-set prior; train real ones from hand labels:

   ```bash
   bun run Scripts/LabelSamples.ts --Count=500        # Runs/Labels/sample.jsonl, label: null -> 0|1
   bun run foundry:train-edu --Labels=Runs/Labels/sample.jsonl   # writes Data/EduWeights.json
   ```

   The score is stored as `quality_score`; documents below `Threshold` are `Rejected` with the two most
   negative features in `reject_reason`. `BackfillLicenses` promotes with the same threshold.

## Spec-driven collection

The corpus is designed before it is collected (Plan: Brain-Strengthening, phase 2). `Data/Spec.json`
(schema: `Foundry/DataSpec.ts`; curation guide: `Data/Spec.md`) names the slices with a `TargetMb`
byte budget each, a hand-picked list of permissive repositories (each with `Slice`, `Include`/`Exclude`
globs and a `Note`), global `Exclusions` (path globs + a 200 B – 50 KB file-size window) and the
training caps.

```bash
bun run foundry:ingest-spec --Status              # slice -> target / collected MB / % / files / repos done
bun run foundry:ingest-spec --DryRun --Slice=go   # what would be fetched (no network, no writes)
bun run foundry:ingest-spec                       # collect every pending entry
bun run foundry:ingest-spec --Repo=honojs/hono --Force
# also: --Spec=<path> (or FOUNDRY_SPEC), --Store=memory|postgres (postgres -> documents_code)
```

- **Provider** — `CreateGitHubRepoListProvider` (`Foundry/GitHubRepoListProvider.ts`) walks the explicit
  list instead of searching. Each repo's license comes from the GitHub API (`NOASSERTION` resolved
  through `SpdxDetector`); a non-permissive repo is skipped before download. Entries of one repo share
  a single tarball download. Files are kept by: global `PathGlobs` → entry `Exclude`/`Include` → the
  file's language must belong to a wanted slice → size window → `IsSubstantiveCodePath` →
  `PrepareRepoFile` (per-file license header + content gate). Rows are `Origin: "web-permissive"`,
  `Source: owner/name`, `Provenance: owner/name@ref:path`.
- **Curation opt-ins** — tests (`test/`, `__tests__/`, `*.test.*`) are kept unless an entry excludes
  them (the spec targets ~10% tests); `docs/`, `site/`, `examples/`, `migrations/` come back only when
  the matching `Include` glob names them. `.md`/`.mdx` files are stored as `Lang: "markdown"` for the
  `docs` slice.
- **Budgets** — `IngestSpec` (`Foundry/SpecIngest.ts`) measures Filtered bytes per slice from the
  store (`RepoSummaries` by tier + lang), skips a repo whose slice is already at `TargetMb`, and
  re-measures after each stored repo (a slice overshoots by at most the repo in progress). Entries
  already in the store are skipped unless `--Force`. One `IngestFailureBudget` bounds the whole run.
