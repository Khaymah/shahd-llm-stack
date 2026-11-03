// Spec-driven ingestion (Plan: Brain-Strengthening, phase 2): collect exactly the repositories that
// Data/Spec.json names, slice by slice, until each slice reaches its TargetMb. See Data/Spec.md.
//   bun run foundry:ingest-spec                        # collect everything still pending
//   bun run foundry:ingest-spec --Status               # slice -> target / collected / % / repos done
//   bun run foundry:ingest-spec --DryRun --Slice=ts    # list what would be fetched, fetch nothing
//   bun run foundry:ingest-spec --Repo=honojs/hono --Force
// Flags: --Spec=<path> (default Data/Spec.json, or FOUNDRY_SPEC), --Slice=<slice>, --Repo=<owner/name>,
// --Status, --DryRun, --Force (re-collect entries already in the store), --Store=memory|postgres.
// With Postgres the rows land in the `code` kind table (documents_code), where training reads them.

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { LoadDataSpec } from "../Foundry/DataSpec.ts";
import { IngestSpec, MeasureSpec, PlanSpec, RenderSpecPlan, RenderSpecStatus, ValidateSpecFilter } from "../Foundry/SpecIngest.ts";
import type { SpecFilter } from "../Foundry/SpecIngest.ts";
import { IngestAbortError } from "../Foundry/Ingest.ts";
import { DataSpecPath, GitHubToken, ResolveFoundryStores, ResolveStore } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";

const SpecPath = DataSpecPath();
const Filter: SpecFilter = {
  Slice: ReadArg("--Slice=", "") || undefined,
  Repo: ReadArg("--Repo=", "") || undefined,
  Force: ReadFlag("--Force"),
};
const Status = ReadFlag("--Status");
const DryRun = ReadFlag("--DryRun");

// ResolveStore decides memory vs postgres; for postgres we write to the per-kind `code` table (the
// legacy `documents` table it would otherwise open is no longer read by training or the dashboard).
const { Store: Resolved, Kind } = ResolveStore();
const Stores = Kind === "postgres" ? ResolveFoundryStores() : null;
const Store = Stores?.Kind("code") ?? Resolved;

let ExitCode = 0;
try {
  const Spec = LoadDataSpec(SpecPath); // throws a readable Zod/JSON error for a malformed spec
  ValidateSpecFilter(Spec, Filter);
  console.log(`spec=${SpecPath} (v${Spec.Version}, ${Spec.Repos.length} entries) | store=${Kind}${Stores !== null ? " (documents_code)" : ""}\n`);
  if (Status) {
    console.log(RenderSpecStatus(await MeasureSpec(Spec, Store)));
  } else if (DryRun) {
    console.log(RenderSpecPlan(PlanSpec(Spec, await MeasureSpec(Spec, Store), Filter)));
  } else {
    const Config = LoadConfig({ UseCli: true, UseEnv: false });
    if (!Config.Data.WebEnabled) {
      console.log("web ingestion disabled (Config.Data.WebEnabled=false) — nothing fetched.");
    } else {
      const Token = GitHubToken();
      if (Token === undefined) console.log("note: no GITHUB_TOKEN — the unauthenticated API allows ~60 requests/hour (2 per repo).\n");
      const Result = await IngestSpec(Spec, Store, { ...Filter, IngestedAt: new Date().toISOString(), EmbeddingDim: Config.Data.EmbeddingDim, Provider: { Token } });
      const S = Result.Stats;
      console.log(`\nstored ${S.Ingested} files (new ${S.New}, duplicate ${S.Duplicate}, failed ${S.Failed}) -> Filtered=${S.ByTier.Filtered} Raw=${S.ByTier.Raw} Rejected=${S.ByTier.Rejected}\n`);
      console.log(RenderSpecStatus(Result.After));
    }
  }
} catch (Caught) {
  ExitCode = 1;
  if (Caught instanceof IngestAbortError) console.error(`aborted: ${Caught.message} (partial: ${Caught.Stats.Ingested} stored in the failing repo)`);
  else console.error(`ingest-spec failed: ${(Caught as Error).message}`);
} finally {
  await Stores?.Close();
}
process.exit(ExitCode);
