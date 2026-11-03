// One-off maintenance: retro-clean documents ingested BEFORE the license-header strip landed.
// Scans every per-kind table (documents_<kind>) for rows that still carry a leading LICENSE/COPYRIGHT
// banner, strips it, and re-ingests the cleaned content through the normal path (IngestDocuments —
// which recomputes id = hash(origin + license + content), content hash, embedding, and bytes) into the
// SAME kind table, then deletes the old row. Because the id is content-derived, a cleaned row gets a
// NEW primary key, so the stale dirty row must be removed explicitly — and it is removed ONLY after
// the cleaned row is confirmed stored (inserted, or already present). A failed re-ingest keeps the
// dirty row, so nothing is ever lost; re-running retries it. Idempotent: a second run finds nothing.
//
//   bun run Scripts/NormalizeCorpus.ts            # migrate in place
//   bun run Scripts/NormalizeCorpus.ts --DryRun   # count only, no writes

import postgres from "postgres";
import { IngestDocuments } from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, Origin } from "../Foundry/FoundryBarrel.ts";
import { StripLicenseHeader } from "../Foundry/ContentNormalizer.ts";
import { DataKinds, TableForKind } from "../Foundry/DataKinds.ts";
import type { DocumentStore } from "../Foundry/DocumentStore.ts";
import { DatabaseUrl, ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

type DirtyRow = { id: string; content: string; source: string; license: string; lang: string; provenance: string; origin: string; ingested_at: string };
type KindTally = { Scanned: number; Migrated: number; Unchanged: number; Failed: number };

if ((process.env["DATABASE_URL"] ?? "") === "") {
  throw new Error("NormalizeCorpus needs DATABASE_URL (this migration only applies to the Postgres store)");
}
const DryRun = ReadFlag("--DryRun");
const BatchSize = 300;

const Stores = ResolveFoundryStores();
// A separate small client for the content-prefilter scan (a SQL-side ILIKE that the store API does not
// expose); every write still goes through the store.
const Sql = postgres(DatabaseUrl(), { max: 1, onnotice: () => {} });

// Rows that plausibly begin with a license banner (SQL pre-filter; the real decision is made in JS
// by StripLicenseHeader, which only strips a genuine LEADING license block). Keyset-paged by id.
function SelectDirty(Table: string, AfterId: string): Promise<DirtyRow[]> {
  return Sql<DirtyRow[]>`
    select id, content, source, license, lang, provenance, origin, ingested_at
    from ${Sql(Table)}
    where id collate "C" > ${AfterId}
      and ( (content like '/*%' and (content ilike '%copyright%' or content ilike '%licensed under%' or content ilike '%SPDX-License%'))
         or ((content like '//%' or content like '#%') and (content ilike '%copyright%' or content ilike '%licensed under%')) )
    order by id collate "C"
    limit ${BatchSize}`;
}

async function TableExists(Table: string): Promise<boolean> {
  const Rows = await Sql<{ exists: boolean }[]>`select to_regclass(${`public.${Table}`}) is not null as exists`;
  return Rows[0]?.exists === true;
}

async function NormalizeTable(Table: string, Store: DocumentStore): Promise<KindTally> {
  const Tally: KindTally = { Scanned: 0, Migrated: 0, Unchanged: 0, Failed: 0 };
  let AfterId = "";
  for (;;) {
    const Rows = await SelectDirty(Table, AfterId);
    if (Rows.length === 0) break;
    for (const Row of Rows) {
      AfterId = Row.id; // advance the cursor past every scanned row (incl. unchanged/failed ones)
      Tally.Scanned++;
      const Cleaned = StripLicenseHeader(Row.content);
      if (Cleaned === Row.content) {
        Tally.Unchanged++; // copyright lives in the body, not a leading banner — leave it alone
        continue;
      }
      if (DryRun) {
        Tally.Migrated++;
        continue;
      }
      const Input: SourceInput = { Source: Row.source, License: Row.license, Lang: Row.lang, Content: Cleaned, Provenance: Row.provenance, Origin: Row.origin as Origin };
      const Stats = await IngestDocuments([Input], Store, Row.ingested_at); // store the cleaned row under its new id
      // Confirmed = the cleaned document is now in the table (newly inserted, or an identical clean copy
      // was already there). Anything else (a failed upsert) keeps the dirty row for a later re-run.
      if (Stats.Failed > 0 || Stats.New + Stats.Duplicate !== 1) {
        Tally.Failed++;
        console.warn(`  kept dirty row ${Row.id} (${Row.provenance}): cleaned copy was not stored`);
        continue;
      }
      await Store.DeleteById(Row.id); // then drop the stale dirty row
      Tally.Migrated++;
    }
    console.log(`  [${Table}] scanned=${Tally.Scanned}  migrated=${Tally.Migrated}  unchanged=${Tally.Unchanged}  failed=${Tally.Failed}`);
  }
  return Tally;
}

let Migrated = 0;
let Scanned = 0;
let Unchanged = 0;
let Failed = 0;
try {
  for (const Kind of DataKinds) {
    const Table = TableForKind(Kind);
    if (!(await TableExists(Table))) {
      console.log(`[${Kind}] ${Table} missing — skipped (run \`bun run foundry:migrate\`)`);
      continue;
    }
    const Tally = await NormalizeTable(Table, Stores.Kind(Kind));
    Migrated += Tally.Migrated;
    Scanned += Tally.Scanned;
    Unchanged += Tally.Unchanged;
    Failed += Tally.Failed;
  }
  const Total = (await Stores.Stats()).reduce((Acc, K) => Acc + K.Count, 0);
  console.log(`${DryRun ? "[dry-run] would migrate" : "migrated"} ${Migrated} rows (scanned ${Scanned}, ${Unchanged} left as body-only, ${Failed} failed and kept). Corpus now ${Total} docs.`);
} finally {
  await Sql.end();
  await Stores.Close();
}
process.exit(Failed > 0 ? 1 : 0);
