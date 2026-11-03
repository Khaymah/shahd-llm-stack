// Copy rows from the legacy single `documents` table into the per-kind tables (Phase 9). Each data
// kind has its own table (documents_<kind>, identical schema) so data types stay physically separate.
// Classification of the EXISTING legacy rows: text-* from oasst -> conversation, text-* from wikipedia
// (or other) -> knowledge, everything else -> code. Dry-run by default (reports the split); --Apply
// first ensures the full schema (every kind table + indexes, same as `bun run foundry:migrate`) and
// then copies rows in (ON CONFLICT DO NOTHING — idempotent). Works on a fresh database too: with no
// legacy table there is simply nothing to copy, but --Apply still creates every kind table. The legacy
// table is KEPT so the move is verifiable and reversible before anything is dropped.
//
//   bun run Scripts/MigrateKinds.ts            # dry run: show the per-kind split
//   bun run Scripts/MigrateKinds.ts --Apply    # ensure tables/indexes and copy rows in

import postgres from "postgres";
import { DataKinds, TableForKind } from "../Foundry/DataKinds.ts";
import type { DataKind } from "../Foundry/DataKinds.ts";
import { DocumentColumnNames, EnsureFoundryDocumentSchema, LegacyDocumentsTableName } from "../Foundry/FoundrySchema.ts";
import { DatabaseUrl } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

const Apply = ReadFlag("--Apply");
const Sql = postgres(DatabaseUrl(), { onnotice: () => {} });
const Legacy = Sql(LegacyDocumentsTableName);
const Columns = Sql([...DocumentColumnNames]);

// The WHERE clause that assigns an existing legacy row to a kind. Mutually exclusive + exhaustive.
const KindWhere: Record<DataKind, ReturnType<typeof Sql> | null> = {
  conversation: Sql`lang like 'text-%' and source like 'oasst%'`,
  knowledge: Sql`lang like 'text-%' and source not like 'oasst%'`,
  code: Sql`lang not like 'text-%'`,
  books: null, // reserved — no legacy rows map here
  web: null,
  instruction: null,
};

console.log(`Migrate ${LegacyDocumentsTableName} -> per-kind tables. Mode: ${Apply ? "APPLY" : "DRY-RUN"}\n`);
try {
  if (Apply) {
    const { Tables, Trigram } = await EnsureFoundryDocumentSchema(Sql);
    console.log(`schema: ${Tables.length} tables + indexes ensured (trigram ${Trigram ? "on" : "off"})`);
  }
  const HasLegacy = await Sql<{ exists: boolean }[]>`select to_regclass(${`public.${LegacyDocumentsTableName}`}) is not null as exists`;
  if (HasLegacy[0]?.exists !== true) {
    console.log(`no legacy \`${LegacyDocumentsTableName}\` table — nothing to copy.${Apply ? "" : " Re-run with --Apply (or `bun run foundry:migrate`) to create the per-kind tables."}`);
  } else {
    let TotalRows = 0;
    for (const Kind of DataKinds) {
      const Where = KindWhere[Kind];
      if (Where === null) {
        console.log(`${Kind.padEnd(13)} — reserved (no legacy rows map here)`);
        continue;
      }
      const CountRows = await Sql<{ n: number }[]>`select count(*)::int n from ${Legacy} where ${Where}`;
      const N = CountRows[0]?.n ?? 0;
      const Table = Sql(TableForKind(Kind));
      if (Apply) {
        // Explicit column list on both sides: never relies on the two tables' physical column order.
        const Moved = await Sql`insert into ${Table} (${Columns}) select ${Columns} from ${Legacy} where ${Where} on conflict (id) do nothing`;
        console.log(`${Kind.padEnd(13)} ${String(N).padStart(7)} rows -> ${TableForKind(Kind)} (${Moved.count} inserted)`);
      } else {
        console.log(`${Kind.padEnd(13)} ${String(N).padStart(7)} rows -> ${TableForKind(Kind)}`);
      }
      TotalRows += N;
    }
    console.log(`\n${Apply ? "APPLIED" : "DRY-RUN"}: ${TotalRows} rows across kinds. Legacy \`${LegacyDocumentsTableName}\` table kept (drop it only after verifying).`);
    if (!Apply) console.log("Re-run with --Apply to create the per-kind tables and copy the rows.");
  }
} finally {
  await Sql.end();
}
