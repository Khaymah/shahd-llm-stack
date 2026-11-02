// Bring the Foundry database to its full schema in ONE idempotent step: the legacy `documents` table,
// every per-kind `documents_<kind>` table, and all of their indexes (B-tree for the filters/orders the
// stores use, plus pg_trgm GIN search indexes when the extension is available — skipped with a warning
// otherwise). Portable: JSONB embedding, runs on any Postgres. Safe to re-run at any time and in any
// order relative to MigrateKinds; it never copies or deletes rows.
//   docker compose up -d   (or point DATABASE_URL at an existing Postgres)
//   bun run foundry:migrate
// To move rows out of a pre-Phase-9 single `documents` table, follow up with
//   bun run Scripts/MigrateKinds.ts --Apply

import postgres from "postgres";
import { EnsureFoundryDocumentSchema } from "../Foundry/FoundrySchema.ts";
import { DatabaseUrl } from "./FoundryEnv.ts";

const Sql = postgres(DatabaseUrl(), { onnotice: () => {} });
try {
  const { Tables, Trigram } = await EnsureFoundryDocumentSchema(Sql);
  console.log(`migrated: ${Tables.length} document tables ready (${Tables.join(", ")}); trigram search indexes: ${Trigram ? "on" : "off (pg_trgm unavailable)"}`);
} finally {
  await Sql.end();
}
