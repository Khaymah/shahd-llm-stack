// Drizzle schema + DDL for the Data Foundry's Postgres store (M3b). One `documents_<kind>` table per
// data kind (plus the legacy `documents` table), all with the identical schema. The embedding is
// stored as JSONB (number[]), which works on ANY Postgres with no extension — portable and verifiable
// on a stock server. Similarity ranking is done in-app (cosine); for large-scale vector search, swap
// the column to pgvector (docker-compose ships the pgvector image) and push the ranking into SQL.
//
// The DDL helpers below (EnsureDocumentTable / EnsureDocumentIndexes) are the ONE home of the table
// definition and its indexes: FoundryMigrate, MigrateKinds and the Postgres tests all call them, so a
// per-kind table always ends up with exactly the same columns and indexes as every other one, in any
// run order. CreateMigrationGate is the shared lazy "migrate once, retry on failure" gate the small
// self-migrating stores (chat / checkpoints / collection ledger) use.

import { pgTable, text, integer, real, jsonb } from "drizzle-orm/pg-core";
import type postgres from "postgres";
import { DataKinds, TableForKind } from "./DataKinds.ts";

export const EmbeddingDimensions = 256;

// One document table's columns. Factory (not a single fixed table) so each data kind can have its own
// physically-separate table (documents_code, documents_conversation, …) with the identical schema.
export function DocumentsTable(Name: string) {
  return pgTable(Name, {
    id: text("id").primaryKey(),
    tier: text("tier").notNull(),
    origin: text("origin").notNull(),
    source: text("source").notNull(),
    license: text("license").notNull(),
    lang: text("lang").notNull(),
    content: text("content").notNull(),
    bytes: integer("bytes").notNull(),
    quality: real("quality_score").notNull(),
    contentHash: text("content_hash").notNull(),
    embedding: jsonb("embedding").$type<number[]>().notNull(),
    rejectReason: text("reject_reason"),
    provenance: text("provenance").notNull(),
    ingestedAt: text("ingested_at").notNull(),
  });
}

// The default/legacy table (pre-per-kind). Kept for back-compat; kind stores use DocumentsTable(name).
export const LegacyDocumentsTableName = "documents";
export const Documents = DocumentsTable(LegacyDocumentsTableName);

export type DocumentTable = ReturnType<typeof DocumentsTable>;
export type DocumentRow = typeof Documents.$inferSelect;
export type DocumentInsert = typeof Documents.$inferInsert;

// The physical column order of every document table. Row copies between tables (MigrateKinds) name
// these explicitly instead of `select *`, so a table whose columns were created in a different order
// can never silently receive shifted values.
export const DocumentColumnNames = [
  "id",
  "tier",
  "origin",
  "source",
  "license",
  "lang",
  "content",
  "bytes",
  "quality_score",
  "content_hash",
  "embedding",
  "reject_reason",
  "provenance",
  "ingested_at",
] as const;

type SqlClient = ReturnType<typeof postgres>;

/** Create one document table (idempotent). Self-contained DDL — no `LIKE documents`, so a fresh
 *  database can get every per-kind table without the legacy table existing first. */
export async function EnsureDocumentTable(Sql: SqlClient, Table: string): Promise<void> {
  await Sql`
    CREATE TABLE IF NOT EXISTS ${Sql(Table)} (
      id            text PRIMARY KEY,
      tier          text NOT NULL,
      origin        text NOT NULL,
      source        text NOT NULL,
      license       text NOT NULL,
      lang          text NOT NULL,
      content       text NOT NULL,
      bytes         integer NOT NULL,
      quality_score real NOT NULL,
      content_hash  text NOT NULL,
      embedding     jsonb NOT NULL,
      reject_reason text,
      provenance    text NOT NULL,
      ingested_at   text NOT NULL
    )`;
}

/** Install pg_trgm once per database. Returns false (with a warning) when the extension is not
 *  available or the role may not create it — the trigram indexes are then skipped, and search simply
 *  falls back to a sequential scan (same results, slower). */
export async function EnsureTrigramExtension(Sql: SqlClient): Promise<boolean> {
  try {
    await Sql`CREATE EXTENSION IF NOT EXISTS pg_trgm`;
    return true;
  } catch (Caught) {
    console.warn(`EnsureTrigramExtension: pg_trgm unavailable — search indexes skipped (${(Caught as Error).message})`);
    return false;
  }
}

/**
 * Create every index a document table needs (idempotent: CREATE INDEX IF NOT EXISTS). Index names are
 * `<table>_<what>_idx`, which is also the name Postgres generated for the tier index when older
 * per-kind tables were cloned with `LIKE documents INCLUDING ALL` — so re-running against those is a
 * no-op instead of a duplicate index.
 *   tier / (tier, id "C")    — ByTier (training reads, ORDER BY id COLLATE "C") and keyset paging
 *                              within a tier; byte-order collation so the order never depends on the
 *                              server locale and matches the in-memory store exactly
 *   source / (tier, source)  — per-repo rollups, skip-learned, license backfill
 *   license                  — facet filter + NOASSERTION backfill
 *   content_hash             — exact dedup across licenses/origins at ingest (ExistingContentHashes)
 *   (ingested_at, id) DESC   — the browse order (newest first, id tie-break) without a sort
 *   trigram on content/provenance — the case-insensitive substring search (only with pg_trgm)
 */
export async function EnsureDocumentIndexes(Sql: SqlClient, Table: string, WithTrigram: boolean): Promise<void> {
  const Name = (What: string) => Sql(`${Table}_${What}_idx`);
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("tier")} ON ${Sql(Table)} (tier)`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("tier_id")} ON ${Sql(Table)} (tier, id COLLATE "C")`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("source")} ON ${Sql(Table)} (source)`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("license")} ON ${Sql(Table)} (license)`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("tier_source")} ON ${Sql(Table)} (tier, source)`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("content_hash")} ON ${Sql(Table)} (content_hash)`;
  await Sql`CREATE INDEX IF NOT EXISTS ${Name("ingested_at_id")} ON ${Sql(Table)} (ingested_at DESC, id DESC)`;
  if (!WithTrigram) return;
  // Per-index guard: a trigram build can fail on its own (e.g. disk/maintenance_work_mem) without
  // invalidating the B-tree indexes above — warn and keep going rather than aborting the migration.
  for (const Column of ["content", "provenance"] as const) {
    try {
      await Sql`CREATE INDEX IF NOT EXISTS ${Name(`${Column}_trgm`)} ON ${Sql(Table)} USING gin (${Sql(Column)} gin_trgm_ops)`;
    } catch (Caught) {
      console.warn(`EnsureDocumentIndexes: trigram index on ${Table}.${Column} skipped (${(Caught as Error).message})`);
    }
  }
}

/** Every document table the Foundry uses: the legacy `documents` table (still the default for the
 *  single-store CLI scripts) plus one `documents_<kind>` table per data kind. */
export function AllDocumentTables(): string[] {
  return [LegacyDocumentsTableName, ...DataKinds.map(TableForKind)];
}

/** Bring a database to the full document schema — every table AND every index — idempotently, in any
 *  order relative to MigrateKinds. Schema only: no rows are copied (that is MigrateKinds --Apply), so
 *  re-running it can never resurrect rows deleted from a kind table. Returns what it ensured. */
export async function EnsureFoundryDocumentSchema(Sql: SqlClient): Promise<{ Tables: string[]; Trigram: boolean }> {
  const Trigram = await EnsureTrigramExtension(Sql);
  const Tables = AllDocumentTables();
  for (const Table of Tables) {
    await EnsureDocumentTable(Sql, Table);
    await EnsureDocumentIndexes(Sql, Table, Trigram);
  }
  return { Tables, Trigram };
}

/**
 * A lazy, self-healing migration gate. The returned function runs Migrate on first use and shares the
 * in-flight attempt with concurrent callers; a SUCCESSFUL migration is remembered forever, while a
 * FAILED one is forgotten so the next call retries (a DB that was down at startup recovers without a
 * process restart). The failure is still thrown to the caller that awaited it, so it surfaces where
 * the query would have failed anyway.
 */
export function CreateMigrationGate(Migrate: () => Promise<void>, Label: string): () => Promise<void> {
  let Pending: Promise<void> | null = null;
  return () => {
    if (Pending === null) {
      const Attempt = Migrate();
      Pending = Attempt;
      Attempt.catch((Caught: unknown) => {
        if (Pending === Attempt) Pending = null; // forget the failure → the next call retries
        console.warn(`${Label}: migration failed (will retry on next use): ${(Caught as Error).message}`);
      });
    }
    return Pending;
  };
}
