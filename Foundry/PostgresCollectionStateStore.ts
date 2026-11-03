// Postgres-backed CollectionStateStore: the collection ledger lives in the SAME database as the corpus.
// Accepts a shared postgres client (so it reuses FoundryStores' single connection pool) or its own URL.
// The table is created on first use (idempotent), matching PostgresChatStore/CheckpointStore.

import postgres from "postgres";
import type { CollectionStateStore, CollectionState } from "./CollectionState.ts";
import type { DataKind } from "./DataKinds.ts";
import { CreateMigrationGate } from "./FoundrySchema.ts";

type Row = { source_key: string; kind: string; cursor: string; collected: number; exhausted: boolean; updated_at: string };

function FromRow(R: Row): CollectionState {
  return { SourceKey: R.source_key, Kind: R.kind as DataKind, Cursor: R.cursor, Collected: Number(R.collected), Exhausted: R.exhausted, UpdatedAt: R.updated_at };
}

export class PostgresCollectionStateStore implements CollectionStateStore {
  private Sql: ReturnType<typeof postgres>;
  private OwnsSql: boolean;
  // Lazy, self-healing migration gate (see CreateMigrationGate): retried on the next call if it failed.
  private Ready: () => Promise<void>;

  constructor(UrlOrSql: string | ReturnType<typeof postgres>) {
    this.OwnsSql = typeof UrlOrSql === "string";
    this.Sql = typeof UrlOrSql === "string" ? postgres(UrlOrSql, { onnotice: () => {} }) : UrlOrSql;
    this.Ready = CreateMigrationGate(() => this.Migrate(), "PostgresCollectionStateStore");
  }

  private async Migrate(): Promise<void> {
    await this.Sql`CREATE TABLE IF NOT EXISTS collection_state (
      source_key text PRIMARY KEY,
      kind       text NOT NULL,
      cursor     text NOT NULL DEFAULT '{}',
      collected  integer NOT NULL DEFAULT 0,
      exhausted  boolean NOT NULL DEFAULT false,
      updated_at text NOT NULL
    )`;
  }

  async Get(SourceKey: string): Promise<CollectionState | null> {
    await this.Ready();
    const Rows = (await this.Sql`SELECT source_key, kind, cursor, collected, exhausted, updated_at FROM collection_state WHERE source_key = ${SourceKey}`) as unknown as Row[];
    return Rows[0] ? FromRow(Rows[0]) : null;
  }

  async Upsert(State: CollectionState): Promise<void> {
    await this.Ready();
    await this.Sql`INSERT INTO collection_state (source_key, kind, cursor, collected, exhausted, updated_at)
      VALUES (${State.SourceKey}, ${State.Kind}, ${State.Cursor}, ${State.Collected}, ${State.Exhausted}, ${State.UpdatedAt})
      ON CONFLICT (source_key) DO UPDATE SET
        kind = EXCLUDED.kind, cursor = EXCLUDED.cursor, collected = EXCLUDED.collected,
        exhausted = EXCLUDED.exhausted, updated_at = EXCLUDED.updated_at`;
  }

  async All(): Promise<CollectionState[]> {
    await this.Ready();
    const Rows = (await this.Sql`SELECT source_key, kind, cursor, collected, exhausted, updated_at FROM collection_state ORDER BY updated_at DESC`) as unknown as Row[];
    return Rows.map(FromRow);
  }

  /** Close the pool — only if this store opened it (a shared client is owned by its creator). */
  async Close(): Promise<void> {
    if (this.OwnsSql) await this.Sql.end();
  }
}
