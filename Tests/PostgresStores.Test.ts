// Postgres-backed store behavior that the in-memory contract cannot cover: the schema migration
// (tables + indexes, idempotent), the file-backed checkpoint store (sha256 verification, crash-window
// .bak fallback, legacy in-row migration, delete-removes-file), and the chat store's transactional
// pairs. DB tests self-skip without DATABASE_URL (CI runs them against a real Postgres service). The
// migration gate and checkpoint-name sanitizer are pure and always run.

import { describe, test, expect } from "bun:test";
import postgres from "postgres";
import { randomBytes } from "node:crypto";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { BuildCheckpoint, WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { CreateMigrationGate, DocumentColumnNames, EnsureFoundryDocumentSchema, AllDocumentTables } from "../Foundry/FoundrySchema.ts";
import { PostgresCheckpointStore, SafeCheckpointName } from "../Foundry/PostgresCheckpointStore.ts";
import { PostgresChatStore } from "../Foundry/PostgresChatStore.ts";

const DatabaseUrl = process.env["DATABASE_URL"] ?? "";
const HasDb = DatabaseUrl !== "";
const DbTimeoutMs = 60_000;

function TinyCheckpoint(Step: number): Checkpoint {
  const Config = LoadConfig({ Overrides: { Model: { EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2 } }, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed + Step);
  const Model = new Shahd(Config, Rng.InitRng);
  return BuildCheckpoint(Model, CreateOptimizer(Model.Parameters(), Config), Rng, { Step, Corpus: "test" });
}

describe("CreateMigrationGate", () => {
  test("retries after a failure, shares the in-flight attempt, remembers success", async () => {
    let Calls = 0;
    let FailNext = true;
    const Gate = CreateMigrationGate(async () => {
      Calls++;
      if (FailNext) {
        FailNext = false;
        throw new Error("db down");
      }
    }, "TestGate");
    await expect(Gate()).rejects.toThrow("db down");
    expect(Calls).toBe(1);
    await Promise.all([Gate(), Gate(), Gate()]); // one retry shared by concurrent callers
    expect(Calls).toBe(2);
    await Gate();
    expect(Calls).toBe(2); // success is remembered — no re-migration
  });
});

describe("SafeCheckpointName", () => {
  test("clean names map to themselves; others are sanitized with a collision-proof suffix", () => {
    expect(SafeCheckpointName("Chat-v2.1_final")).toBe("Chat-v2.1_final");
    const A = SafeCheckpointName("a b");
    const B = SafeCheckpointName("a_b");
    expect(B).toBe("a_b");
    expect(A).not.toBe(B);
    expect(A.startsWith("a_b-")).toBe(true);
    for (const Hostile of ["../../etc/passwd", "..", ".", "", "x/y\\z"]) {
      const Safe = SafeCheckpointName(Hostile);
      expect(Safe).toMatch(/^[A-Za-z0-9._-]+$/);
      expect(Safe.startsWith(".")).toBe(false);
      expect(basename(Safe)).toBe(Safe);
    }
  });
});

describe.skipIf(!HasDb)("Foundry schema migration (Postgres)", () => {
  test("EnsureFoundryDocumentSchema is idempotent and gives every table the same indexes", async () => {
    const Sql = postgres(DatabaseUrl, { onnotice: () => {} });
    try {
      const First = await EnsureFoundryDocumentSchema(Sql);
      const Second = await EnsureFoundryDocumentSchema(Sql); // re-run: no error, same result
      expect(Second.Tables).toEqual(First.Tables);
      expect(First.Tables).toEqual(AllDocumentTables());
      for (const Table of First.Tables) {
        const Cols = await Sql<{ column_name: string }[]>`select column_name from information_schema.columns where table_schema = 'public' and table_name = ${Table} order by ordinal_position`;
        expect(Cols.map((C) => C.column_name)).toEqual([...DocumentColumnNames]);
        const Idx = await Sql<{ indexname: string }[]>`select indexname from pg_indexes where schemaname = 'public' and tablename = ${Table}`;
        const Names = new Set(Idx.map((I) => I.indexname));
        for (const What of ["tier", "tier_id", "source", "license", "tier_source", "ingested_at_id"]) expect(Names.has(`${Table}_${What}_idx`)).toBe(true);
        if (First.Trigram) {
          expect(Names.has(`${Table}_content_trgm_idx`)).toBe(true);
          expect(Names.has(`${Table}_provenance_trgm_idx`)).toBe(true);
        }
      }
    } finally {
      await Sql.end();
    }
  }, DbTimeoutMs);
});

describe.skipIf(!HasDb)("PostgresCheckpointStore (Postgres + files)", () => {
  async function WithCheckpointStore(Body: (Store: PostgresCheckpointStore, Name: string, Sql: ReturnType<typeof postgres>) => Promise<void>): Promise<void> {
    const Dir = mkdtempSync(join(tmpdir(), "shahd-ckpt-"));
    const Store = new PostgresCheckpointStore(DatabaseUrl, Dir);
    const Sql = postgres(DatabaseUrl, { onnotice: () => {} });
    const Name = `test ckpt ${randomBytes(4).toString("hex")}`; // a space → exercises name sanitizing
    try {
      await Body(Store, Name, Sql);
    } finally {
      await Sql`DELETE FROM checkpoints WHERE name = ${Name}`;
      await Sql.end();
      await Store.Close();
      rmSync(Dir, { recursive: true, force: true });
    }
  }

  test("Save writes the payload to disk + metadata row; Load verifies and round-trips", async () => {
    await WithCheckpointStore(async (Store, Name, Sql) => {
      const Ckpt = TinyCheckpoint(3);
      await Store.Save(Name, Ckpt, "2026-07-01T00:00:00.000Z");
      const Path = Store.PathFor(Name);
      expect(existsSync(Path)).toBe(true);
      const Rows = await Sql<{ data: string | null; path: string; sha256: string; size_bytes: string }[]>`SELECT data, path, sha256, size_bytes FROM checkpoints WHERE name = ${Name}`;
      expect(Rows[0]?.data).toBeNull();
      expect(Rows[0]?.path).toBe(Path);
      expect(Rows[0]?.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(Number(Rows[0]?.size_bytes)).toBeGreaterThan(0);
      const Loaded = await Store.Load(Name);
      expect(Loaded?.Checksum).toBe(Ckpt.Checksum);
      expect((await Store.List()).find((S) => S.Name === Name)?.Step).toBe(3);
      expect(await Store.Load(`${Name}-absent`)).toBeNull();
    });
  }, DbTimeoutMs);

  test("Load rejects a tampered payload, but accepts the .bak left by a save that crashed mid-way", async () => {
    await WithCheckpointStore(async (Store, Name) => {
      const A = TinyCheckpoint(1);
      await Store.Save(Name, A, "2026-07-01T00:00:00.000Z");
      // Crash window: the file was atomically replaced (old one rotated to .bak) but the row not updated.
      WriteCheckpointObject(Store.PathFor(Name), TinyCheckpoint(2));
      expect((await Store.Load(Name))?.Checksum).toBe(A.Checksum);
      // Plain corruption with no matching backup → loud failure, never wrong weights.
      writeFileSync(Store.PathFor(Name), "{}");
      rmSync(`${Store.PathFor(Name)}.bak`, { force: true });
      await expect(Store.Load(Name)).rejects.toThrow("sha256");
    });
  }, DbTimeoutMs);

  test("a legacy in-row checkpoint is migrated to disk on first Load", async () => {
    await WithCheckpointStore(async (Store, Name, Sql) => {
      const Ckpt = TinyCheckpoint(5);
      await Store.List(); // runs the migration gate so the new columns exist
      await Sql`INSERT INTO checkpoints (name, format_version, data, meta, created_at) VALUES (${Name}, ${Ckpt.FormatVersion}, ${JSON.stringify(Ckpt)}, ${"{}"}, ${"2026-07-01T00:00:00.000Z"})`;
      const Loaded = await Store.Load(Name);
      expect(Loaded?.Checksum).toBe(Ckpt.Checksum);
      const Rows = await Sql<{ data: string | null; path: string | null }[]>`SELECT data, path FROM checkpoints WHERE name = ${Name}`;
      expect(Rows[0]?.data).toBeNull();
      expect(Rows[0]?.path).toBe(Store.PathFor(Name));
      expect(existsSync(Store.PathFor(Name))).toBe(true);
      expect((await Store.Load(Name))?.Checksum).toBe(Ckpt.Checksum); // now served from the file
    });
  }, DbTimeoutMs);

  test("Delete removes the row, the payload file and its .bak", async () => {
    await WithCheckpointStore(async (Store, Name) => {
      await Store.Save(Name, TinyCheckpoint(1), "2026-07-01T00:00:00.000Z");
      await Store.Save(Name, TinyCheckpoint(2), "2026-07-02T00:00:00.000Z"); // creates .bak
      const Path = Store.PathFor(Name);
      expect(existsSync(`${Path}.bak`)).toBe(true);
      await Store.Delete(Name);
      expect(existsSync(Path)).toBe(false);
      expect(existsSync(`${Path}.bak`)).toBe(false);
      expect(await Store.Load(Name)).toBeNull();
      expect((await Store.List()).some((S) => S.Name === Name)).toBe(false);
    });
  }, DbTimeoutMs);
});

describe.skipIf(!HasDb)("PostgresChatStore (Postgres)", () => {
  test("AddMessage bumps the conversation; DeleteConversation removes both sides", async () => {
    const Store = new PostgresChatStore(DatabaseUrl);
    const Id = `test-conv-${randomBytes(4).toString("hex")}`;
    try {
      await Store.CreateConversation(Id, "t", "2026-07-01T00:00:00.000Z");
      await Store.AddMessage(Id, "user", "hi", "2026-07-02T00:00:00.000Z");
      await Store.AddMessage(Id, "assistant", "hello", "2026-07-03T00:00:00.000Z", [{ Step: 1, Kind: "think", Text: "step" }]);
      expect((await Store.GetMessages(Id)).map((M) => M.Content)).toEqual(["hi", "hello"]);
      expect((await Store.ListConversations()).find((C) => C.Id === Id)?.UpdatedAt).toBe("2026-07-03T00:00:00.000Z");
      await Store.DeleteConversation(Id);
      expect(await Store.GetMessages(Id)).toEqual([]);
      expect((await Store.ListConversations()).some((C) => C.Id === Id)).toBe(false);
    } finally {
      await Store.DeleteConversation(Id);
      await Store.Close();
    }
  }, DbTimeoutMs);
});
