// Model checkpoints (weights + optimizer + RNG + config + tokenizer): the PAYLOAD lives on disk as
// `<CheckpointDir>/<SafeName>.ckpt` (written atomically by Brain's WriteCheckpointObject), and Postgres
// keeps only a small METADATA row per name — format version, meta (params/arch/…), file path, sha256
// and size. Storing the payload as one JSON TEXT cell capped models at ~33M params (≈32 B/param vs
// Postgres' 1 GB field limit) and pushed every save through the DB; the file + row split removes both.
//
// Local-only design: the trainer subprocess and the dashboard run on the same machine and share the
// directory. The row stores the ABSOLUTE path, so processes with different working directories agree.
// Load verifies the file against the row's sha256 before parsing (a swapped/truncated file fails loudly).
//
// Legacy rows (pre-split) still carry the payload in the nullable `data` column. They are migrated
// lazily and exactly once: Load writes the payload to disk, records path/sha256/size and NULLs `data`;
// Save simply overwrites the row with the file-backed form.

import postgres from "postgres";
import { createHash } from "node:crypto";
import { createReadStream, readFileSync, rmSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { ParseCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import { WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { CreateMigrationGate } from "./FoundrySchema.ts";

/** Where checkpoint payload files live unless FOUNDRY_CHECKPOINT_DIR says otherwise. */
export const DefaultCheckpointDir = "Checkpoints";
export const CheckpointFileExtension = ".ckpt";
// Length of the name-hash suffix that keeps two sanitized names ("a b" / "a_b") from sharing a file.
const NameHashChars = 8;

/** The checkpoint payload directory: FOUNDRY_CHECKPOINT_DIR, else DefaultCheckpointDir (relative to the
 *  working directory). The one home for this setting; Scripts/FoundryEnv re-exports it for scripts. */
export function ResolveCheckpointDir(): string {
  const FromEnv = process.env["FOUNDRY_CHECKPOINT_DIR"];
  return FromEnv !== undefined && FromEnv.trim() !== "" ? FromEnv.trim() : DefaultCheckpointDir;
}

/** A filesystem-safe file stem for a checkpoint name. Names already made of [A-Za-z0-9._-] map to
 *  themselves; anything else is sanitized AND suffixed with a short hash of the original name, so
 *  distinct names can never collide on one file (and no name can escape the directory). */
export function SafeCheckpointName(Name: string): string {
  const Sanitized = Name.replace(/[^A-Za-z0-9._-]/g, "_");
  const Clean = Sanitized === Name && Name !== "" && !/^\.+$/.test(Name);
  if (Clean) return Name;
  return `${Sanitized.replace(/^\.+/, "_")}-${createHash("sha256").update(Name).digest("hex").slice(0, NameHashChars)}`;
}

// Format = "chat" (SFT/agent model) or "base" (pretrained autocomplete); Step = training steps done
// so far (for the dashboard's "resume/extend from step N"). Embed/Layers/Heads/Block are the exact
// architecture — the dashboard's "Resume training" prefills them so a re-run matches the checkpoint's
// arch exactly (a NumHeads mismatch would otherwise silently retrain from scratch). All from the Meta.
// Batch/Lr/MaxSteps carry the TRAINING SEMANTICS a resumed run inherits — the dashboard prefills
// (and locks) them so what the form shows matches what the server will actually do.
export type CheckpointSummary = { Name: string; CreatedAt: string; Params: number; Vocab: number; Arch: string; Corpus: string; Format: string; Step: number; Embed: number; Layers: number; Heads: number; Block: number; Batch: number; Lr: number; MaxSteps: number };

type MetaRow = { name: string; created_at: string; meta: string };
type LocatorRow = { path: string | null; sha256: string | null; has_legacy: boolean };
type LegacyRow = { data: string | null };
type StoredMeta = { params: number; vocab: number; arch: string; corpus: string; format: string; step: number; embed: number; layers: number; heads: number; block: number; batch: number; lr: number; maxSteps: number };
type StoredFile = { Path: string; Sha256: string; SizeBytes: number };

function MetaOf(Ckpt: Checkpoint): StoredMeta {
  const Meta = Ckpt.Meta as Record<string, unknown>;
  const M = Ckpt.Config.Model;
  const Corpus = String(Meta["Corpus"] ?? "");
  // Prefer the explicit Meta.Format; infer from the corpus tag for older checkpoints that predate it.
  const Format = typeof Meta["Format"] === "string" ? (Meta["Format"] as string) : Corpus.includes("sft") ? "chat" : "base";
  return {
    params: Ckpt.Params.reduce((Acc, P) => Acc + P.Rows * P.Cols, 0),
    vocab: M.VocabSize,
    arch: `emb${M.EmbedDim} L${M.NumLayers} h${M.NumHeads} ctx${M.BlockSize}`,
    corpus: Corpus,
    format: Format,
    step: Number(Meta["Step"] ?? 0),
    embed: M.EmbedDim,
    layers: M.NumLayers,
    heads: M.NumHeads,
    block: M.BlockSize,
    batch: Ckpt.Config.Training.BatchSize,
    lr: Ckpt.Config.Optimizer.LearningRate,
    maxSteps: Ckpt.Config.Schedule.MaxSteps,
  };
}

// Streamed sha256 of a file (constant memory, whatever the checkpoint size).
async function HashFile(Path: string): Promise<string> {
  const Hash = createHash("sha256");
  for await (const Chunk of createReadStream(Path)) Hash.update(Chunk as Buffer);
  return Hash.digest("hex");
}

// Read a payload file and accept it only if its sha256 matches; otherwise say why it was refused.
function ReadIfMatches(Path: string, Sha256: string): { Bytes: Buffer | null; Problem: string } {
  let Bytes: Buffer;
  try {
    Bytes = readFileSync(Path);
  } catch (Caught) {
    return { Bytes: null, Problem: `is unreadable (${(Caught as Error).message})` };
  }
  const Actual = createHash("sha256").update(Bytes).digest("hex");
  return Actual === Sha256 ? { Bytes, Problem: "" } : { Bytes: null, Problem: `does not match its recorded sha256 (got ${Actual.slice(0, 12)}…)` };
}

// Best-effort removal of a payload file and the `.bak` copy WriteCheckpointObject rotates beside it.
function RemovePayload(Path: string): void {
  for (const Target of [Path, `${Path}.bak`]) {
    try {
      rmSync(Target, { force: true });
    } catch (Caught) {
      console.warn(`PostgresCheckpointStore: could not remove ${Target}: ${(Caught as Error).message}`);
    }
  }
}

export class PostgresCheckpointStore {
  private Sql: ReturnType<typeof postgres>;
  private Dir: string;
  // Lazy, self-healing migration gate: retried on the next call if the DB was unavailable.
  private Ready: () => Promise<void>;

  /** Dir defaults to ResolveCheckpointDir() (FOUNDRY_CHECKPOINT_DIR or Checkpoints/). */
  constructor(Url: string, Dir: string = ResolveCheckpointDir()) {
    this.Sql = postgres(Url, { onnotice: () => {} }); // silence idempotent CREATE-IF-NOT-EXISTS notices that spammed the train log
    this.Dir = resolve(Dir);
    this.Ready = CreateMigrationGate(() => this.Migrate(), "PostgresCheckpointStore");
  }

  private async Migrate(): Promise<void> {
    // `data` is nullable and kept only so legacy rows (payload in-row) can still be read + migrated.
    await this.Sql`CREATE TABLE IF NOT EXISTS checkpoints (name TEXT PRIMARY KEY, format_version INT NOT NULL, data TEXT, meta TEXT NOT NULL, created_at TEXT NOT NULL)`;
    await this.Sql`ALTER TABLE checkpoints ALTER COLUMN data DROP NOT NULL`;
    await this.Sql`ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS path TEXT`;
    await this.Sql`ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS sha256 TEXT`;
    await this.Sql`ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS size_bytes BIGINT`;
  }

  /** Absolute payload path for a checkpoint name. */
  PathFor(Name: string): string {
    return join(this.Dir, `${SafeCheckpointName(Name)}${CheckpointFileExtension}`);
  }

  // Atomically write the payload file, then fingerprint what actually landed on disk.
  private async WritePayload(Name: string, Ckpt: Checkpoint): Promise<StoredFile> {
    const Path = this.PathFor(Name);
    WriteCheckpointObject(Path, Ckpt);
    return { Path, Sha256: await HashFile(Path), SizeBytes: statSync(Path).size };
  }

  async Save(Name: string, Ckpt: Checkpoint, CreatedAt: string): Promise<void> {
    await this.Ready();
    // File first, row second: a crash in between leaves the PREVIOUS row describing the previous
    // payload, which survives as `.bak` and is what Load falls back to — never a row claiming a
    // payload that was never written.
    const File = await this.WritePayload(Name, Ckpt);
    const Meta = JSON.stringify(MetaOf(Ckpt));
    await this.Sql`
      INSERT INTO checkpoints (name, format_version, data, meta, path, sha256, size_bytes, created_at)
      VALUES (${Name}, ${Ckpt.FormatVersion}, NULL, ${Meta}, ${File.Path}, ${File.Sha256}, ${File.SizeBytes}, ${CreatedAt})
      ON CONFLICT (name) DO UPDATE SET format_version = EXCLUDED.format_version, data = NULL, meta = EXCLUDED.meta,
        path = EXCLUDED.path, sha256 = EXCLUDED.sha256, size_bytes = EXCLUDED.size_bytes, created_at = EXCLUDED.created_at`;
  }

  async Load(Name: string): Promise<Checkpoint | null> {
    await this.Ready();
    const Rows = (await this.Sql`SELECT path, sha256, data IS NOT NULL AS has_legacy FROM checkpoints WHERE name = ${Name} LIMIT 1`) as unknown as LocatorRow[];
    const Row = Rows[0];
    if (Row === undefined) return null;
    if (Row.path === null || Row.sha256 === null) {
      if (!Row.has_legacy) throw new Error(`PostgresCheckpointStore: checkpoint "${Name}" has neither a payload file nor legacy in-row data`);
      return this.MigrateLegacy(Name);
    }
    // The main file normally matches. If a save crashed after the atomic file replace but before the
    // row update, the row still describes the PREVIOUS payload — which WriteCheckpointObject kept as
    // `.bak` — so that copy is accepted when (and only when) its hash matches the row.
    const Main = ReadIfMatches(Row.path, Row.sha256);
    if (Main.Bytes !== null) return ParseCheckpoint(Main.Bytes.toString("utf8"));
    const Backup = ReadIfMatches(`${Row.path}.bak`, Row.sha256);
    if (Backup.Bytes !== null) {
      console.warn(`PostgresCheckpointStore: "${Name}" main payload ${Main.Problem}; loaded the matching .bak copy`);
      return ParseCheckpoint(Backup.Bytes.toString("utf8"));
    }
    throw new Error(`PostgresCheckpointStore: payload for "${Name}" at ${Row.path} ${Main.Problem} (expected sha256 ${Row.sha256.slice(0, 12)}…) — corrupt, truncated, overwritten, or deleted`);
  }

  // One-time move of a legacy in-row payload to disk. The parsed checkpoint is returned even if the
  // write-back fails (e.g. read-only disk) — loading must not regress for a migration side effect.
  private async MigrateLegacy(Name: string): Promise<Checkpoint | null> {
    const Rows = (await this.Sql`SELECT data FROM checkpoints WHERE name = ${Name} LIMIT 1`) as unknown as LegacyRow[];
    const Data = Rows[0]?.data ?? null;
    if (Data === null) return null; // deleted (or migrated by another process) between the two reads
    const Ckpt = ParseCheckpoint(Data);
    try {
      const File = await this.WritePayload(Name, Ckpt);
      await this.Sql`
        UPDATE checkpoints SET data = NULL, path = ${File.Path}, sha256 = ${File.Sha256}, size_bytes = ${File.SizeBytes}
        WHERE name = ${Name} AND data IS NOT NULL`;
      console.log(`PostgresCheckpointStore: migrated legacy checkpoint "${Name}" to ${File.Path} (${File.SizeBytes} bytes)`);
    } catch (Caught) {
      console.warn(`PostgresCheckpointStore: legacy checkpoint "${Name}" loaded but not migrated to disk: ${(Caught as Error).message}`);
    }
    return Ckpt;
  }

  async List(): Promise<CheckpointSummary[]> {
    await this.Ready();
    const Rows = (await this.Sql`SELECT name, created_at, meta FROM checkpoints ORDER BY created_at DESC`) as unknown as MetaRow[];
    return Rows.map((R) => {
      const M = JSON.parse(R.meta) as Partial<StoredMeta>;
      const Corpus = M.corpus ?? "";
      return {
        Name: R.name,
        CreatedAt: R.created_at,
        Params: M.params ?? 0,
        Vocab: M.vocab ?? 0,
        Arch: M.arch ?? "",
        Corpus,
        Format: M.format ?? (Corpus.includes("sft") ? "chat" : "base"),
        Step: M.step ?? 0,
        Embed: M.embed ?? 0,
        Layers: M.layers ?? 0,
        Heads: M.heads ?? 0,
        Block: M.block ?? 0,
        Batch: M.batch ?? 0, // 0 = an older checkpoint predating these fields; the UI skips prefill
        Lr: M.lr ?? 0,
        MaxSteps: M.maxSteps ?? 0,
      };
    });
  }

  /** Delete the metadata row AND its payload file (+ `.bak`). Row first: if the file removal then
   *  fails, the worst case is an orphan file on disk (warned), never a row pointing at nothing. */
  async Delete(Name: string): Promise<void> {
    await this.Ready();
    const Rows = (await this.Sql`DELETE FROM checkpoints WHERE name = ${Name} RETURNING path`) as unknown as { path: string | null }[];
    for (const Row of Rows) if (Row.path !== null) RemovePayload(Row.path);
  }

  async Close(): Promise<void> {
    await this.Sql.end();
  }
}
