// Resolve Foundry run settings from the environment (Bun loads .env automatically) with CLI
// override. One home for this so every foundry script behaves the same: it defaults to Postgres
// whenever DATABASE_URL is set, reads GITHUB_TOKEN/BRAVE_API_KEY/queries/repos from env, and lets
// a --Flag override any of them.

import { InMemoryDocumentStore } from "../Foundry/FoundryBarrel.ts";
import type { DocumentStore } from "../Foundry/FoundryBarrel.ts";
import { PostgresDocumentStore } from "../Foundry/PostgresDocumentStore.ts";
import { FoundryStores } from "../Foundry/FoundryStores.ts";
import { ResolveCheckpointDir } from "../Foundry/PostgresCheckpointStore.ts";
import { DefaultDataSpecPath } from "../Foundry/DataSpec.ts";
import { ReadArg } from "./ScriptArgs.ts";

export function DatabaseUrl(): string {
  return process.env["DATABASE_URL"] ?? "postgres://shahd:shahd@localhost:5432/shahd";
}

/** Checkpoint payload directory: FOUNDRY_CHECKPOINT_DIR > "Checkpoints". The store writes
 *  `<dir>/<SafeName>.ckpt` there and keeps only metadata in Postgres (see PostgresCheckpointStore). */
export function CheckpointDir(): string {
  return ResolveCheckpointDir();
}

export function GitHubToken(): string | undefined {
  return process.env["GITHUB_TOKEN"];
}

export function BraveKey(): string | undefined {
  return process.env["BRAVE_API_KEY"];
}

/** Store: --Store CLI > FOUNDRY_STORE env > (DATABASE_URL present ? postgres : memory). */
export function ResolveStore(): { Store: DocumentStore; Kind: string } {
  const Choice = ReadArg("--Store=", "") || process.env["FOUNDRY_STORE"] || (process.env["DATABASE_URL"] !== undefined ? "postgres" : "memory");
  if (Choice === "postgres") return { Store: new PostgresDocumentStore(DatabaseUrl()), Kind: "postgres" };
  return { Store: new InMemoryDocumentStore(), Kind: "memory" };
}

/** The per-kind store set (Postgres only — kind separation is a database feature). Needs DATABASE_URL. */
export function ResolveFoundryStores(): FoundryStores {
  return new FoundryStores(DatabaseUrl());
}

/** Web search/repo query: --Query CLI > FOUNDRY_QUERY env > the provided default. */
export function Query(Default: string): string {
  return ReadArg("--Query=", "") || process.env["FOUNDRY_QUERY"] || Default;
}

/** The data spec file: --Spec CLI > FOUNDRY_SPEC env > DefaultDataSpecPath (Data/Spec.json). */
export function DataSpecPath(): string {
  return ReadArg("--Spec=", "") || process.env["FOUNDRY_SPEC"] || DefaultDataSpecPath;
}

/** Own-repo roots: --Repos CLI > FOUNDRY_REPOS env > ".". Comma-separated. */
export function RepoRoots(): string[] {
  const Raw = ReadArg("--Repos=", "") || process.env["FOUNDRY_REPOS"] || ".";
  return Raw.split(",").map((S) => S.trim()).filter((S) => S.length > 0);
}

/** The chat model loaded at startup — the only name allowed the checkpoint-FILE fallback. */
export function DefaultCheckpointName(): string {
  return process.env["FOUNDRY_CHECKPOINT_NAME"] ?? "Seed";
}

/** Checkpoint FILE fallback for the default model: --Checkpoint CLI > FOUNDRY_CHECKPOINT env > default. */
export function CheckpointFile(): string {
  return ReadArg("--Checkpoint=", "") || process.env["FOUNDRY_CHECKPOINT"] || "Checkpoints/Foundry.ckpt";
}

/** Self-consistency samples per chat turn (SHAHD_SELF_CONSISTENCY, default 1 = off). N>1 answers N times
 *  with different seeds and majority-votes — test-time compute; needs Temperature>0 to diverge. */
export function SelfConsistencySamples(): number {
  return Math.max(1, Math.floor(Number(process.env["SHAHD_SELF_CONSISTENCY"] ?? "1")) || 1);
}
