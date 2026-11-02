// Collection progress state (data engine, Phase 1). Each collectable SOURCE keeps a durable row of how
// much it has produced and whether it is finished — the "collection ledger". This turns blind re-runs
// into stateful collection: a bounded dataset (OASST1) can be marked Exhausted so the UI shows
// "complete" instead of re-pulling it, and a streaming source records its running total across runs.
// Cursor is opaque per-source resume state (a row offset for a paged dataset, the last query for GitHub)
// so a future run can continue where the last stopped. Mirrors ChatStore: an interface with an in-memory
// (tests) and a Postgres (product) implementation, both on the shared corpus database.

import type { DataKind } from "./DataKinds.ts";

export type CollectionState = {
  SourceKey: string; // a collectable unit: "oasst:all", "wikipedia:en", "github:stars:>1000 language:ts"
  Kind: DataKind;
  Cursor: string; // opaque per-source resume state as JSON ("{}" when none)
  Collected: number; // running total of NEW documents ingested for this source (across runs)
  Exhausted: boolean; // a bounded source confirmed fully collected (a re-run produced only duplicates)
  UpdatedAt: string;
};

export interface CollectionStateStore {
  /** One source's state, or null if it was never collected. */
  Get(SourceKey: string): Promise<CollectionState | null>;
  /** Insert or replace a source's state (idempotent on SourceKey). */
  Upsert(State: CollectionState): Promise<void>;
  /** All known sources' state, most recently updated first — the dashboard's collection ledger. */
  All(): Promise<CollectionState[]>;
}

/** The unit a provider's per-run cap (Settings.MaxRepos) counts: GitHub/local repo providers cap
 *  REPOSITORIES, dataset providers cap DOCUMENTS. Exhaustion must compare like with like. */
export type CapUnit = "repos" | "documents";
export type UnitCount = { Count: number; Unit: CapUnit };

/** Whether a run CONFIRMS a bounded source is fully collected: it produced 0 new docs (only duplicates,
 *  so it re-checked existing data) AND ran dry BEFORE the per-run cap (Produced < Cap) rather than being
 *  merely truncated by it. This guard is trust-critical: without `Produced < Cap`, re-running OASST with
 *  a cap of 2000 (of ~14k) would falsely announce "fully collected" — the exact "you're stuck at this
 *  data" signal the whole data-engine effort set out to eliminate. Produced and Cap MUST be in the same
 *  unit (a repo cap vs. a file count compared nothing meaningful); mismatched units never confirm
 *  exhaustion. Streaming sources are never exhausted. */
export function ComputeExhausted(
  Semantics: "bounded" | "streaming" | undefined,
  New: number,
  Duplicate: number,
  Produced: UnitCount,
  Cap: UnitCount,
): boolean {
  if (Produced.Unit !== Cap.Unit) return false; // cannot confirm across units — never claim "complete"
  return Semantics === "bounded" && New === 0 && Duplicate > 0 && Produced.Count < Cap.Count;
}
