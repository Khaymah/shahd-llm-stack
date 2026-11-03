// Foundry ingestion + export flow (M3). IngestDocuments classifies each input into a tier, hashes
// its content (stable id + exact-dup key), embeds it, and upserts it to the store. The web-fetching
// itself is NOT here — callers pass inputs tagged with an Origin (local / web-permissive /
// web-general), so ingestion stays pure and testable and the risky network fetch is an injected
// concern. ExportTrainingText materializes only the training-eligible (Filtered) tier.

import { createHash } from "node:crypto";
import type { DocumentStore } from "./DocumentStore.ts";
import type { DocumentRecord, Origin, Tier } from "./DocumentRecord.ts";
import { ClassifyDocument } from "./Tiering.ts";
import { HashingEmbedding } from "./Embedding.ts";
import { SanitizeText } from "./ContentNormalizer.ts";

export type SourceInput = {
  Source: string;
  License: string;
  Lang: string;
  Content: string;
  Provenance: string;
  Origin: Origin;
};

// Ingested = New + Duplicate — kept for back-compat. New = rows that did not already exist; Duplicate =
// content dedup hits: re-ingesting the same bytes, under the same OR another license/origin (see
// IsContentDuplicate), or twice within one batch. A Duplicate is counted in ByTier but not stored. Reporting New vs
// Duplicate is what stops the "ingested 14355 again!" illusion when a bounded source is re-collected.
// Failed = rows whose Upsert threw (counted, never silently dropped).
export type IngestStats = { Ingested: number; New: number; Duplicate: number; ByTier: Record<Tier, number>; Failed: number };

// Failure-budget thresholds. A run that keeps failing is broken (store down, schema drift), not
// "resilient": past these limits the run aborts with a clear error instead of churning on.
/** Abort after this many failed upserts IN A ROW (no success in between). */
export const MaxConsecutiveIngestFailures = 10;
/** Abort when more than this fraction of attempted upserts failed… */
export const MaxIngestFailureRatio = 0.2;
/** …but only once at least this many upserts were attempted (small samples are too noisy). */
export const MinAttemptsForFailureRatio = 50;
/** After the first failure is logged in full, one summary warn line is emitted per this many failures. */
export const FailureLogEvery = 100;

/** Thrown when an ingest run exceeds its failure budget. Stats carries the partial counts of the
 *  IngestDocuments call that tripped it, so the caller can still account for what WAS stored. */
export class IngestAbortError extends Error {
  Stats: IngestStats;
  constructor(Message: string, Stats: IngestStats) {
    super(Message);
    this.name = "IngestAbortError";
    this.Stats = Stats;
  }
}

/** Run-scoped failure accounting shared across every IngestDocuments call of one collect run (one
 *  call per repo/batch), so the thresholds apply to the whole run, not per batch. Logs the first
 *  failure's message at warn level, then suppresses repeats to one summary line per FailureLogEvery. */
export class IngestFailureBudget {
  Attempted = 0;
  Failed = 0;
  private Consecutive = 0;
  private Warn: (Message: string) => void;

  constructor(Warn: (Message: string) => void = (Message: string): void => console.warn(Message)) {
    this.Warn = Warn;
  }

  /** Surface a non-fatal problem through the same warn channel (does not count as a failure). */
  Report(Message: string): void {
    this.Warn(Message);
  }

  RecordSuccess(): void {
    this.Attempted++;
    this.Consecutive = 0;
  }

  /** Count one failure; returns an abort reason when the budget is exceeded, else null. */
  RecordFailure(Provenance: string, Message: string): string | null {
    this.Attempted++;
    this.Failed++;
    this.Consecutive++;
    if (this.Failed === 1) this.Warn(`IngestDocuments: skipped ${Provenance}: ${Message} (further failures summarized every ${FailureLogEvery})`);
    else if (this.Failed % FailureLogEvery === 0) this.Warn(`IngestDocuments: ${this.Failed} failures so far (latest ${Provenance}: ${Message})`);
    if (this.Consecutive >= MaxConsecutiveIngestFailures) {
      return `ingest aborted: ${this.Consecutive} consecutive store failures (last: ${Message})`;
    }
    if (this.Attempted >= MinAttemptsForFailureRatio && this.Failed / this.Attempted > MaxIngestFailureRatio) {
      return `ingest aborted: ${this.Failed}/${this.Attempted} upserts failed (> ${Math.round(MaxIngestFailureRatio * 100)}%; last: ${Message})`;
    }
    return null;
  }
}

// Exact content dedup ACROSS licenses/origins (Plan §2.2 layer 2). The row id includes Origin + License
// (so a Rejected copy can never be "laundered" by re-tagging), which alone would let the same bytes
// under MIT and under Apache become two Filtered rows — trained twice. The content hash closes that:
//   - a document that would be Filtered is a Duplicate when its bytes are already stored as Filtered;
//   - any other document (Rejected/Raw) is a Duplicate when its bytes are stored in ANY tier.
// So a training copy is never blocked by an earlier Rejected/Raw copy of the same bytes (that would
// lose data), and no second training copy or redundant audit copy is ever stored. Same rule within
// one batch (the same file twice in a repo, or vendored into two repos of one run).
type KnownHashes = { Any: Set<string>; Filtered: Set<string> };

function IsContentDuplicate(Known: KnownHashes, ContentHash: string, Tier: Tier): boolean {
  return Tier === "Filtered" ? Known.Filtered.has(ContentHash) : Known.Any.has(ContentHash);
}

function RememberHash(Known: KnownHashes, ContentHash: string, Tier: Tier): void {
  Known.Any.add(ContentHash);
  if (Tier === "Filtered") Known.Filtered.add(ContentHash);
}

/** The batch's already-stored hashes: one batched lookup for any tier, one for the Filtered subset.
 *  A failed lookup degrades to "nothing known" with a warning — the id-level Upsert dedup and the
 *  in-batch dedup still apply, and a store that is really down trips the upsert failure budget. */
async function LookupKnownHashes(Store: DocumentStore, Hashes: string[], Budget: IngestFailureBudget): Promise<KnownHashes> {
  try {
    const Any = await Store.ExistingContentHashes(Hashes);
    const Filtered = Any.size === 0 ? new Set<string>() : await Store.ExistingContentHashes([...Any], "Filtered");
    return { Any, Filtered };
  } catch (Caught) {
    Budget.Report(`IngestDocuments: content-hash lookup failed, cross-license dedup limited to this batch: ${(Caught as Error).message}`);
    return { Any: new Set(), Filtered: new Set() };
  }
}

/** The exact-dedup key: sha256 of the sanitized bytes (32 hex chars), independent of origin/license. */
export function ContentHashOf(Content: string): string {
  return createHash("sha256").update(Content).digest("hex").slice(0, 32);
}

export async function IngestDocuments(
  Inputs: SourceInput[],
  Store: DocumentStore,
  IngestedAt: string,
  EmbeddingDim = 256,
  OnEach?: (Done: number, Total: number, Provenance: string) => void, // per-document progress hook
  Budget: IngestFailureBudget = new IngestFailureBudget(), // pass a shared one to bound a whole run
): Promise<IngestStats> {
  const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
  let Ingested = 0;
  let New = 0;
  let Duplicate = 0;
  let Failed = 0;
  let Done = 0;
  // Sanitize FIRST so id/hash/embedding/bytes are all computed on the exact bytes we will store —
  // and so a NUL/lone-surrogate from a binary-ish file can never make the store reject the row.
  // Hashing the whole batch up front lets the cross-license dedup be ONE batched store lookup.
  const Prepared = Inputs.map((Input) => {
    const Content = SanitizeText(Input.Content);
    return { Input, Content, ContentHash: ContentHashOf(Content) };
  });
  const Known = Prepared.length === 0 ? { Any: new Set<string>(), Filtered: new Set<string>() } : await LookupKnownHashes(Store, Prepared.map((P) => P.ContentHash), Budget);
  for (const { Input, Content, ContentHash } of Prepared) {
    const Decision = ClassifyDocument(Input.License, Content, Input.Origin, { Lang: Input.Lang });
    if (IsContentDuplicate(Known, ContentHash, Decision.Tier)) {
      // Same bytes already stored (possibly under another license/origin): counted, not stored.
      ByTier[Decision.Tier]++;
      Ingested++;
      Duplicate++;
      Done++;
      OnEach?.(Done, Inputs.length, Input.Provenance);
      continue;
    }
    // The dedup/primary key includes Origin + License so provenance-distinct copies of identical
    // bytes do NOT overwrite each other (a web-general Raw copy must not clobber a local Filtered
    // one, and a Rejected doc must not be "laundered" into Filtered by re-tagging its license).
    const Id = createHash("sha256").update(`${Input.Origin}\0${Input.License}\0${Content}`).digest("hex").slice(0, 32);
    const Record: DocumentRecord = {
      Id,
      Tier: Decision.Tier,
      Origin: Input.Origin,
      Source: Input.Source,
      License: Input.License,
      Lang: Input.Lang,
      Content,
      Bytes: Buffer.byteLength(Content, "utf8"),
      QualityScore: Decision.QualityScore,
      ContentHash,
      Embedding: HashingEmbedding(Content, EmbeddingDim),
      RejectReason: Decision.RejectReason,
      Provenance: Input.Provenance,
      IngestedAt,
    };
    // One bad row must not abort a whole multi-repo Learn run: count it and continue — until the
    // run-wide failure budget says the store is broken, then abort loudly with the partial stats.
    try {
      const Inserted = await Store.Upsert(Record);
      ByTier[Decision.Tier]++;
      Ingested++;
      if (Inserted) {
        New++;
        RememberHash(Known, ContentHash, Decision.Tier);
      } else {
        Duplicate++;
        Known.Any.add(ContentHash); // the same id (hence the same bytes) is stored, tier unknown here
      }
      Budget.RecordSuccess();
    } catch (Caught) {
      Failed++;
      const AbortReason = Budget.RecordFailure(Input.Provenance, (Caught as Error).message);
      if (AbortReason !== null) throw new IngestAbortError(AbortReason, { Ingested, New, Duplicate, ByTier, Failed });
    }
    Done++;
    OnEach?.(Done, Inputs.length, Input.Provenance);
  }
  return { Ingested, New, Duplicate, ByTier, Failed };
}

/** Materialize the training-eligible (Filtered) tier as one training-ready text blob. */
export async function ExportTrainingText(Store: DocumentStore, Separator = "\n\n"): Promise<string> {
  const Filtered = await Store.ByTier("Filtered");
  return Filtered.map((Doc) => Doc.Content).join(Separator);
}
