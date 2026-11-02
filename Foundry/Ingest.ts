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

// Ingested = successful upserts (New + Duplicate) — kept for back-compat. New = rows that did not
// already exist; Duplicate = content-hash dedup hits (re-ingesting the same bytes). Reporting New vs
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
  for (const Input of Inputs) {
    // Sanitize FIRST so id/hash/embedding/bytes are all computed on the exact bytes we will store —
    // and so a NUL/lone-surrogate from a binary-ish file can never make the store reject the row.
    const Content = SanitizeText(Input.Content);
    const Decision = ClassifyDocument(Input.License, Content, Input.Origin);
    const ContentHash = createHash("sha256").update(Content).digest("hex").slice(0, 32);
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
      if (Inserted) New++;
      else Duplicate++;
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
