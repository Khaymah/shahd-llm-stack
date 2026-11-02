// The Foundry's persistence boundary (M3). The model never depends on this; only the data-curation
// layer does. Async so a Postgres/pgvector implementation fits the same interface as the in-memory
// one used in tests. FindSimilar powers semantic near-dup and "find similar" over embeddings.
//
// Projection rule: a read only pulls the columns its caller needs. Browse pages use ListSummaries (no
// content, no embedding); training reads use ByTier (content, no embedding); aggregates (Stats,
// RepoSummaries, Count) never touch content. Only Query / DocumentById / All return full records.

import type { DocumentRecord, Origin, Tier } from "./DocumentRecord.ts";

export type SimilarHit = { Doc: DocumentRecord; Score: number };

// Per-repo rollup for the dashboard (accordion list + "already learned" skip).
export type RepoSummary = { Source: string; Files: number; Bytes: number };

// A filter for the data-browser: any subset of tier/lang/license/origin/source + a free-text search.
// Search is a case-insensitive SUBSTRING match against provenance OR content — the characters are
// literal (`my_func` never matches `myXfunc`, `%` is just a percent sign). Omitted fields are
// unconstrained. Empty filter = everything.
export type DocumentFilter = { Tier?: Tier; Lang?: string; License?: string; Origin?: Origin; Source?: string; Search?: string };

// One page of a filtered browse: the rows for this page plus the TOTAL matching count (so the UI can
// render "page X of Y" without loading every row).
export type DocumentPage = { Rows: DocumentRecord[]; Total: number };

// A document without its heavy columns — what a browse table renders. Content/embedding are fetched
// separately (DocumentById / ContentByIds) only for the rows actually opened.
export type DocumentSummary = Omit<DocumentRecord, "Content" | "Embedding">;
export type SummaryPage = { Rows: DocumentSummary[]; Total: number };

// A training read: everything but the embedding (256 floats/doc that training never uses).
export type TierDocument = Omit<DocumentRecord, "Embedding">;

/**
 * Options for ListSummaries. Two paging modes:
 *  - Offset paging (default): newest-first by ingestion, id tie-break — the browse order.
 *  - Keyset paging (AfterId set, "" for the first page): ascending id, strictly after AfterId; Offset
 *    is ignored. Stable while rows are being retiered/inserted — the streaming path for maintenance.
 * Total is always the full count matching Filter (AfterId does not narrow it).
 */
export type ListOptions = { Filter?: DocumentFilter; Offset?: number; Limit?: number; AfterId?: string };

// Aggregate counts for the dashboard cards — computed WITHOUT loading document content.
export type FoundryStats = {
  Total: number;
  ByTier: Record<Tier, number>;
  ByLang: Record<string, number>;
  ByLicense: Record<string, number>;
  FilteredBytes: number;
};

/** Hard ceiling on any page a store returns (Query, ListSummaries, DocumentsBySource, ContentByIds) —
 *  one request can never materialize an unbounded slice of the corpus. */
export const MaxPageLimit = 500;
/** Page size used when a caller passes no limit, zero, a negative, or a non-number. */
export const DefaultPageLimit = 50;

/** Normalize a caller-supplied page limit into [1, MaxPageLimit] (non-positive/NaN → default). */
export function ClampPageLimit(Limit: number | undefined): number {
  if (Limit === undefined || !Number.isFinite(Limit) || Limit <= 0) return DefaultPageLimit;
  return Math.min(Math.floor(Limit), MaxPageLimit) || DefaultPageLimit;
}

/** Normalize a caller-supplied offset (negative/NaN → 0). */
export function ClampOffset(Offset: number | undefined): number {
  if (Offset === undefined || !Number.isFinite(Offset) || Offset <= 0) return 0;
  return Math.floor(Offset);
}

/** The search needle both stores match on: trimmed, or null when there is nothing to search for. */
export function NormalizeSearch(Search: string | undefined): string | null {
  const Trimmed = Search?.trim() ?? "";
  return Trimmed === "" ? null : Trimmed;
}

export interface DocumentStore {
  /** Insert a document by Id (content hash) if it is not already stored. Returns true if the row was
   *  NEWLY inserted, false if it already existed — in which case the stored row is left untouched
   *  (first-seen provenance wins; re-collecting the same bytes never rewrites source/provenance).
   *  Reclassification of an existing row goes through Retier / ReclassifyBySource instead. */
  Upsert(Doc: DocumentRecord): Promise<boolean>;
  All(): Promise<DocumentRecord[]>;
  /** Documents in a tier, in ascending Id order (deterministic across runs), without embeddings.
   *  Limit caps the read (per-kind training size control); it is NOT clamped to MaxPageLimit. */
  ByTier(Tier: Tier, Limit?: number): Promise<TierDocument[]>;
  FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]>;
  Count(): Promise<number>;
  /** Distinct repo/source names already ingested (used to skip re-learning). */
  Sources(): Promise<string[]>;
  /** Per-repo file counts + bytes, most files first (the dashboard's repo list). An optional filter
   *  narrows the rollup (e.g. { License: "NOASSERTION" } for the license backfill). */
  RepoSummaries(Filter?: DocumentFilter): Promise<RepoSummary[]>;
  /** Documents belonging to one repo/source (accordion contents). Limit is clamped to MaxPageLimit. */
  DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]>;
  /** One document's full record by id (content hash) — powers the file viewer. Null if absent. */
  DocumentById(Id: string): Promise<DocumentRecord | null>;
  /**
   * Which of these content hashes (sha256 of the stored bytes, see Ingest) are already stored in this
   * table — in any tier, or only in Tier when given. Batched exact-dedup ACROSS licenses/origins (the
   * id includes origin + license, the content hash does not). Any number of hashes may be passed (the
   * Postgres store chunks the lookup); duplicates in the input are fine; [] → empty set.
   */
  ExistingContentHashes(Hashes: readonly string[], Tier?: Tier): Promise<Set<string>>;
  /** Content for a batch of ids (absent ids are simply missing from the map). At most MaxPageLimit
   *  ids are read per call — callers page through ListSummaries and fetch content per page. */
  ContentByIds(Ids: readonly string[]): Promise<Map<string, string>>;
  /**
   * A filtered, paginated page of FULL documents. Offset/Limit page the results (Limit clamped to
   * MaxPageLimit); Total is the full matching count. Rows come newest-first (by ingestion, id
   * tie-break). Prefer ListSummaries for browse tables — this one carries content and embeddings.
   */
  Query(Filter: DocumentFilter, Offset: number, Limit: number): Promise<DocumentPage>;
  /** A page of documents WITHOUT content/embedding (browse tables, maintenance scans). See ListOptions. */
  ListSummaries(Opts: ListOptions): Promise<SummaryPage>;
  /** Delete one document by id. Returns the number removed (0 if the id was absent, else 1). */
  DeleteById(Id: string): Promise<number>;
  /** Bulk-delete every document matching a filter (corpus cleanup). Returns how many were removed. */
  DeleteMatching(Filter: DocumentFilter): Promise<number>;
  /** Aggregate dashboard stats (counts by tier/lang/license + filtered bytes), computed efficiently. */
  Stats(): Promise<FoundryStats>;
  /**
   * Explicitly reclassify ONE stored document (the only way an existing row's tier changes, since
   * Upsert never overwrites). RejectReason is set as given (null clears it); QualityScore is updated
   * only when provided. Returns true if the id existed.
   */
  Retier(Id: string, Tier: Tier, RejectReason: string | null, QualityScore?: number): Promise<boolean>;
  /**
   * Relabel a source's NOASSERTION docs once its real license is verified (license-backfill). Docs
   * that also pass quality (>= MinQuality) are promoted to Filtered with NewLicense and reject_reason
   * cleared; the rest keep their tier but get NewLicense + a low-quality reason. Scoped to
   * license='NOASSERTION' so only the previously-unresolved rows are touched. Atomic: both relabels
   * commit together or not at all. Returns the split.
   */
  ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }>;
}
