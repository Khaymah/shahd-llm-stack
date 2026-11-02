// The unit the Data Foundry stores and inspects (M3). Every ingested document lands in exactly one
// tier with full provenance, a quality score, a content hash (for exact-dup detection), and an
// embedding (for semantic near-dup / "find similar"). Tiers:
//   Filtered — permissive + passed quality: eligible for training.
//   Raw      — general-web / unverified: kept for INSPECTION ONLY, never trained until licensed.
//   Rejected — non-permissive or low-quality: kept with a reason so the corpus is auditable.

export type Tier = "Filtered" | "Raw" | "Rejected";

// Where a document came from — decides tiering. "owned" = our own repos (trained on regardless of
// license, no third-party risk); "curated" = a general/text dataset we explicitly vetted and approved
// as a whole (OASST, Wikipedia, public-domain books) — its single source-level license is recorded for
// provenance but not re-checked per document; "web-general" is always isolated to Raw; "synthetic" = a
// conversation written by a local open-weights teacher (Scripts/SynthesizeSft.ts), stored as a
// shahd-chat-v1 document whose License is "<source SPDX>;teacher:<model>" (Foundry/SyntheticLicense.ts).
export type Origin = "owned" | "local" | "web-permissive" | "web-general" | "curated" | "synthetic";

export type DocumentRecord = {
  Id: string; // stable id = content hash
  Tier: Tier;
  Origin: Origin;
  Source: string; // repo/dataset/site name
  License: string; // SPDX id or "unknown"
  Lang: string; // language tag
  Content: string;
  Bytes: number;
  QualityScore: number;
  ContentHash: string;
  Embedding: number[];
  RejectReason: string | null;
  Provenance: string; // path or URL
  IngestedAt: string; // ISO timestamp (supplied by the caller — keeps ingestion pure/testable)
};
