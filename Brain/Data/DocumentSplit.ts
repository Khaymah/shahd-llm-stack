// The document-level train/validation membership rule for store-backed corpora — ONE home, shared by
// the pretraining script (Scripts/TrainOnFoundry.ts) and the evaluation suite's frozen held-out set
// (Brain/Eval/Suite/HeldOutBpb.ts), so "validation" means the same documents in both places.
//
// Membership is a property of the DOCUMENT (its id hashes into a 0..99 bucket; buckets below
// ValidationPercent are validation), not of a shuffle: a resumed run — or a later run over a grown
// corpus — keeps every already-seen document on the same side, so nothing that was validation ever
// leaks into training, and the eval suite can select the held-out documents without replaying a run.

import { createHash } from "node:crypto";

/** Buckets a document id hashes into. */
export const SplitBucketCount = 100;

/** Buckets [0, ValidationPercent) are validation / held-out (10% of documents). */
export const ValidationPercent = 10;

// Hex digits of the sha256 read as the bucket source: 8 hex = 32 bits, far more than 100 buckets need,
// and small enough to stay an exact JS integer.
const BucketHexDigits = 8;
const HexRadix = 16;

/** The id's split bucket in [0, SplitBucketCount): sha256(id), first 32 bits, mod 100. */
export function DocumentSplitBucket(Id: string): number {
  return Number.parseInt(createHash("sha256").update(Id).digest("hex").slice(0, BucketHexDigits), HexRadix) % SplitBucketCount;
}

/** True when the document belongs to the validation / held-out side. */
export function IsValidationDocument(Id: string): boolean {
  return DocumentSplitBucket(Id) < ValidationPercent;
}
