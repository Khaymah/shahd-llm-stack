// The document-level train/validation membership rule for store-backed corpora — ONE home, shared by
// the pretraining script (Scripts/TrainOnFoundry.ts) and the evaluation suite's frozen held-out set
// (Brain/Eval/Suite/HeldOutBpb.ts), so "validation" means the same documents in both places.
//
// Membership is a property of the DOCUMENT (its id hashes into a 0..99 bucket; buckets below
// ValidationPercent are validation), not of a shuffle: a resumed run — or a later run over a grown
// corpus — keeps every already-seen document on the same side, so nothing that was validation ever
// leaks into training, and the eval suite can select the held-out documents without replaying a run.
// The SFT script (Scripts/TrainSftChat.ts) applies the same rule, so chat fine-tuning never trains on a
// document the held-out BPB measures.

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

/** Split documents by the membership rule above: { Train, Validation }, each in the input order. */
export function PartitionByDocumentSplit<T extends { Id: string }>(Docs: readonly T[]): { Train: T[]; Validation: T[] } {
  const Train: T[] = [];
  const Validation: T[] = [];
  for (const Doc of Docs) (IsValidationDocument(Doc.Id) ? Validation : Train).push(Doc);
  return { Train, Validation };
}

/** How many documents to read so that about Wanted of them land on the TRAINING side (the rule keeps
 *  ValidationPercent% aside, so read Wanted / (1 - ValidationPercent/100), rounded up). */
export function TrainingSideReadSize(Wanted: number): number {
  if (Wanted <= 0) return 0;
  return Math.ceil((Wanted * SplitBucketCount) / (SplitBucketCount - ValidationPercent));
}

/**
 * Per-document "is validation" flags for a corpus that must have BOTH sides non-empty (a training run
 * needs a train stream and a validation stream). Membership follows the bucket rule; on a tiny corpus
 * where the hash put everything on one side, the single most borderline document is moved across —
 * chosen by (bucket, id), so the fix-up is itself deterministic. Needs at least 2 documents.
 */
export function AssignValidationSides(Ids: readonly string[]): boolean[] {
  if (Ids.length < 2) throw new Error(`AssignValidationSides: need >= 2 documents for a train/validation split (got ${Ids.length})`);
  const Buckets = Ids.map(DocumentSplitBucket);
  const IsVal = Buckets.map((B) => B < ValidationPercent);
  const ByBucket = Ids.map((_Id, I) => I).sort((A, B) => Buckets[A] - Buckets[B] || (Ids[A] < Ids[B] ? -1 : Ids[A] > Ids[B] ? 1 : 0));
  if (!IsVal.includes(true)) IsVal[ByBucket[0]] = true;
  if (!IsVal.includes(false)) IsVal[ByBucket[ByBucket.length - 1]] = false;
  return IsVal;
}
