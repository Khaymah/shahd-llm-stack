// Integer keys for an ordered token pair (A, B) — the one encoding shared by the merge trainer and the
// encoder. A string key ("a,b") allocates and hashes a fresh string per lookup; a number key is an
// exact, allocation-free Map key. Ids stay far below PairKeyBase, so A * PairKeyBase + B is exact
// (< 2^42, well inside the 2^53 safe-integer range) and decodes back without loss.

/** Exclusive upper bound on a token id that can appear in a pair key (2^21 = 2,097,152 ids). */
export const PairKeyBase = 2 ** 21;

/** Exact integer key for the ordered pair (A, B). */
export function PairKey(A: number, B: number): number {
  return A * PairKeyBase + B;
}

/** The left id of a pair key. */
export function PairLeft(Key: number): number {
  return Math.floor(Key / PairKeyBase);
}

/** The right id of a pair key. */
export function PairRight(Key: number): number {
  return Key % PairKeyBase;
}

/** Throws when a vocabulary of VocabSize ids would overflow the pair-key encoding. */
export function AssertPairKeyCapacity(VocabSize: number, Where: string): void {
  if (VocabSize > PairKeyBase) throw new Error(`${Where}: vocab ${VocabSize} exceeds the pair-key capacity ${PairKeyBase}`);
}
