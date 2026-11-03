// Near-duplicate detection via MinHash (Phase 3). Raw code scrapes are full of near-dupes
// (vendored deps, forks, boilerplate); exact-hash dedup misses them, and duplicated data wastes
// training signal and raises verbatim-memorization risk (REVIEW.md). MinHash estimates Jaccard
// similarity of k-shingle sets cheaply.
//
// SCALE: comparing every pair of signatures is O(n²) — 200k docs is 2e10 signature comparisons and
// never finishes. NearDuplicateGroups therefore uses LSH BANDING: each signature is cut into bands,
// documents are bucketed by band content, and only documents sharing at least one bucket are
// compared. The band layout is derived from the signature length and the threshold so that recall
// is EXACT, not probabilistic (see LshBandLayout), which keeps the output identical to the
// brute-force reference (NearDuplicateGroupsBruteForce, kept for tests and tiny inputs).

const FnvOffset = 2166136261;
const FnvPrime = 16777619;

// Second, independent multiplier for the band key's high bits (golden-ratio constant), so two band
// contents collide only when BOTH 32-bit hashes collide. A collision is never wrong — candidates
// are always re-verified against the exact signature similarity — it only costs a comparison.
const BandHashMix = 0x9e3779b1;
const BandKeyHighBits = 2 ** 21; // 32 + 21 = 53 bits: the key stays an exact JS safe integer
const BandKeyLowMask = BandKeyHighBits - 1;

function HashString(Text: string, Seed: number): number {
  let H = (FnvOffset ^ Seed) >>> 0;
  for (let I = 0; I < Text.length; I++) {
    H ^= Text.charCodeAt(I);
    H = Math.imul(H, FnvPrime);
  }
  return H >>> 0;
}

/** Character k-gram shingles (whitespace-normalized) — the set MinHash approximates. */
export function Shingles(Text: string, K = 5): string[] {
  const Norm = Text.replace(/\s+/g, " ").trim();
  if (Norm.length <= K) return [Norm];
  const Out: string[] = [];
  for (let I = 0; I + K <= Norm.length; I++) Out.push(Norm.slice(I, I + K));
  return Out;
}

/** MinHash signature: for each of NumHashes seeds, the min hash over all shingles. */
export function MinHashSignature(Text: string, NumHashes = 64, K = 5): number[] {
  const Grams = Shingles(Text, K);
  const Sig = new Array<number>(NumHashes).fill(0xffffffff);
  for (const Gram of Grams) {
    for (let S = 0; S < NumHashes; S++) {
      const H = HashString(Gram, S);
      if (H < Sig[S]) Sig[S] = H;
    }
  }
  return Sig;
}

/** Estimated Jaccard similarity = fraction of signature positions that agree. */
export function EstimateJaccard(SigA: number[], SigB: number[]): number {
  const N = Math.min(SigA.length, SigB.length);
  let Equal = 0;
  for (let I = 0; I < N; I++) if (SigA[I] === SigB[I]) Equal++;
  return Equal / N;
}

/** How a signature of length L is cut into bands for a given threshold.
 *  MinAgreements: the fewest agreeing positions E with E / L >= Threshold (L + 1 = unreachable).
 *  BandStarts:    Bands + 1 offsets; band B covers positions [BandStarts[B], BandStarts[B + 1]). */
export type LshLayout = { MinAgreements: number; Bands: number; BandStarts: number[] };

/** Derive an EXACT-recall band layout. A pair meeting the threshold agrees on >= E positions, so it
 *  disagrees on at most D = L - E. With D + 1 disjoint bands covering all L positions, those D
 *  disagreements cannot touch every band (pigeonhole): at least one band matches completely, and
 *  the pair shares that band's bucket. More bands would only add candidates; fewer could miss pairs.
 *  Example: L = 64, Threshold = 0.8 -> E = 52, 13 bands of 4-5 rows.
 *  Bands = 0 means banding cannot prune (Threshold <= 0 matches everything, or the threshold is
 *  unreachable) — callers handle those cases directly. */
export function LshBandLayout(SignatureLength: number, Threshold: number): LshLayout {
  const L = SignatureLength;
  // Same comparison as EstimateJaccard(...) >= Threshold, so float rounding cannot make the two
  // disagree about a borderline pair. Scan from the top: E/L is monotonic in E.
  let E = L + 1;
  for (let Candidate = L; Candidate >= 0; Candidate--) {
    if (Candidate / L >= Threshold) E = Candidate;
    else break;
  }
  if (L === 0 || E === 0 || E > L) return { MinAgreements: E, Bands: 0, BandStarts: [0] };
  const Bands = L - E + 1;
  const BandStarts: number[] = [];
  for (let B = 0; B <= Bands; B++) BandStarts.push(Math.floor((B * L) / Bands)); // sizes differ by <= 1
  return { MinAgreements: E, Bands, BandStarts };
}

function AssertUniformLength(Sigs: number[][]): number {
  const L = Sigs.length === 0 ? 0 : Sigs[0].length;
  for (let I = 1; I < Sigs.length; I++) {
    if (Sigs[I].length !== L) {
      throw new Error(`NearDedup: signature ${I} has length ${Sigs[I].length}, expected ${L} (all signatures must share NumHashes)`);
    }
  }
  return L;
}

/** Reference implementation: greedy grouping over EVERY pair, O(n²). Document I (ascending) opens a
 *  group if still ungrouped and claims every later ungrouped J with EstimateJaccard >= Threshold.
 *  Kept as the oracle the LSH path must match exactly (and for very small inputs). */
export function NearDuplicateGroupsBruteForce(Sigs: number[][], Threshold: number): number[][] {
  AssertUniformLength(Sigs);
  const GroupOf = new Array<number>(Sigs.length).fill(-1);
  const Groups: number[][] = [];
  for (let I = 0; I < Sigs.length; I++) {
    if (GroupOf[I] !== -1) continue;
    const Group = [I];
    GroupOf[I] = Groups.length;
    for (let J = I + 1; J < Sigs.length; J++) {
      if (GroupOf[J] === -1 && EstimateJaccard(Sigs[I], Sigs[J]) >= Threshold) {
        GroupOf[J] = Groups.length;
        Group.push(J);
      }
    }
    Groups.push(Group);
  }
  return Groups;
}

function BandKey(Sig: number[], Start: number, End: number): number {
  let Low = (FnvOffset ^ Start) >>> 0;
  let High = (BandHashMix ^ End) >>> 0;
  for (let P = Start; P < End; P++) {
    const V = Sig[P] >>> 0;
    Low = Math.imul(Low ^ V, FnvPrime) >>> 0;
    High = Math.imul(High ^ V, BandHashMix) >>> 0;
    High ^= High >>> 15;
  }
  return (High & BandKeyLowMask) * 2 ** 32 + Low;
}

/** LSH-banded grouping with output IDENTICAL to NearDuplicateGroupsBruteForce (same greedy rule,
 *  same group order, same member order) — banding only prunes pairs that cannot reach Threshold. */
export function NearDuplicateGroupsFromSignatures(Sigs: number[][], Threshold: number): number[][] {
  const L = AssertUniformLength(Sigs);
  const N = Sigs.length;
  const Layout = LshBandLayout(L, Threshold);
  if (Layout.Bands === 0) {
    // Unreachable threshold (or empty signatures): nothing can pair, every doc stands alone.
    if (L === 0 || Layout.MinAgreements > L) {
      const Singletons: number[][] = [];
      for (let I = 0; I < N; I++) Singletons.push([I]);
      return Singletons;
    }
    return NearDuplicateGroupsBruteForce(Sigs, Threshold); // Threshold <= 0: everything matches
  }

  // Buckets[B]: band-key -> ascending doc indices (inserted in index order). Keys[B][I] remembers
  // each doc's key so the grouping pass does not rehash.
  const Buckets: Map<number, number[]>[] = [];
  const Keys: Float64Array[] = [];
  for (let B = 0; B < Layout.Bands; B++) {
    const Start = Layout.BandStarts[B];
    const End = Layout.BandStarts[B + 1];
    const Bucket = new Map<number, number[]>();
    const BandKeys = new Float64Array(N);
    for (let I = 0; I < N; I++) {
      const Key = BandKey(Sigs[I], Start, End);
      BandKeys[I] = Key;
      const Members = Bucket.get(Key);
      if (Members === undefined) Bucket.set(Key, [I]);
      else Members.push(I);
    }
    Buckets.push(Bucket);
    Keys.push(BandKeys);
  }

  const GroupOf = new Int32Array(N).fill(-1);
  const SeenBy = new Int32Array(N).fill(-1); // SeenBy[J] === I: pair (I, J) already verified this pass
  const Groups: number[][] = [];
  for (let I = 0; I < N; I++) {
    if (GroupOf[I] !== -1) continue;
    const GroupId = Groups.length;
    GroupOf[I] = GroupId;
    const Claimed: number[] = [];
    for (let B = 0; B < Layout.Bands; B++) {
      const Members = Buckets[B].get(Keys[B][I]);
      if (Members === undefined || Members.length === 1) continue;
      for (const J of Members) {
        if (J <= I || GroupOf[J] !== -1 || SeenBy[J] === I) continue;
        SeenBy[J] = I;
        if (EstimateJaccard(Sigs[I], Sigs[J]) >= Threshold) Claimed.push(J);
      }
    }
    // Brute force claims J in ascending order; claim membership never depends on order within one
    // pass (only on GroupOf before it), so sorting reproduces its member order exactly.
    Claimed.sort((A, B) => A - B);
    for (const J of Claimed) GroupOf[J] = GroupId;
    Groups.push([I, ...Claimed]);
  }
  return Groups;
}

/** Group documents whose estimated Jaccard >= Threshold. Same greedy semantics and output as the
 *  O(n²) reference, but LSH-banded: near-linear for corpora without massive near-dup clusters. */
export function NearDuplicateGroups(Docs: string[], Threshold = 0.8, NumHashes = 64): number[][] {
  return NearDuplicateGroupsFromSignatures(Docs.map((D) => MinHashSignature(D, NumHashes)), Threshold);
}

/** Keep one representative per near-duplicate group; return the indices to keep. */
export function DedupedIndices(Docs: string[], Threshold = 0.8): number[] {
  return NearDuplicateGroups(Docs, Threshold).map((G) => G[0]);
}
