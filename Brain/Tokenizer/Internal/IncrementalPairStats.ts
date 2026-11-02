// Incremental pair statistics for BPE merge training (the classic incremental algorithm).
//
// State: every unique pretoken "piece" (byte ids + its corpus frequency), a weighted count per adjacent
// pair, and a pair -> pieces index. A merge rewrites ONLY the pieces that contain the merged pair and
// applies the resulting count deltas — nothing is recounted globally. The best pair is drawn from a lazy
// max-heap (PairHeap).
//
// Reproducing the reference tie-break exactly. The reference trainer (TrainBpeReference) recounts every
// pair into a Map in (piece order, position) order and keeps the FIRST pair with the highest count, so
// among equal counts the pair whose first occurrence comes earliest wins. Here every pair carries an
// Order = PieceIndex * OffsetSpan + ByteOffset of that first occurrence. Byte offsets are stable under
// merges (merging never reorders bytes), so comparing offsets equals comparing current positions.
// Within one merge a pair's occurrences only ever move in ONE direction:
//   - pairs that gain occurrences all contain the new id Z, which never existed before — so they are
//     brand-new pairs and their Order is exact (the minimum over the new occurrences);
//   - pairs that lose occurrences never contain Z; their stored Order becomes a LOWER BOUND (the first
//     occurrence may have been removed) and the pair is marked dirty.
// The heap orders by (Count desc, Order asc). A dirty pair reaching the top gets its exact Order
// recomputed from its pieces; if that moved it, it is re-pushed. A clean top entry therefore has the
// highest count and the smallest TRUE Order among all pairs with that count — the reference's choice.

import { PairHeap } from "./PairHeap.ts";
import { PairKey, PairLeft, PairRight } from "./PairKey.ts";

export const ByteVocab = 256;

// Rebuild the lazy heap from the live counts once stale entries outnumber live pairs by this factor
// (bounds memory: every count change pushes a fresh entry).
const HeapCompactionFactor = 4;
const MinHeapEntriesBeforeCompaction = 1 << 16;

export type PieceSet = { Ids: number[][]; Weights: number[] };

export class IncrementalPairStats {
  private readonly Pieces: number[][];
  private readonly Weights: readonly number[];
  private readonly Counts = new Map<number, number>();
  private readonly Where = new Map<number, number[]>(); // superset of the pieces holding the pair
  private readonly Orders = new Map<number, number>(); // exact first occurrence, or a lower bound if dirty
  private readonly Dirty = new Set<number>();
  private readonly ByteLength: number[] = [];
  private readonly Heap = new PairHeap();
  private readonly Seen: Int32Array; // per-piece visit stamp (dedupes the pair -> pieces lists)
  private readonly OffsetSpan: number;
  private Epoch = 0;
  private NextId = ByteVocab;

  constructor(Pieces: PieceSet) {
    this.Pieces = Pieces.Ids;
    this.Weights = Pieces.Weights;
    this.Seen = new Int32Array(this.Pieces.length).fill(-1);
    for (let Id = 0; Id < ByteVocab; Id++) this.ByteLength.push(1);
    let MaxBytes = 0;
    for (const Ids of this.Pieces) MaxBytes = Math.max(MaxBytes, Ids.length);
    this.OffsetSpan = MaxBytes + 1;
    for (let P = 0; P < this.Pieces.length; P++) {
      const Ids = this.Pieces[P];
      const Weight = this.Weights[P];
      let Offset = 0;
      for (let I = 0; I + 1 < Ids.length; I++) {
        const Key = PairKey(Ids[I], Ids[I + 1]);
        this.Counts.set(Key, (this.Counts.get(Key) ?? 0) + Weight);
        this.AddWhere(Key, P);
        // Pieces are visited in order and positions left to right: the first visit IS the minimum.
        if (!this.Orders.has(Key)) this.Orders.set(Key, P * this.OffsetSpan + Offset);
        Offset += this.ByteLength[Ids[I]];
      }
    }
    this.RebuildHeap();
  }

  /** The next pair to merge (reference tie-break), or null when no adjacent pair remains. */
  SelectBest(): number | null {
    while (this.Heap.Size > 0) {
      const Key = this.Heap.TopKey();
      const Count = this.Heap.TopCount();
      const Order = this.Heap.TopOrder();
      this.Heap.Pop();
      if (this.Counts.get(Key) !== Count || this.Orders.get(Key) !== Order) continue; // stale entry
      if (this.Dirty.has(Key)) {
        const Exact = this.ExactOrder(Key);
        this.Dirty.delete(Key);
        this.Orders.set(Key, Exact);
        if (Exact !== Order) {
          this.Heap.Push(Count, Exact, Key);
          continue;
        }
      }
      return Key;
    }
    return null;
  }

  /** Merge every occurrence of the pair (left to right, non-overlapping); returns the new id. */
  Merge(Key: number): number {
    const A = PairLeft(Key);
    const B = PairRight(Key);
    const Z = this.NextId++;
    this.ByteLength.push(this.ByteLength[A] + this.ByteLength[B]);
    const Delta = new Map<number, number>();
    const NewFirst = new Map<number, number>();
    const Stamp = this.NextEpoch();
    for (const P of this.Where.get(Key) ?? []) {
      if (this.Seen[P] === Stamp) continue;
      this.Seen[P] = Stamp;
      const Ids = this.Pieces[P];
      if (FirstPairIndex(Ids, A, B) < 0) continue; // stale index entry
      const Weight = this.Weights[P];
      for (let I = 0; I + 1 < Ids.length; I++) {
        const Old = PairKey(Ids[I], Ids[I + 1]);
        Delta.set(Old, (Delta.get(Old) ?? 0) - Weight);
      }
      const Next = ReplacePair(Ids, A, B, Z);
      let Offset = 0;
      for (let I = 0; I + 1 < Next.length; I++) {
        const Pair = PairKey(Next[I], Next[I + 1]);
        Delta.set(Pair, (Delta.get(Pair) ?? 0) + Weight);
        if (Next[I] === Z || Next[I + 1] === Z) {
          this.AddWhere(Pair, P);
          const At = P * this.OffsetSpan + Offset;
          const Known = NewFirst.get(Pair);
          if (Known === undefined || At < Known) NewFirst.set(Pair, At);
        }
        Offset += this.ByteLength[Next[I]];
      }
      this.Pieces[P] = Next;
    }
    for (const [Pair, Change] of Delta) {
      if (Change === 0) continue;
      const Count = (this.Counts.get(Pair) ?? 0) + Change;
      if (Count <= 0) {
        this.Forget(Pair);
        continue;
      }
      this.Counts.set(Pair, Count);
      if (Change < 0) {
        this.Dirty.add(Pair); // lost occurrences: the stored Order is now only a lower bound
      } else {
        this.Orders.set(Pair, NewFirst.get(Pair) ?? Number.MAX_SAFE_INTEGER); // brand-new pair: exact
      }
      this.Heap.Push(Count, this.Orders.get(Pair) ?? 0, Pair);
    }
    this.Forget(Key);
    if (this.Heap.Size > MinHeapEntriesBeforeCompaction && this.Heap.Size > HeapCompactionFactor * this.Counts.size) this.RebuildHeap();
    return Z;
  }

  private Forget(Key: number): void {
    this.Counts.delete(Key);
    this.Where.delete(Key);
    this.Orders.delete(Key);
    this.Dirty.delete(Key);
  }

  private AddWhere(Key: number, Piece: number): void {
    const List = this.Where.get(Key);
    if (List === undefined) this.Where.set(Key, [Piece]);
    else if (List[List.length - 1] !== Piece) List.push(Piece);
  }

  private NextEpoch(): number {
    this.Epoch++;
    return this.Epoch;
  }

  /** Exact first-occurrence Order of a pair; prunes stale and duplicate entries from its piece list. */
  private ExactOrder(Key: number): number {
    const A = PairLeft(Key);
    const B = PairRight(Key);
    const Stamp = this.NextEpoch();
    const Kept: number[] = [];
    let Best = Number.MAX_SAFE_INTEGER;
    for (const P of this.Where.get(Key) ?? []) {
      if (this.Seen[P] === Stamp) continue;
      this.Seen[P] = Stamp;
      const Ids = this.Pieces[P];
      const At = FirstPairIndex(Ids, A, B);
      if (At < 0) continue;
      Kept.push(P);
      if (P * this.OffsetSpan >= Best) continue;
      let Offset = 0;
      for (let I = 0; I < At; I++) Offset += this.ByteLength[Ids[I]];
      Best = Math.min(Best, P * this.OffsetSpan + Offset);
    }
    this.Where.set(Key, Kept);
    return Best;
  }

  private RebuildHeap(): void {
    this.Heap.Clear();
    for (const [Key, Count] of this.Counts) this.Heap.Push(Count, this.Orders.get(Key) ?? 0, Key);
  }
}

/** Index of the first adjacent (A, B) in Ids, or -1. */
function FirstPairIndex(Ids: readonly number[], A: number, B: number): number {
  for (let I = 0; I + 1 < Ids.length; I++) if (Ids[I] === A && Ids[I + 1] === B) return I;
  return -1;
}

/** Ids with every (A, B) replaced by Z, scanning left to right without overlap (the reference rule). */
export function ReplacePair(Ids: readonly number[], A: number, B: number, Z: number): number[] {
  const Next: number[] = [];
  let I = 0;
  while (I < Ids.length) {
    if (I + 1 < Ids.length && Ids[I] === A && Ids[I + 1] === B) {
      Next.push(Z);
      I += 2;
    } else {
      Next.push(Ids[I]);
      I += 1;
    }
  }
  return Next;
}
