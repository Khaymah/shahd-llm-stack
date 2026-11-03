// Offline BPE merge training: learn a merge table from a corpus. Works over UNIQUE pretoken pieces
// weighted by frequency.
//
//   TrainBpe          — the production trainer: incremental pair statistics + a lazy max-heap
//                       (Internal/IncrementalPairStats.ts). A merge touches only the pieces holding the
//                       merged pair, so 16K merges over tens of MB run in minutes on a CPU.
//   TrainBpeReference — the original straightforward trainer (recount every pair per merge). Kept as the
//                       executable specification: TrainBpe must produce the IDENTICAL merge list
//                       (same tie-break), which Tests/Bpe.Test.ts asserts, so merge tables trained
//                       before the switch stay reproducible.
//
// Corpus: one string (pretokenized as a whole — the historical behavior) or an array of documents
// (pretokenized one by one, so no chunk spans two documents and no giant joined string is built).

import { CodePretokenize } from "./CodePretokenizer.ts";
import type { BpeModel } from "./BytePairEncoder.ts";
import { AssertPairKeyCapacity, PairLeft, PairRight } from "./Internal/PairKey.ts";
import { ByteVocab, IncrementalPairStats, ReplacePair } from "./Internal/IncrementalPairStats.ts";
import type { PieceSet } from "./Internal/IncrementalPairStats.ts";

const Utf8Encoder = new TextEncoder();

export type Pretokenizer = (Text: string) => string[];
export type BpeCorpus = string | readonly string[];

/** Unique pretoken pieces in first-occurrence order, as UTF-8 byte ids, with their frequencies. */
export function CollectPieces(Corpus: BpeCorpus, Pretokenize: Pretokenizer = CodePretokenize): PieceSet {
  const Frequency = new Map<string, number>();
  const Documents = typeof Corpus === "string" ? [Corpus] : Corpus;
  for (const Document of Documents) {
    for (const Chunk of Pretokenize(Document)) Frequency.set(Chunk, (Frequency.get(Chunk) ?? 0) + 1);
  }
  const Ids: number[][] = [];
  const Weights: number[] = [];
  for (const [Chunk, Count] of Frequency) {
    Ids.push(Array.from(Utf8Encoder.encode(Chunk)));
    Weights.push(Count);
  }
  return { Ids, Weights };
}

function AssertMergeCount(NumMerges: number): void {
  if (!Number.isInteger(NumMerges) || NumMerges < 0) throw new Error(`TrainBpe: NumMerges must be a non-negative integer, got ${NumMerges}`);
  AssertPairKeyCapacity(ByteVocab + NumMerges, "TrainBpe");
}

/** Learn up to NumMerges merges (fewer when the corpus runs out of adjacent pairs). */
export function TrainBpe(Corpus: BpeCorpus, NumMerges: number, Pretokenize: Pretokenizer = CodePretokenize): BpeModel {
  AssertMergeCount(NumMerges);
  const Stats = new IncrementalPairStats(CollectPieces(Corpus, Pretokenize));
  const Merges: [number, number][] = [];
  while (Merges.length < NumMerges) {
    const Best = Stats.SelectBest();
    if (Best === null) break;
    Merges.push([PairLeft(Best), PairRight(Best)]);
    Stats.Merge(Best);
  }
  return { Merges };
}

/** The original trainer: recount every pair each merge. Slow; the specification TrainBpe must match. */
export function TrainBpeReference(Corpus: BpeCorpus, NumMerges: number, Pretokenize: Pretokenizer = CodePretokenize): BpeModel {
  AssertMergeCount(NumMerges);
  const { Ids: Pieces, Weights } = CollectPieces(Corpus, Pretokenize);
  const Merges: [number, number][] = [];
  for (let M = 0; M < NumMerges; M++) {
    // Weighted adjacent-pair counts across all unique pieces, in (piece, position) insertion order.
    const PairCount = new Map<string, number>();
    for (let P = 0; P < Pieces.length; P++) {
      const Ids = Pieces[P];
      for (let I = 0; I < Ids.length - 1; I++) {
        const Key = `${Ids[I]},${Ids[I + 1]}`;
        PairCount.set(Key, (PairCount.get(Key) ?? 0) + Weights[P]);
      }
    }
    if (PairCount.size === 0) break;
    let BestKey = "";
    let BestCount = -1;
    for (const [Key, Count] of PairCount) {
      if (Count > BestCount) {
        BestCount = Count;
        BestKey = Key;
      }
    }
    const Parts = BestKey.split(",");
    const A = Number(Parts[0]);
    const B = Number(Parts[1]);
    const NewId = ByteVocab + Merges.length;
    Merges.push([A, B]);
    for (let P = 0; P < Pieces.length; P++) Pieces[P] = ReplacePair(Pieces[P], A, B, NewId);
  }
  return { Merges };
}
