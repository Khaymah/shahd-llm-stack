import { test, expect } from "bun:test";
import {
  LshBandLayout,
  MinHashSignature,
  NearDuplicateGroups,
  NearDuplicateGroupsBruteForce,
  NearDuplicateGroupsFromSignatures,
} from "../Brain/Data/NearDedup.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

const Vocabulary = [
  "const", "let", "function", "return", "if", "else", "for", "while", "class", "new", "this",
  "await", "async", "import", "export", "value", "index", "count", "result", "error", "data",
  "items", "push", "map", "filter", "reduce", "length", "string", "number", "boolean", "null",
];

function RandomDoc(Rng: SeededRng, Words: number): string[] {
  const Out: string[] = [];
  for (let W = 0; W < Words; W++) Out.push(Vocabulary[Rng.NextUint32() % Vocabulary.length] + String(Rng.NextUint32() % 50));
  return Out;
}

/** 2,000 docs: ~60% random originals, ~40% planted near-duplicates of an EARLIER doc with 0-12
 *  word edits — so true similarities spread across the whole 0.5..1.0 range the thresholds probe. */
function PlantedCorpus(Count: number, Seed: number): string[] {
  const Rng = new SeededRng(Seed);
  const Docs: string[][] = [];
  for (let I = 0; I < Count; I++) {
    if (Docs.length > 0 && Rng.NextFloat() < 0.4) {
      const Copy = [...Docs[Rng.NextUint32() % Docs.length]];
      const Edits = Rng.NextUint32() % 13;
      for (let E = 0; E < Edits; E++) Copy[Rng.NextUint32() % Copy.length] = RandomDoc(Rng, 1)[0];
      Docs.push(Copy);
    } else {
      Docs.push(RandomDoc(Rng, 40));
    }
  }
  return Docs.map((Words) => Words.join(" "));
}

test("LshBandLayout is pigeonhole-exact: D + 1 bands for D allowed mismatches, covering every position", () => {
  const Layout = LshBandLayout(64, 0.8);
  expect(Layout.MinAgreements).toBe(52); // 52/64 = 0.8125 >= 0.8 > 51/64
  expect(Layout.Bands).toBe(13);
  expect(Layout.BandStarts[0]).toBe(0);
  expect(Layout.BandStarts[Layout.Bands]).toBe(64);
  for (let B = 0; B < Layout.Bands; B++) expect(Layout.BandStarts[B + 1]).toBeGreaterThan(Layout.BandStarts[B]);
  expect(LshBandLayout(64, 1).Bands).toBe(1); // exact equality: one band = the whole signature
  expect(LshBandLayout(64, 0).Bands).toBe(0); // everything matches: banding cannot prune
  expect(LshBandLayout(64, 1.5).MinAgreements).toBe(65); // unreachable
});

test("LSH groups equal brute-force groups exactly on 2,000 docs with planted near-duplicates", () => {
  const Docs = PlantedCorpus(2000, 77);
  const Sigs = Docs.map((D) => MinHashSignature(D, 64));
  for (const Threshold of [0.5, 0.7, 0.8, 0.9, 1]) {
    const Brute = NearDuplicateGroupsBruteForce(Sigs, Threshold);
    const Lsh = NearDuplicateGroupsFromSignatures(Sigs, Threshold);
    expect(Lsh).toEqual(Brute);
    // Non-trivial: the planted duplicates really form multi-member groups at every threshold.
    expect(Lsh.filter((G) => G.length > 1).length).toBeGreaterThan(50);
  }
  // Public API path (docs -> signatures -> LSH) agrees with the reference too.
  expect(NearDuplicateGroups(Docs.slice(0, 300), 0.8)).toEqual(NearDuplicateGroupsBruteForce(Sigs.slice(0, 300), 0.8));
}, 60_000);

test("degenerate inputs: empty corpus, NaN/unreachable/zero thresholds, mismatched lengths", () => {
  expect(NearDuplicateGroupsFromSignatures([], 0.8)).toEqual([]);
  const Sigs = [[1, 2, 3, 4], [1, 2, 3, 5], [9, 9, 9, 9]];
  expect(NearDuplicateGroupsFromSignatures(Sigs, Number.NaN)).toEqual(NearDuplicateGroupsBruteForce(Sigs, Number.NaN));
  expect(NearDuplicateGroupsFromSignatures(Sigs, 2)).toEqual([[0], [1], [2]]);
  expect(NearDuplicateGroupsFromSignatures(Sigs, 0)).toEqual([[0, 1, 2]]);
  expect(NearDuplicateGroupsFromSignatures(Sigs, 0.75)).toEqual([[0, 1], [2]]);
  expect(() => NearDuplicateGroupsFromSignatures([[1, 2], [1]], 0.8)).toThrow(/must share NumHashes/);
});

test("20,000 signatures group in well under 5 s (the brute force would need 2e8 comparisons)", () => {
  const Rng = new SeededRng(5);
  const Sigs: number[][] = [];
  for (let I = 0; I < 20_000; I++) {
    if (I > 0 && Rng.NextFloat() < 0.2) {
      const Copy = [...Sigs[Rng.NextUint32() % I]];
      for (let E = Rng.NextUint32() % 10; E > 0; E--) Copy[Rng.NextUint32() % 64] = Rng.NextUint32();
      Sigs.push(Copy);
    } else {
      const Sig: number[] = [];
      for (let P = 0; P < 64; P++) Sig.push(Rng.NextUint32());
      Sigs.push(Sig);
    }
  }
  const Start = performance.now();
  const Groups = NearDuplicateGroupsFromSignatures(Sigs, 0.8);
  const ElapsedMs = performance.now() - Start;
  expect(ElapsedMs).toBeLessThan(5000);
  expect(Groups.reduce((Sum, G) => Sum + G.length, 0)).toBe(20_000); // every doc in exactly one group
  expect(Groups.some((G) => G.length > 1)).toBe(true);
}, 30_000);
