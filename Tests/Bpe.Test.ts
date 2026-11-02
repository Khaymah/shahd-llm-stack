import { test, expect } from "bun:test";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TrainBpe, TrainBpeReference } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder, LoadBpeModel, SameMerges } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { CodePretokenize } from "../Brain/Tokenizer/CodePretokenizer.ts";
import { ChunkCache } from "../Brain/Tokenizer/Internal/ChunkCache.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

test("byte-level BPE round-trips ASCII and non-ASCII (no OOV)", () => {
  const Corpus =
    "function add(a, b) { return a + b; }\n".repeat(20) + "const cafe = 'a b c';\n".repeat(5);
  const Model = TrainBpe(Corpus, 100);
  const Encoder = new BytePairEncoder(Model);

  // Includes text never seen in training (unicode identifier, emoji) — byte-level must still round-trip.
  for (const Text of ["function add", "return a + b", "café", "日本語 🎉", "xyz_123"]) {
    expect(Encoder.Decode(Encoder.Encode(Text))).toBe(Text);
  }
  expect(Encoder.VocabSize).toBe(256 + Model.Merges.length);
});

test("BPE compresses frequently-seen text below its raw byte length", () => {
  const Corpus = "abcabcabc ".repeat(80);
  const Model = TrainBpe(Corpus, 50);
  const Encoder = new BytePairEncoder(Model);
  const Ids = Encoder.Encode("abcabc");
  expect(Ids.length).toBeLessThan(6); // "abcabc" is 6 raw bytes; merges shorten it
  expect(Encoder.Decode(Ids)).toBe("abcabc");
});

test("empty merge table degenerates to raw bytes", () => {
  const Encoder = new BytePairEncoder({ Merges: [] });
  expect(Encoder.VocabSize).toBe(256);
  const Ids = Encoder.Encode("Hi!");
  expect(Ids).toEqual([72, 105, 33]); // ASCII byte values
  expect(Encoder.Decode(Ids)).toBe("Hi!");
});

// --- the incremental trainer must reproduce the reference trainer merge-for-merge -----------------

const CodeCorpus = [
  "export function add(a: number, b: number): number { return a + b; }",
  "export const sum = (xs: number[]) => xs.reduce((acc, x) => acc + x, 0);",
  "class Stack<T> { private items: T[] = []; push(item: T) { this.items.push(item); } }",
  "    if (value === null || value === undefined) { throw new Error('missing value'); }",
  "// café, naïve, 日本語, emoji 🎉 — non-ASCII bytes must merge like any others",
  "for (let i = 0; i < 10; i++) { console.log(i * i); }",
].join("\n\n");

test("TrainBpe yields the IDENTICAL merge list to TrainBpeReference (code corpus)", () => {
  const Fast = TrainBpe(CodeCorpus.repeat(3), 300);
  const Reference = TrainBpeReference(CodeCorpus.repeat(3), 300);
  expect(Fast.Merges.length).toBeGreaterThan(100);
  expect(Fast.Merges).toEqual(Reference.Merges);
});

test("TrainBpe matches the reference on tie-heavy and degenerate corpora (until pairs run out)", () => {
  const Identity = (Text: string): string[] => [Text];
  for (const Corpus of ["aaaa aaa aa a", "abab ab ba baba", "xyxyxy yxyx", "a".repeat(50) + " " + "ab".repeat(30), "", "z"]) {
    expect(TrainBpe(Corpus, 80).Merges).toEqual(TrainBpeReference(Corpus, 80).Merges);
    expect(TrainBpe(Corpus, 80, Identity).Merges).toEqual(TrainBpeReference(Corpus, 80, Identity).Merges);
  }
});

test("TrainBpe matches the reference on randomized corpora, as one string and as documents", () => {
  const Rng = new SeededRng(17);
  const Alphabet = ["a", "b", "c", " ", "\n", "é", "_", "1"];
  for (let Trial = 0; Trial < 40; Trial++) {
    let Text = "";
    const Length = 50 + Math.floor(Rng.NextFloat() * 300);
    const Width = 2 + Math.floor(Rng.NextFloat() * (Alphabet.length - 1));
    for (let I = 0; I < Length; I++) Text += Alphabet[Math.floor(Rng.NextFloat() * Width)];
    expect(TrainBpe(Text, 150).Merges).toEqual(TrainBpeReference(Text, 150).Merges);
    const Documents = Text.split("\n");
    expect(TrainBpe(Documents, 150).Merges).toEqual(TrainBpeReference(Documents, 150).Merges);
  }
});

test("TrainBpe rejects a merge count the pair keys cannot represent", () => {
  expect(() => TrainBpe("abc", -1)).toThrow(/NumMerges/);
  expect(() => TrainBpe("abc", 1.5)).toThrow(/NumMerges/);
  expect(() => TrainBpe("abc", 2 ** 22)).toThrow(/pair-key capacity/);
});

// --- the faster encoder must emit bit-identical ids ------------------------------------------------

/** The original encoder rule (string pair keys, no cache) — the specification for Encode. */
function ReferenceEncode(Merges: readonly [number, number][], Text: string): number[] {
  const RankOf = new Map<string, number>();
  Merges.forEach((Pair, Rank) => RankOf.set(`${Pair[0]},${Pair[1]}`, Rank));
  const Out: number[] = [];
  for (const Chunk of CodePretokenize(Text)) {
    const Ids = Array.from(new TextEncoder().encode(Chunk));
    for (;;) {
      let BestRank = Infinity;
      let BestPos = -1;
      for (let I = 0; I < Ids.length - 1; I++) {
        const Rank = RankOf.get(`${Ids[I]},${Ids[I + 1]}`);
        if (Rank !== undefined && Rank < BestRank) {
          BestRank = Rank;
          BestPos = I;
        }
      }
      if (BestPos === -1) break;
      Ids.splice(BestPos, 2, 256 + BestRank);
    }
    Out.push(...Ids);
  }
  return Out;
}

test("Encode is bit-identical to the reference rule, cold and with a warm chunk cache", () => {
  const Model = TrainBpe(CodeCorpus.repeat(4), 250);
  const Encoder = new BytePairEncoder(Model);
  const Samples = [CodeCorpus, "function add", "return a + b", "café", "日本語 🎉", "xyz_123", " ".repeat(600), "x".repeat(1000)];
  for (const Text of Samples) {
    const Expected = ReferenceEncode(Model.Merges, Text);
    expect(Encoder.Encode(Text)).toEqual(Expected); // cold
    expect(Encoder.Encode(Text)).toEqual(Expected); // every chunk now served from the cache
    expect(Encoder.Decode(Expected)).toBe(Text);
  }
});

test("a returned id array is the caller's own: mutating it never corrupts the cache", () => {
  const Encoder = new BytePairEncoder(TrainBpe(CodeCorpus, 60));
  const First = Encoder.Encode("return value;");
  const Copy = [...First];
  First.length = 0;
  expect(Encoder.Encode("return value;")).toEqual(Copy);
});

test("ChunkCache stays within its capacity and keeps recently used chunks", () => {
  const Cache = new ChunkCache(8);
  for (let I = 0; I < 100; I++) {
    Cache.Set(`k${I}`, [I]);
    Cache.Get("k0"); // keep touching one key: it must survive every generation flip
    expect(Cache.Size).toBeLessThanOrEqual(8);
  }
  expect(Cache.Get("k0")).toEqual([0]);
  expect(Cache.Get("k1")).toBeUndefined();
  expect(() => new ChunkCache(1)).toThrow(/capacity/);
});

test("BytePairEncoder rejects a merge table referencing ids that do not exist yet", () => {
  expect(() => new BytePairEncoder({ Merges: [[256, 1]] })).toThrow(/does not exist yet/);
  expect(() => new BytePairEncoder({ Merges: [[1, 2], [257, 3]] })).toThrow(/merge #1/);
  expect(() => new BytePairEncoder({ Merges: [[1, 2], [256, 3]] })).not.toThrow();
});

// --- persisted tokenizer file ---------------------------------------------------------------------

test("LoadBpeModel round-trips a tokenizer file and validates it", () => {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-bpe-"));
  try {
    const Merges = TrainBpe(CodeCorpus, 40).Merges;
    const File = { Kind: "Bpe", Version: "v1", Merges, TrainedOn: { SampleMb: 0.001, PerSlice: { ts: { Mb: 0.001, Docs: 6, TargetMb: 1 } }, Source: "corpus:inline" }, CreatedAt: "2026-11-02T00:00:00.000Z" };
    const Good = join(Dir, "Tokenizer.v1.json");
    writeFileSync(Good, JSON.stringify(File));
    const Loaded = LoadBpeModel(Good);
    expect(Loaded.Version).toBe("v1");
    expect(SameMerges(Loaded.Merges, Merges)).toBe(true);
    expect(new BytePairEncoder(Loaded).Encode("return a + b")).toEqual(new BytePairEncoder({ Merges }).Encode("return a + b"));

    const WrongKind = join(Dir, "Char.json");
    writeFileSync(WrongKind, JSON.stringify({ ...File, Kind: "Char" }));
    expect(() => LoadBpeModel(WrongKind)).toThrow(/not a Bpe tokenizer file/);
    const Forward = join(Dir, "Forward.json");
    writeFileSync(Forward, JSON.stringify({ ...File, Merges: [[300, 1]] }));
    expect(() => LoadBpeModel(Forward)).toThrow(/does not exist yet/);
    const Broken = join(Dir, "Broken.json");
    writeFileSync(Broken, "{ not json");
    expect(() => LoadBpeModel(Broken)).toThrow(/cannot read/);
    expect(() => LoadBpeModel(join(Dir, "Missing.json"))).toThrow(/cannot read/);
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
});

test("SameMerges compares order and content", () => {
  expect(SameMerges([[1, 2], [3, 4]], [[1, 2], [3, 4]])).toBe(true);
  expect(SameMerges([[1, 2], [3, 4]], [[3, 4], [1, 2]])).toBe(false);
  expect(SameMerges([[1, 2]], [[1, 2], [3, 4]])).toBe(false);
});
