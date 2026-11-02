import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { WithTapeOff } from "../Brain/Tensor/Tape.ts";
import { KvCache } from "../Brain/Sampling/KvCache.ts";
import { CachedDecoder, CachedForwardStep, CachedGenerate, CachedGenerateAsync } from "../Brain/Sampling/CachedForward.ts";
import { Generate, GenerateAsync } from "../Brain/Sampling/Generate.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

const LogitTolerance = 1e-9;
const BlockSize = 8;
const VocabSize = 12;
const Greedy = { Temperature: 0, TopK: 0, TopP: 1 };
const Shared = { EmbedDim: 16, NumLayers: 2, BlockSize, VocabSize, MlpRatio: 2 };

// Every architecture axis the cached path must honor, including the four required combinations.
const Architectures: [string, ConfigOverride["Model"]][] = [
  ["Learned/LayerNorm/Relu/MHA", { ...Shared, NumHeads: 2 }],
  ["Learned/LayerNorm/Relu/MHA untied", { ...Shared, NumHeads: 1, WeightTying: false }],
  ["Rope/RmsNorm/SwiGlu/MHA", { ...Shared, NumHeads: 4, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" }],
  ["Rope/RmsNorm/SwiGlu/GQA 2-of-4", { ...Shared, NumHeads: 4, KvHeads: 2, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" }],
  ["Rope/LayerNorm/Relu", { ...Shared, NumHeads: 2, PositionScheme: "Rope" }],
  ["Learned/RmsNorm/GeGlu/GQA 1-of-2", { ...Shared, NumHeads: 2, KvHeads: 1, NormKind: "RmsNorm", MlpKind: "GeGlu" }],
];

function Build(Model: ConfigOverride["Model"]): Shahd {
  const Config = LoadConfig({ Overrides: { Model }, UseCli: false, UseEnv: false });
  return new Shahd(Config, CreateRngStreams(Config.Training.Seed).InitRng);
}

/** Last row of the full Tensor forward over the trailing BlockSize window — what Generate samples from. */
function FullLastRow(Model: Shahd, Ids: number[]): Float64Array {
  const Logits = WithTapeOff(() => Model.Forward(Ids.slice(-BlockSize)));
  return Float64Array.from(Logits.Data.subarray((Logits.Rows - 1) * VocabSize));
}

function MaxAbsDiff(A: ArrayLike<number>, B: ArrayLike<number>): number {
  expect(A.length).toBe(B.length);
  let Max = 0;
  for (let I = 0; I < A.length; I++) Max = Math.max(Max, Math.abs(A[I] - B[I]));
  return Max;
}

for (const [Label, Arch] of Architectures) {
  test(`[${Label}] every cached step's logits match the full forward's row`, () => {
    const Model = Build(Arch);
    const Seq = [1, 5, 2, 8, 3, 0, 7, 11];
    const Full = WithTapeOff(() => Model.Forward(Seq));
    const Cache = KvCache.ForModel(Model);
    for (let P = 0; P < Seq.length; P++) {
      const Logits = CachedForwardStep(Model, Cache, Seq[P], P);
      expect(MaxAbsDiff(Logits, Full.Data.subarray(P * VocabSize, (P + 1) * VocabSize))).toBeLessThan(LogitTolerance);
    }
  });

  test(`[${Label}] decoder logits match the sliding-window forward past BlockSize`, () => {
    const Model = Build(Arch);
    const Rng = new SeededRng(17);
    const Seq = Array.from({ length: 3 * BlockSize + 2 }, () => Math.floor(Rng.NextFloat() * VocabSize));
    const Decoder = new CachedDecoder(Model, Seq.slice(0, 2));
    expect(MaxAbsDiff(Decoder.Logits, FullLastRow(Model, Seq.slice(0, 2)))).toBeLessThan(LogitTolerance);
    for (let Len = 3; Len <= Seq.length; Len++) {
      Decoder.Push(Seq[Len - 1]);
      expect(MaxAbsDiff(Decoder.Logits, FullLastRow(Model, Seq.slice(0, Len)))).toBeLessThan(LogitTolerance);
    }
  });

  test(`[${Label}] CachedGenerate ids === Generate ids (greedy), well past BlockSize`, () => {
    const Model = Build(Arch);
    for (const Prompt of [[1, 5, 2], [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]]) { // second prompt > BlockSize
      const NewTokens = 2 * BlockSize + 3;
      const Uncached = Generate(Model, Prompt, NewTokens, Greedy, new SeededRng(1));
      const Cached = CachedGenerate(Model, Prompt, NewTokens, Greedy, new SeededRng(1));
      expect(Cached.length).toBe(Prompt.length + NewTokens);
      expect(Cached).toEqual(Uncached);
    }
  });
}

test("CachedGenerate matches Generate under stochastic sampling (same RNG draws)", () => {
  const Model = Build(Architectures[3][1]);
  const Options = { Temperature: 0.9, TopK: 5, TopP: 0.9 };
  const Uncached = Generate(Model, [2, 7], 20, Options, new SeededRng(5));
  const Cached = CachedGenerate(Model, [2, 7], 20, Options, new SeededRng(5));
  expect(Cached).toEqual(Uncached);
});

test("CachedGenerateAsync streams the same ids as CachedGenerate and honors ShouldStop", async () => {
  const Model = Build(Architectures[2][1]);
  const Streamed: number[] = [];
  const Ids = await CachedGenerateAsync(Model, [1, 2], 10, Greedy, new SeededRng(1), (Id) => Streamed.push(Id));
  expect(Ids).toEqual(CachedGenerate(Model, [1, 2], 10, Greedy, new SeededRng(1)));
  expect(Streamed).toEqual(Ids.slice(2));
  const Stopped = await CachedGenerateAsync(Model, [1, 2], 10, Greedy, new SeededRng(1), () => undefined, () => true);
  expect(Stopped).toEqual([1, 2]);
});

test("an empty prompt is rejected by every generation entry point", async () => {
  const Model = Build(Architectures[0][1]);
  const Rng = new SeededRng(1);
  expect(() => Generate(Model, [], 3, Greedy, Rng)).toThrow(/empty prompt/);
  expect(() => CachedGenerate(Model, [], 3, Greedy, Rng)).toThrow(/empty prompt/);
  await expect(GenerateAsync(Model, [], 3, Greedy, Rng, () => undefined)).rejects.toThrow(/empty prompt/);
  await expect(CachedGenerateAsync(Model, [], 3, Greedy, Rng, () => undefined)).rejects.toThrow(/empty prompt/);
});

test("cached path validates token ids and position contiguity", () => {
  const Model = Build(Architectures[2][1]);
  const Cache = KvCache.ForModel(Model);
  expect(() => CachedForwardStep(Model, Cache, VocabSize, 0)).toThrow(/out of range/);
  expect(() => CachedForwardStep(Model, Cache, -1, 0)).toThrow(/out of range/);
  expect(() => CachedForwardStep(Model, Cache, 1.5, 0)).toThrow(/out of range/);
  expect(() => CachedForwardStep(Model, Cache, 1, 3)).toThrow(/contiguous/);
  expect(() => CachedGenerate(Model, [1, VocabSize], 2, Greedy, new SeededRng(1))).toThrow(/out of range/);
  expect(Cache.Length).toBe(0); // nothing half-appended by the rejected calls
});

test("GQA cache holds one slot per KV head, not per query head", () => {
  const Model = Build(Architectures[3][1]);
  const Cache = KvCache.ForModel(Model);
  expect(Cache.KvHeads).toBe(2);
  CachedForwardStep(Model, Cache, 3, 0);
  expect(Cache.GetKeys(0, 1).length).toBe(Model.Config.Derived.HeadDim);
  expect(() => Cache.GetKeys(0, 2)).toThrow();
});

test("MaxNewTokens = 0 returns the prompt untouched on both paths", () => {
  const Model = Build(Architectures[0][1]);
  expect(CachedGenerate(Model, [4, 2], 0, Greedy, new SeededRng(1))).toEqual([4, 2]);
  expect(Generate(Model, [4, 2], 0, Greedy, new SeededRng(1))).toEqual([4, 2]);
});
