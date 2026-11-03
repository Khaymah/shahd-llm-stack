import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CrossEntropy } from "../Brain/Ops/OpsBarrel.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { CachedGenerate } from "../Brain/Sampling/CachedForward.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

function Build(Model: ConfigOverride["Model"]): Shahd {
  const Config = LoadConfig({ Overrides: { Model }, UseCli: false, UseEnv: false });
  return new Shahd(Config, CreateRngStreams(Config.Training.Seed).InitRng);
}

function GradcheckModel(Model: Shahd): boolean {
  const Ids = [1, 4, 2, 7, 3];
  const Targets = [4, 2, 7, 3, 6];
  return GradCheck(Model.Parameters(), () => CrossEntropy(Model.Forward(Ids), Targets), { Tolerance: 1e-3 }).Passed;
}

test("full-model backward gradchecks with the modern stack (RoPE + RMSNorm + SwiGLU)", () => {
  const Model = Build({ EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 9, MlpRatio: 2, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu", WeightTying: true });
  expect(GradcheckModel(Model)).toBe(true);
});

test("full-model backward gradchecks with GeGLU + RMSNorm", () => {
  const Model = Build({ EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 9, MlpRatio: 2, NormKind: "RmsNorm", MlpKind: "GeGlu" });
  expect(GradcheckModel(Model)).toBe(true);
});

test("RoPE config validation rejects an odd head dim", () => {
  expect(() =>
    LoadConfig({ Overrides: { Model: { EmbedDim: 12, NumHeads: 4, PositionScheme: "Rope" } }, UseCli: false, UseEnv: false }),
  ).toThrow();
});

test("full-model backward gradchecks with GQA (KvHeads < NumHeads)", () => {
  const Model = Build({ EmbedDim: 16, NumLayers: 1, NumHeads: 4, KvHeads: 2, BlockSize: 8, VocabSize: 9, MlpRatio: 2 });
  expect(GradcheckModel(Model)).toBe(true);
});

test("GQA validation rejects KvHeads not dividing NumHeads", () => {
  expect(() =>
    LoadConfig({ Overrides: { Model: { NumHeads: 4, KvHeads: 3 } }, UseCli: false, UseEnv: false }),
  ).toThrow();
});

test("modern-stack model generates; cached path matches the uncached path", () => {
  const Model = Build({ EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" });
  const Out = Generate(Model, [1, 2, 3], 4, DefaultSampling, new SeededRng(1));
  expect(Out.length).toBe(7);
  const Greedy = { ...DefaultSampling, Temperature: 0 };
  expect(CachedGenerate(Model, [1, 2, 3], 4, Greedy, new SeededRng(1))).toEqual(Generate(Model, [1, 2, 3], 4, Greedy, new SeededRng(1)));
});
