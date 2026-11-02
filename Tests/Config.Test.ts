import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { DropoutNotImplementedMessage } from "../Brain/Config/ValidateConfig.ts";

const Isolated = { UseCli: false, UseEnv: false } as const;

test("LoadConfig derives HeadDim and AttentionScale from the single home (L4-safe)", () => {
  const Config = LoadConfig({ ConfigPath: "Configs/Phase1Small.Config.json", ...Isolated });
  expect(Config.Model.EmbedDim).toBe(128);
  expect(Config.Model.NumHeads).toBe(4);
  expect(Config.Derived.HeadDim).toBe(32);
  expect(Config.Derived.AttentionScale).toBeCloseTo(1 / Math.sqrt(32), 12);
  expect(Config.Derived.MlpHidden).toBe(512);
});

test("LoadConfig rejects EmbedDim not divisible by NumHeads (L4 guard)", () => {
  expect(() =>
    LoadConfig({ Overrides: { Model: { EmbedDim: 10, NumHeads: 3 } }, ...Isolated }),
  ).toThrow();
});

test("resolved config is deep-frozen", () => {
  const Config = LoadConfig(Isolated);
  expect(Object.isFrozen(Config)).toBe(true);
  expect(Object.isFrozen(Config.Model)).toBe(true);
  expect(Object.isFrozen(Config.Derived)).toBe(true);
});

test("ConfigHash is stable for identical inputs and changes with content", () => {
  const A = LoadConfig(Isolated);
  const B = LoadConfig(Isolated);
  expect(A.ConfigHash).toBe(B.ConfigHash);
  const C = LoadConfig({ Overrides: { Model: { NumLayers: 8 } }, ...Isolated });
  expect(C.ConfigHash).not.toBe(A.ConfigHash);
});

test("scaled residual init shrinks the init scale with depth", () => {
  const Config = LoadConfig({
    Overrides: { Model: { UseScaledResidualInit: true, InitScale: 0.02, NumLayers: 2 } },
    ...Isolated,
  });
  expect(Config.Derived.ResidualInitScale).toBeCloseTo(0.02 / Math.sqrt(4), 12);
});

test("Adam betas must lie strictly inside (0, 1)", () => {
  for (const Beta of [1, 0, -0.1, 1.5]) {
    expect(() => LoadConfig({ Overrides: { Optimizer: { Beta1: Beta } }, ...Isolated })).toThrow(/Beta1/);
    expect(() => LoadConfig({ Overrides: { Optimizer: { Beta2: Beta } }, ...Isolated })).toThrow(/Beta2/);
  }
  expect(() => LoadConfig({ Overrides: { Optimizer: { Beta1: 0.9, Beta2: 0.999 } }, ...Isolated })).not.toThrow();
});

test("Dropout is reserved: any value other than 0 is rejected with an explicit message", () => {
  expect(() => LoadConfig({ Overrides: { Model: { Dropout: 0.1 } }, ...Isolated })).toThrow(DropoutNotImplementedMessage);
  expect(() => LoadConfig({ Overrides: { Model: { Dropout: 0 } }, ...Isolated })).not.toThrow();
});

test("NaN/Infinity are rejected wherever they would poison training", () => {
  const Bad = [Number.NaN, Number.POSITIVE_INFINITY];
  for (const Value of Bad) {
    expect(() => LoadConfig({ Overrides: { Optimizer: { LearningRate: Value } }, ...Isolated })).toThrow(/LearningRate/);
    expect(() => LoadConfig({ Overrides: { Optimizer: { WeightDecay: Value } }, ...Isolated })).toThrow(/WeightDecay/);
    expect(() => LoadConfig({ Overrides: { Optimizer: { GradClipNorm: Value } }, ...Isolated })).toThrow(/GradClipNorm/);
    expect(() => LoadConfig({ Overrides: { Optimizer: { Epsilon: Value } }, ...Isolated })).toThrow(/Epsilon/);
    expect(() => LoadConfig({ Overrides: { Model: { InitScale: Value } }, ...Isolated })).toThrow(/InitScale/);
    expect(() => LoadConfig({ Overrides: { Schedule: { WarmupSteps: Value } }, ...Isolated })).toThrow(/WarmupSteps/);
  }
});

test("WarmupSteps may not exceed MaxSteps", () => {
  expect(() => LoadConfig({ Overrides: { Schedule: { WarmupSteps: 11, MaxSteps: 10 } }, ...Isolated })).toThrow(/WarmupSteps/);
  expect(() => LoadConfig({ Overrides: { Schedule: { WarmupSteps: 10, MaxSteps: 10 } }, ...Isolated })).not.toThrow();
});

test("Schedule.DecayFraction defaults to 0.1 and must lie strictly inside (0, 0.5)", () => {
  expect(LoadConfig(Isolated).Schedule.DecayFraction).toBe(0.1);
  for (const Bad of [0, -0.1, 0.5, 0.9, Number.NaN]) {
    expect(() => LoadConfig({ Overrides: { Schedule: { DecayFraction: Bad } }, ...Isolated })).toThrow(/DecayFraction/);
  }
  expect(() => LoadConfig({ Overrides: { Schedule: { Kind: "Wsd", DecayFraction: 0.49 } }, ...Isolated })).not.toThrow();
});

test("Wsd refuses a warmup that runs into the decay window", () => {
  expect(() => LoadConfig({ Overrides: { Schedule: { Kind: "Wsd", WarmupSteps: 95, MaxSteps: 100, DecayFraction: 0.1 } }, ...Isolated })).toThrow(/decay window/);
  expect(() => LoadConfig({ Overrides: { Schedule: { Kind: "Wsd", WarmupSteps: 90, MaxSteps: 100, DecayFraction: 0.1 } }, ...Isolated })).not.toThrow();
  // The same warmup is fine for Cosine (no decay window).
  expect(() => LoadConfig({ Overrides: { Schedule: { Kind: "Cosine", WarmupSteps: 95, MaxSteps: 100 } }, ...Isolated })).not.toThrow();
});
