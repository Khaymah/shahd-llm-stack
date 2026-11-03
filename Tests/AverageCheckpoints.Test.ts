import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { BuildCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint, ParseCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import { ChecksumPayload, DecodeFloat64 } from "../Brain/Checkpoint/CheckpointFormat.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { AverageCheckpoints } from "../Brain/Checkpoint/AverageCheckpoints.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

const Tiny: ConfigOverride = { Model: { EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2 } };
const Tokenizer = { Kind: "Char" as const, Chars: "abcdefghij".split("") };

/** A worker checkpoint whose every weight and moment is filled with Value. */
function Worker(Value: number, Step: number, Override: ConfigOverride = Tiny): Checkpoint {
  const Config = LoadConfig({ Overrides: Override, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  const Model = new Shahd(Config, Rng.InitRng);
  const Opt = CreateOptimizer(Model.Parameters(), Config);
  for (const P of Model.Parameters()) P.Data.fill(Value);
  for (const M of Opt.M) M.fill(Value / 10);
  for (const V of Opt.V) V.fill(Value / 100);
  Opt.StepCount = Step;
  return BuildCheckpoint(Model, Opt, Rng, { Step }, Tokenizer);
}

test("weights and Adam moments are the element-wise mean; the step is the furthest worker's", () => {
  const { Ckpt, Report } = AverageCheckpoints([Worker(1, 700), Worker(2, 700), Worker(6, 690)], { Round: 1 });
  expect(Report).toEqual({ Workers: 3, WorkerSteps: [700, 700, 690], Step: 700 });
  for (const P of Ckpt.Params) expect([...DecodeFloat64(P.Data)].every((X) => Math.abs(X - 3) < 1e-12)).toBe(true);
  expect([...DecodeFloat64(Ckpt.Optimizer.M[0])].every((X) => Math.abs(X - 0.3) < 1e-12)).toBe(true);
  expect([...DecodeFloat64(Ckpt.Optimizer.V[0])].every((X) => Math.abs(X - 0.03) < 1e-12)).toBe(true);
  expect(Ckpt.Optimizer.StepCount).toBe(700);
  expect(Ckpt.Meta["Step"]).toBe(700);
  expect(Ckpt.Meta["Round"]).toBe(1);
  expect(Ckpt.Checksum).toBe(ChecksumPayload(Ckpt.Params, Ckpt.Optimizer, Ckpt.Config));
});

test("the averaged checkpoint survives a save/parse round trip and loads into a model", () => {
  const { Ckpt } = AverageCheckpoints([Worker(1, 10), Worker(3, 10)]);
  const Parsed = ParseCheckpoint(JSON.stringify(Ckpt)); // verifies the checksum
  const Config = LoadConfig({ Overrides: Tiny, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  const Model = new Shahd(Config, Rng.InitRng);
  const Opt = CreateOptimizer(Model.Parameters(), Config);
  ApplyCheckpoint(Parsed, Model, Opt, Rng);
  expect([...Model.Parameters()[0].Data].every((X) => Math.abs(X - 2) < 1e-6)).toBe(true);
});

test("incompatible workers are refused instead of averaged", () => {
  const Wider: ConfigOverride = { Model: { ...Tiny.Model, EmbedDim: 16 } };
  expect(() => AverageCheckpoints([Worker(1, 5), Worker(1, 5, Wider)])).toThrow(/config/);
  const OtherTokenizer = { ...Worker(1, 5), TokenizerState: { Kind: "Char", Chars: ["x"] } };
  expect(() => AverageCheckpoints([Worker(1, 5), OtherTokenizer])).toThrow(/tokenizer/);
  expect(() => AverageCheckpoints([])).toThrow(/no checkpoints/);
});
