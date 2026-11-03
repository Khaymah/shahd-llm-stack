import { test, expect, spyOn } from "bun:test";
import { rmSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { CrossEntropy } from "../Brain/Ops/OpsBarrel.ts";
import { Backward } from "../Brain/Autograd/Backward.ts";
import { SaveCheckpoint, BuildCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { LoadRunnableModelFrom } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { DefaultConfig } from "../Brain/Config/Constants.ts";
import { LoadCheckpoint, ApplyCheckpoint, ApplyCheckpointWeights } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

const CkptPath = "Checkpoints/CheckpointTest.ckpt";
const BaseOverride: ConfigOverride = {
  Model: { EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2 },
};

function Build(Override: ConfigOverride) {
  const Config = LoadConfig({ Overrides: Override, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  const Model = new Shahd(Config, Rng.InitRng);
  const Opt = CreateOptimizer(Model.Parameters(), Config);
  return { Config, Rng, Model, Opt };
}

test("checkpoint round-trips weights, optimizer state, and RNG", () => {
  const A = Build(BaseOverride);
  const Ids = [1, 4, 2, 5, 3];
  const Targets = [4, 2, 5, 3, 6];
  for (let S = 0; S < 3; S++) {
    A.Opt.ZeroGrad();
    Backward(CrossEntropy(A.Model.Forward(Ids), Targets));
    A.Opt.Step(0.01);
  }
  for (let I = 0; I < 10; I++) A.Rng.DataRng.NextFloat();
  SaveCheckpoint(CkptPath, A.Model, A.Opt, A.Rng, { Step: 3 });

  const B = Build(BaseOverride);
  ApplyCheckpoint(LoadCheckpoint(CkptPath), B.Model, B.Opt, B.Rng);

  const PA = A.Model.Parameters();
  const PB = B.Model.Parameters();
  expect(PB.length).toBe(PA.length);
  for (let I = 0; I < PA.length; I++) {
    expect(Array.from(PB[I].Data)).toEqual(Array.from(PA[I].Data));
  }
  expect(B.Opt.StepCount).toBe(A.Opt.StepCount);
  expect(Array.from(B.Opt.M[0])).toEqual(Array.from(A.Opt.M[0]));
  expect(Array.from(B.Opt.V[0])).toEqual(Array.from(A.Opt.V[0]));
  expect(B.Rng.DataRng.GetState()).toBe(A.Rng.DataRng.GetState());

  rmSync(CkptPath, { force: true });
});

test("ApplyCheckpointWeights transfers weights ONLY (warm start keeps a fresh optimizer + RNG)", () => {
  const A = Build(BaseOverride);
  const Ids = [1, 4, 2, 5, 3];
  const Targets = [4, 2, 5, 3, 6];
  for (let S = 0; S < 3; S++) {
    A.Opt.ZeroGrad();
    Backward(CrossEntropy(A.Model.Forward(Ids), Targets));
    A.Opt.Step(0.01);
  }
  SaveCheckpoint(CkptPath, A.Model, A.Opt, A.Rng, { Step: 3 });

  const B = Build(BaseOverride);
  ApplyCheckpointWeights(LoadCheckpoint(CkptPath), B.Model);

  const PA = A.Model.Parameters();
  const PB = B.Model.Parameters();
  for (let I = 0; I < PA.length; I++) expect(Array.from(PB[I].Data)).toEqual(Array.from(PA[I].Data));
  // The SFT stage owns its own optimizer/schedule: nothing from the pretraining run leaks in.
  expect(B.Opt.StepCount).toBe(0);
  expect(Array.from(B.Opt.M[0])).not.toEqual(Array.from(A.Opt.M[0]));

  // Same hard shape guard as the full apply — a wrong-architecture base can never seed a run.
  const C = Build({ Model: { EmbedDim: 16, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2 } });
  expect(() => ApplyCheckpointWeights(LoadCheckpoint(CkptPath), C.Model)).toThrow(/mismatch/);

  rmSync(CkptPath, { force: true });
});

test("ApplyCheckpoint hard-fails on architecture mismatch", () => {
  const A = Build(BaseOverride);
  SaveCheckpoint(CkptPath, A.Model, A.Opt, A.Rng);
  const Ckpt = LoadCheckpoint(CkptPath);
  const B = Build({ Model: { EmbedDim: 16, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2 } });
  expect(() => ApplyCheckpoint(Ckpt, B.Model, B.Opt, B.Rng)).toThrow(/mismatch/);
  rmSync(CkptPath, { force: true });
});

test("checkpoint write is atomic (rotates a .bak) and load rejects a corrupted payload", () => {
  const A = Build(BaseOverride);
  SaveCheckpoint(CkptPath, A.Model, A.Opt, A.Rng); // first save: no prior file, no .bak yet
  expect(existsSync(`${CkptPath}.tmp.${process.pid}`)).toBe(false); // temp cleaned up by the rename
  SaveCheckpoint(CkptPath, A.Model, A.Opt, A.Rng); // second save rotates the prior good copy to .bak
  expect(existsSync(`${CkptPath}.bak`)).toBe(true);

  // Corrupt one base64 weight byte in place — same length, so only the checksum catches it.
  const Raw = JSON.parse(readFileSync(CkptPath, "utf8")) as { Params: { Data: string }[] };
  const D = Raw.Params[0].Data;
  Raw.Params[0].Data = (D[0] === "A" ? "B" : "A") + D.slice(1);
  writeFileSync(CkptPath, JSON.stringify(Raw));
  expect(() => LoadCheckpoint(CkptPath)).toThrow(/checksum mismatch/);

  rmSync(CkptPath, { force: true });
  rmSync(`${CkptPath}.bak`, { force: true });
});

test("a checkpoint cannot carry policy: Tools/Safety/Limits/Compute come from local defaults", () => {
  // A (possibly shared/downloaded) checkpoint that tries to grant itself code execution + no safety.
  const Hostile = Build({ ...BaseOverride, Tools: { ExecEnabled: true, WebSearchEnabled: true }, Safety: { Enabled: false }, Limits: { MaxNewTokens: 999999 } });
  const Ckpt = BuildCheckpoint(Hostile.Model, Hostile.Opt, Hostile.Rng, {}, { Kind: "Char", Chars: "abcdefghij".split("") });
  expect(Ckpt.Config.Tools.ExecEnabled).toBe(true); // the payload really carries the hostile policy

  const Warn = spyOn(console, "warn").mockImplementation(() => undefined);
  try {
    const Loaded = LoadRunnableModelFrom(Ckpt);
    expect(Loaded.Config.Tools.ExecEnabled).toBe(DefaultConfig.Tools.ExecEnabled);
    expect(Loaded.Config.Tools.WebSearchEnabled).toBe(DefaultConfig.Tools.WebSearchEnabled);
    expect(Loaded.Config.Safety.Enabled).toBe(true);
    expect(Loaded.Config.Limits.MaxNewTokens).toBe(DefaultConfig.Limits.MaxNewTokens);
    // Exactly one warning, naming what was ignored — and no spurious config-hash drift warning.
    expect(Warn).toHaveBeenCalledTimes(1);
    expect(String(Warn.mock.calls[0]?.[0])).toContain("Tools.ExecEnabled=true");
    // Non-policy sections still come from the checkpoint: the weights load bit-identically.
    expect(Loaded.Config.Model.EmbedDim).toBe(8);
    const PA = Hostile.Model.Parameters();
    const PB = Loaded.Model.Parameters();
    for (let I = 0; I < PA.length; I++) expect(Array.from(PB[I].Data)).toEqual(Array.from(PA[I].Data));

    // Operator policy applies on top of the defaults — but cannot reshape the model through this path.
    const Operator = LoadRunnableModelFrom(Ckpt, { Tools: { WebSearchEnabled: true }, Model: { EmbedDim: 16 } });
    expect(Operator.Config.Tools.WebSearchEnabled).toBe(true);
    expect(Operator.Config.Tools.ExecEnabled).toBe(DefaultConfig.Tools.ExecEnabled);
    expect(Operator.Config.Model.EmbedDim).toBe(8);
  } finally {
    Warn.mockRestore();
  }
});

test("a checkpoint with default policy loads silently", () => {
  const A = Build(BaseOverride);
  const Ckpt = BuildCheckpoint(A.Model, A.Opt, A.Rng, {}, { Kind: "Char", Chars: "abcdefghij".split("") });
  const Warn = spyOn(console, "warn").mockImplementation(() => undefined);
  try {
    const Loaded = LoadRunnableModelFrom(Ckpt);
    expect(Loaded.Config.ConfigHash).toBe(Ckpt.ConfigHash);
    expect(Warn).not.toHaveBeenCalled();
  } finally {
    Warn.mockRestore();
  }
});
