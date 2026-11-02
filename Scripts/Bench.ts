// Performance gate for the owner: ms/token for cached vs uncached decode, and ms/step for one
// training forward+backward, on the Phase2Modern shape (Configs/Phase2Modern.Config.json).
// Run before and after a change and compare — a regression shows up as a larger number.
//
//   bun run Scripts/Bench.ts [--Tokens=3] [--Steps=2] [--Context=253] [--Backend=Ts|GoFfi] [--Precision=F64|F32]
//
// Context defaults to BlockSize-Tokens: the prompt plus the decoded tokens fill the window exactly,
// the worst case a full serving window pays per token (cached prefill is reported separately). Every
// measurement runs one untimed warm-up first so JIT compilation is not billed to the op.

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { WithTapeOff } from "../Brain/Tensor/Tape.ts";
import { KvCache } from "../Brain/Sampling/KvCache.ts";
import { CachedForwardStep, PrefillCache } from "../Brain/Sampling/CachedForward.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { ForwardBackward } from "../Brain/Training/TrainingStep.ts";
import { ReadArg } from "./ScriptArgs.ts";

const PresetPath = "Configs/Phase2Modern.Config.json";
const Greedy = { Temperature: 0, TopK: 0, TopP: 1 };
const TokenSeed = 7; // fixed so every run benchmarks the same id sequence

const Tokens = Number(ReadArg("--Tokens=", "3"));
const Steps = Number(ReadArg("--Steps=", "2"));
const Backend = ReadArg("--Backend=", "Ts") as "Ts" | "GoFfi";
const Precision = ReadArg("--Precision=", "F64") as "F64" | "F32";

// Dropout is forced to 0: it is not implemented (the schema rejects non-zero), and the bench measures
// the SHAPE of the preset, not its regularization knob.
const Overrides = { Model: { Dropout: 0 }, Compute: { Backend, Precision } };
const Config = LoadConfig({ ConfigPath: PresetPath, Overrides, UseCli: false, UseEnv: false });
const Choice = ActivateFromConfig(Config);
const Model = new Shahd(Config, CreateRngStreams(Config.Training.Seed).InitRng);
const { BlockSize, VocabSize } = Config.Model;
// Prefill + decoded tokens must fit the window, so the cached run never pays a window rebuild.
const Context = Math.min(Number(ReadArg("--Context=", String(BlockSize - Tokens))), BlockSize - Tokens);

const IdRng = new SeededRng(TokenSeed);
const RandomIds = (Count: number): number[] => Array.from({ length: Count }, () => Math.floor(IdRng.NextFloat() * VocabSize));
const Prompt = RandomIds(Context);

/** Mean wall-clock ms of Body over Repeats runs, after one untimed warm-up. */
function TimeMs(Repeats: number, Body: () => void): number {
  Body();
  const Start = performance.now();
  for (let I = 0; I < Repeats; I++) Body();
  return (performance.now() - Start) / Repeats;
}

function BenchUncachedDecode(): number {
  // Generate re-runs the whole context per token; time a fixed number of new tokens.
  return TimeMs(1, () => Generate(Model, Prompt, Tokens, Greedy, new SeededRng(1))) / Tokens;
}

function BenchCachedDecode(): { PrefillMs: number; DecodeMs: number } | string {
  try {
    const Run = (): { PrefillMs: number; DecodeMs: number } =>
      WithTapeOff(() => {
        const Cache = KvCache.ForModel(Model);
        const PrefillStart = performance.now();
        PrefillCache(Model, Cache, Prompt);
        const PrefillMs = performance.now() - PrefillStart;
        const DecodeStart = performance.now();
        for (let S = 0; S < Tokens; S++) CachedForwardStep(Model, Cache, S % VocabSize, Cache.Length);
        return { PrefillMs, DecodeMs: (performance.now() - DecodeStart) / Tokens };
      });
    Run(); // warm-up
    return Run();
  } catch (Error_) {
    return `unsupported (${(Error_ as Error).message.slice(0, 80)})`;
  }
}

function BenchTrainStep(): number {
  const Ids = RandomIds(BlockSize);
  const Targets = RandomIds(BlockSize);
  const Params = Model.Parameters();
  return TimeMs(Steps, () => {
    for (const P of Params) P.ZeroGrad();
    ForwardBackward(Model, Ids, Targets);
  });
}

console.log(`Bench: ${PresetPath} E=${Config.Model.EmbedDim} L=${Config.Model.NumLayers} H=${Config.Model.NumHeads} ` +
  `T=${BlockSize} V=${VocabSize} backend=${Choice.Chosen} context=${Context} tokens=${Tokens} steps=${Steps}`);
const Cached = BenchCachedDecode();
if (typeof Cached === "string") console.log(`cached decode   : ${Cached}`);
else console.log(`cached decode   : ${Cached.DecodeMs.toFixed(1)} ms/token (prefill ${Context} tokens: ${Cached.PrefillMs.toFixed(0)} ms)`);
console.log(`uncached decode : ${BenchUncachedDecode().toFixed(1)} ms/token`);
console.log(`train step      : ${BenchTrainStep().toFixed(1)} ms/step (T=${BlockSize}, forward+backward)`);
