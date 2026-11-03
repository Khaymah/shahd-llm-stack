// The training loop: per step -> scheduled LR, gradient-accumulation, global-norm clip,
// optimizer step, and periodic eval + structured logging.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { Logger } from "../Logging/Logger.ts";
import type { TrainWorkerPool } from "./WorkerPool.ts";
import { AccumulateGradients } from "./GradAccumulation.ts";
import { EvalLoss } from "./EvalLoop.ts";
import { ComputeLr } from "../Optim/OptimBarrel.ts";
import { ClipGradsChecked } from "./RunTrainingSteps.ts";

function Round(X: number): number {
  return Math.round(X * 1e4) / 1e4;
}

// Train steps [StartStep, EndStep) of a full run of Config.Schedule.MaxSteps. Defaults to the whole
// run; a caller can train in chunks (saving a checkpoint between them) by advancing the range while
// the model/optimizer/RNG carry over. The LR schedule always uses the GLOBAL step, so chunking is
// numerically identical to one continuous run.
// Returns the first step NOT executed: EndStep normally, or the current step when ShouldStop fired —
// so the caller can checkpoint at the EXACT pause point. ShouldStop must be synchronous (this loop
// blocks the event loop for its whole range); it is polled once per step, before the step runs.
export function TrainLoop(
  Model: Shahd,
  Optimizer: Optimizer,
  TrainLoader: DataLoader,
  ValLoader: DataLoader,
  Config: ResolvedConfig,
  RunLogger: Logger,
  OnStep?: (Step: number, TrainLoss: number, ElapsedMs: number) => void, // lightweight per-step hook (no eval)
  Range?: { StartStep: number; EndStep: number; StartMs: number },
  Pool?: TrainWorkerPool | null, // sequence-parallel accumulation (Config.Training.Workers); null/undefined = sequential
  ShouldStop?: () => boolean,
): number {
  const MaxSteps = Config.Schedule.MaxSteps;
  const StartStep = Range?.StartStep ?? 0;
  const EndStep = Range?.EndStep ?? MaxSteps;
  const StartMs = Range?.StartMs ?? Date.now();

  for (let Step = StartStep; Step < EndStep; Step++) {
    if (ShouldStop?.() === true) return Step;
    const Lr = ComputeLr(Step, Config);
    // The pooled path is a drop-in: same loader order, same 1/BatchSize grad scaling, same mean
    // loss — only WHERE each sequence's ForwardBackward runs changes (see WorkerPool.ts).
    const TrainLoss = Pool != null
      ? Pool.Accumulate(TrainLoader, Config.Training.BatchSize)
      : AccumulateGradients(Model, Optimizer, TrainLoader, Config.Training.BatchSize);
    // Clip + non-finite guard (shared with the SFT script): throws before Optimizer.Step on NaN/Inf.
    const GradNorm = ClipGradsChecked(Optimizer.Params, Config.Optimizer.GradClipNorm, Step);
    Optimizer.Step(Lr);
    OnStep?.(Step, TrainLoss, Date.now() - StartMs);

    const IsLast = Step === MaxSteps - 1;
    if (Step % Config.Training.EvalInterval === 0 || IsLast) {
      const Eval = EvalLoss(Model, ValLoader, Config.Training.EvalIterations);
      RunLogger.Log({
        Step,
        TrainLoss: Round(TrainLoss),
        ValLoss: Round(Eval.Loss),
        ValBpb: Round(Eval.BitsPerByte),
        Lr: Round(Lr),
        GradNorm: Round(GradNorm),
        ElapsedMs: Date.now() - StartMs,
        ConfigHash: Config.ConfigHash,
      });
    }
  }
  return EndStep;
}
