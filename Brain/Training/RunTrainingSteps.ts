// Bare training-step loop (no eval, no checkpointing, no logging) — the minimal loop shared by the
// demo scripts. For a full run with eval + structured logging, use TrainLoop instead.
//
// Also the single home of the non-finite gradient guard (AssertFiniteGradNorm / ClipGradsChecked),
// shared by TrainLoop and the SFT script so every training path stops on a NaN/Infinity gradient
// BEFORE Optimizer.Step writes it into the weights (and into Adam's moments, where it never washes out).

import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { Tensor } from "../Tensor/Tensor.ts";
import { AccumulateGradients } from "./GradAccumulation.ts";
import { ClipGradGlobalNorm, ComputeLr } from "../Optim/OptimBarrel.ts";

/** Throw when the pre-clip global gradient norm is NaN or ±Infinity; returns the norm otherwise.
 *  Clipping cannot rescue such a step (Infinity scales to 0·Infinity = NaN; NaN never compares
 *  greater than the clip threshold), so the only safe move is to stop before Optimizer.Step. */
export function AssertFiniteGradNorm(GradNorm: number, Step: number): number {
  if (!Number.isFinite(GradNorm)) {
    throw new Error(
      `Non-finite gradient norm at step ${Step}: ${GradNorm} — stopping before the optimizer step ` +
        `corrupts the weights (lower the LR, check the data, or resume from the last checkpoint)`,
    );
  }
  return GradNorm;
}

/** Global-norm clip + finite guard in one call. Returns the pre-clip norm (for logging). */
export function ClipGradsChecked(Params: Tensor[], MaxNorm: number, Step: number): number {
  return AssertFiniteGradNorm(ClipGradGlobalNorm(Params, MaxNorm), Step);
}

export function RunTrainingSteps(Model: Shahd, Optimizer: Optimizer, Loader: DataLoader, Config: ResolvedConfig): void {
  for (let Step = 0; Step < Config.Schedule.MaxSteps; Step++) {
    AccumulateGradients(Model, Optimizer, Loader, Config.Training.BatchSize);
    ClipGradsChecked(Optimizer.Params, Config.Optimizer.GradClipNorm, Step);
    Optimizer.Step(ComputeLr(Step, Config));
  }
}
