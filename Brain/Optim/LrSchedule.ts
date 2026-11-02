// Learning-rate schedules. Step is 0-indexed; the peak LR is Optimizer.LearningRate; the floor is
// MinLrRatio * peak. Every schedule is a pure function of Step (and the config), so a resumed run
// continues on exactly the curve it would have followed.
//   Cosine   — linear warmup, then cosine decay to the floor across [WarmupSteps, MaxSteps).
//   Constant — the peak throughout.
//   Wsd      — warmup-stable-decay: linear warmup, the peak held constant, then a LINEAR decay over the
//              last DecayFraction of MaxSteps that reaches the floor exactly on the last step
//              (MaxSteps - 1). Because the stable phase is flat, a run can be extended (raise MaxSteps
//              on --Resume before the decay window starts) without any LR discontinuity.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { WsdDecayStart } from "../Config/ValidateConfig.ts";

export function ComputeLr(Step: number, Config: ResolvedConfig): number {
  const { Kind, WarmupSteps, MaxSteps, MinLrRatio, DecayFraction } = Config.Schedule;
  const PeakLr = Config.Optimizer.LearningRate;
  const FloorLr = PeakLr * MinLrRatio;

  if (Kind === "Constant") return PeakLr;

  // Linear warmup.
  if (Step < WarmupSteps) return (PeakLr * (Step + 1)) / WarmupSteps;

  // Past the horizon: hold at the floor.
  if (Step >= MaxSteps) return FloorLr;

  if (Kind === "Wsd") {
    const DecayStart = Math.max(WarmupSteps, WsdDecayStart(MaxSteps, DecayFraction));
    if (Step < DecayStart) return PeakLr;
    const LastStep = MaxSteps - 1;
    // DecayStart === LastStep is a one-step window: that step is the floor.
    const Progress = LastStep > DecayStart ? (Step - DecayStart) / (LastStep - DecayStart) : 1;
    return PeakLr + (FloorLr - PeakLr) * Progress;
  }

  // Cosine decay from peak to floor across [WarmupSteps, MaxSteps).
  const Progress = (Step - WarmupSteps) / (MaxSteps - WarmupSteps);
  const Cosine = 0.5 * (1 + Math.cos(Math.PI * Progress));
  return PeakLr * (MinLrRatio + (1 - MinLrRatio) * Cosine);
}
