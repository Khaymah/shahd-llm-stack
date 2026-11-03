// Weight averaging for parallel training (Local SGD, the outer step of DiLoCo with outer lr 1 and no
// outer momentum): K workers start a round from the SAME checkpoint, each trains H steps on different
// batches, and the next round starts from the element-wise mean of their weights. Adam's moments are
// averaged too, so the inner optimizer continues smoothly instead of re-warming its statistics.
//
// Plain averaging is used on purpose: with only a handful of long rounds, an outer Nesterov step
// (lr 0.7, momentum 0.9) overshoots the mean by ~30% on the first round — a risk not worth taking
// before it is measured. All workers must share config, tokenizer and tensor shapes; a mismatch means
// the round was misconfigured, so it throws instead of averaging incompatible models.
// Memory: a 22M-parameter checkpoint is ~700 MB of JSON, so workers are folded in ONE AT A TIME
// (CheckpointAverager.Add) into running f64 sums; only the sums and the current worker are resident.

import type { Checkpoint } from "./CheckpointFormat.ts";
import { ChecksumPayload, DecodeFloat64, EncodeFloat64 } from "./CheckpointFormat.ts";

export type AverageReport = { Workers: number; WorkerSteps: number[]; Step: number };

type Sums = { Params: Float64Array[]; M: Float64Array[]; V: Float64Array[] };

function AddInto(Sum: Float64Array, Encoded: string, What: string, Worker: number): void {
  const Values = DecodeFloat64(Encoded);
  if (Values.length !== Sum.length) throw new Error(`AverageCheckpoints: ${What} has ${Values.length} values in worker ${Worker}, expected ${Sum.length}`);
  for (let I = 0; I < Sum.length; I++) Sum[I] += Values[I];
}

function MeanOf(Sum: Float64Array, Count: number): string {
  const Out = new Float64Array(Sum.length);
  for (let I = 0; I < Sum.length; I++) Out[I] = Sum[I] / Count;
  return EncodeFloat64(Out);
}

/** Folds worker checkpoints in one at a time; Finish() returns the averaged checkpoint. */
export class CheckpointAverager {
  private First: Checkpoint | null = null;
  private Tokenizer = "";
  private Sums: Sums | null = null;
  private Shapes: { Rows: number; Cols: number }[] = [];
  private readonly WorkerSteps: number[] = [];
  private StepCount = 0;

  Add(Ckpt: Checkpoint): void {
    const Index = this.WorkerSteps.length;
    if (this.First === null || this.Sums === null) {
      this.First = { ...Ckpt, Params: [], Optimizer: { M: [], V: [], StepCount: 0 } }; // keep metadata, drop the bulk
      this.Tokenizer = JSON.stringify(Ckpt.TokenizerState);
      this.Sums = {
        Params: Ckpt.Params.map((P) => new Float64Array(P.Rows * P.Cols)),
        M: Ckpt.Optimizer.M.map((M) => new Float64Array(DecodeFloat64(M).length)),
        V: Ckpt.Optimizer.V.map((V) => new Float64Array(DecodeFloat64(V).length)),
      };
      this.Shapes = Ckpt.Params.map((P) => ({ Rows: P.Rows, Cols: P.Cols }));
    }
    const First = this.First;
    if (Ckpt.ConfigHash !== First.ConfigHash) throw new Error(`AverageCheckpoints: worker ${Index} has config ${Ckpt.ConfigHash.slice(0, 12)}, worker 0 has ${First.ConfigHash.slice(0, 12)}`);
    if (JSON.stringify(Ckpt.TokenizerState) !== this.Tokenizer) throw new Error(`AverageCheckpoints: worker ${Index} was trained with a different tokenizer`);
    if (Ckpt.Params.length !== this.Shapes.length) throw new Error(`AverageCheckpoints: worker ${Index} has ${Ckpt.Params.length} tensors, expected ${this.Shapes.length}`);
    if (Ckpt.Optimizer.M.length !== this.Sums.M.length || Ckpt.Optimizer.V.length !== this.Sums.V.length) throw new Error(`AverageCheckpoints: worker ${Index} has a different optimizer state layout`);
    for (const [P, Tensor] of Ckpt.Params.entries()) {
      const Want = this.Shapes[P];
      if (Tensor.Rows !== Want.Rows || Tensor.Cols !== Want.Cols) throw new Error(`AverageCheckpoints: tensor ${P} is ${Tensor.Rows}x${Tensor.Cols} in worker ${Index}, expected ${Want.Rows}x${Want.Cols}`);
      AddInto(this.Sums.Params[P], Tensor.Data, `tensor ${P}`, Index);
    }
    for (const [I, M] of Ckpt.Optimizer.M.entries()) AddInto(this.Sums.M[I], M, `Adam m ${I}`, Index);
    for (const [I, V] of Ckpt.Optimizer.V.entries()) AddInto(this.Sums.V[I], V, `Adam v ${I}`, Index);
    this.StepCount = Math.max(this.StepCount, Ckpt.Optimizer.StepCount);
    this.WorkerSteps.push(Number(Ckpt.Meta["Step"] ?? 0));
  }

  /**
   * The averaged checkpoint. The step (and Adam's step count) is the furthest worker's, so the LR
   * schedule continues from the planned position even when a worker hit its time cap a little early;
   * the per-worker steps are recorded in Meta.ParallelRound.
   */
  Finish(ExtraMeta: Record<string, unknown> = {}): { Ckpt: Checkpoint; Report: AverageReport } {
    const Sums = this.Sums;
    if (this.First === null || Sums === null) throw new Error("AverageCheckpoints: no checkpoints given");
    const Count = this.WorkerSteps.length;
    const Params = this.Shapes.map((Shape, P) => ({ ...Shape, Data: MeanOf(Sums.Params[P], Count) }));
    const Optimizer = { M: Sums.M.map((Sum) => MeanOf(Sum, Count)), V: Sums.V.map((Sum) => MeanOf(Sum, Count)), StepCount: this.StepCount };
    const Step = Math.max(...this.WorkerSteps);
    const Report: AverageReport = { Workers: Count, WorkerSteps: [...this.WorkerSteps], Step };
    const Ckpt: Checkpoint = {
      ...this.First,
      Params,
      Optimizer,
      Meta: { ...this.First.Meta, ...ExtraMeta, Step, ParallelRound: Report },
      Checksum: ChecksumPayload(Params, Optimizer, this.First.Config),
    };
    return { Ckpt, Report };
  }
}

/** Convenience for checkpoints already in memory (tests, small models). */
export function AverageCheckpoints(Ckpts: readonly Checkpoint[], ExtraMeta: Record<string, unknown> = {}): { Ckpt: Checkpoint; Report: AverageReport } {
  const Averager = new CheckpointAverager();
  for (const Ckpt of Ckpts) Averager.Add(Ckpt);
  return Averager.Finish(ExtraMeta);
}
