// Flat-buffer plumbing shared by MatMul and MatMulNT on their compute-backend paths: a transposed
// copy (for backends without the native NT/TN halves) and the grad accumulation of a buffer a
// backend returned. Internal only — not re-exported from OpsBarrel.ts. No gradient math lives here.

import type { NumArray, NumArrayCtor } from "../../Tensor/Tensor.ts";

/** Row-major Data[Rows,Cols] -> a new [Cols,Rows] buffer of the same element type. */
export function TransposeBuffer(Data: NumArray, Rows: number, Cols: number): NumArray {
  const Out = new (Data.constructor as NumArrayCtor)(Rows * Cols);
  for (let R = 0; R < Rows; R++) {
    for (let C = 0; C < Cols; C++) Out[C * Rows + R] = Data[R * Cols + C];
  }
  return Out;
}

/** Target[0..Count) += Source[0..Count) — adds a backend-returned buffer into a grad. */
export function AccumulateInto(Target: NumArray, Source: NumArray, Count: number): void {
  for (let I = 0; I < Count; I++) Target[I] += Source[I];
}
