// Elementwise (Hadamard) product of two same-shape tensors. Needed by gated MLPs (SwiGLU/GeGLU).
// Backward: dA += dOut ⊙ B ; dB += dOut ⊙ A.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Mul(A: Tensor, B: Tensor): Tensor {
  if (A.Rows !== B.Rows || A.Cols !== B.Cols) {
    throw new Error(`Mul: shape mismatch ${A.Rows}x${A.Cols} vs ${B.Rows}x${B.Cols}`);
  }
  const Size = A.Size;
  const Out = new Tensor(A.Rows, A.Cols, undefined, [A, B]);
  const AData = A.Data;
  const BData = B.Data;
  const OutData = Out.Data;
  for (let I = 0; I < Size; I++) OutData[I] = AData[I] * BData[I];

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const AIn = A.Data;
      const BIn = B.Data;
      const AGrad = A.Grad;
      const BGrad = B.Grad;
      for (let I = 0; I < Size; I++) {
        AGrad[I] += OutGrad[I] * BIn[I];
        BGrad[I] += OutGrad[I] * AIn[I];
      }
    };
  }
  return Out;
}
