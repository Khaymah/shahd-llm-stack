// Multiply a tensor by a scalar constant. Backward: dX += Factor * dOut.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Scale(X: Tensor, Factor: number): Tensor {
  const Size = X.Size;
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < Size; I++) OutData[I] = XData[I] * Factor;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const XGrad = X.Grad;
      const OutGrad = Out.Grad;
      for (let I = 0; I < Size; I++) XGrad[I] += Factor * OutGrad[I];
    };
  }
  return Out;
}
