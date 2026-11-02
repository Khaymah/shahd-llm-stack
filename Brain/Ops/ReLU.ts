// ReLU nonlinearity: max(0, x). Backward: gradient passes only where the input was positive.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function ReLU(X: Tensor): Tensor {
  const Size = X.Size;
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < Size; I++) OutData[I] = XData[I] > 0 ? XData[I] : 0;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const XIn = X.Data;
      const XGrad = X.Grad;
      const OutGrad = Out.Grad;
      for (let I = 0; I < Size; I++) {
        if (XIn[I] > 0) XGrad[I] += OutGrad[I];
      }
    };
  }
  return Out;
}
