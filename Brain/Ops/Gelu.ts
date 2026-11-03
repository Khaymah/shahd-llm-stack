// GELU activation (tanh approximation, as used by GPT-2/many code models). Backward is the
// analytic derivative of the approximation.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

const C = Math.sqrt(2 / Math.PI); // 0.79788456...
const A3 = 0.044715;

/** Scalar tanh-approximate GELU — the one forward formula (the KV-cached decode path reuses it). */
export function GeluValue(V: number): number {
  const Inner = C * (V + A3 * V * V * V);
  return 0.5 * V * (1 + Math.tanh(Inner));
}

export function Gelu(X: Tensor): Tensor {
  const Size = X.Size;
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < Size; I++) OutData[I] = GeluValue(XData[I]);

  if (Tape.On) {
    Out.BackwardFn = () => {
      const XIn = X.Data;
      const XGrad = X.Grad;
      const OutGrad = Out.Grad;
      for (let I = 0; I < Size; I++) {
        const V = XIn[I];
        const Inner = C * (V + A3 * V * V * V);
        const T = Math.tanh(Inner);
        const DInner = C * (1 + 3 * A3 * V * V);
        const D = 0.5 * (1 + T) + 0.5 * V * (1 - T * T) * DInner;
        XGrad[I] += D * OutGrad[I];
      }
    };
  }
  return Out;
}
