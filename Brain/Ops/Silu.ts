// SiLU / Swish activation: x * sigmoid(x). The gate function in SwiGLU MLPs (Llama family).
// Backward: d/dx = sig*(1 + x*(1 - sig)), sig = sigmoid(x).

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

/** Scalar SiLU, x * sigmoid(x) — the one forward formula (the KV-cached decode path reuses it). */
export function SiluValue(V: number): number {
  return V * (1 / (1 + Math.exp(-V)));
}

export function Silu(X: Tensor): Tensor {
  const Size = X.Size;
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < Size; I++) OutData[I] = SiluValue(XData[I]);

  if (Tape.On) {
    Out.BackwardFn = () => {
      const XIn = X.Data;
      const XGrad = X.Grad;
      const OutGrad = Out.Grad;
      for (let I = 0; I < Size; I++) {
        const V = XIn[I];
        const Sig = 1 / (1 + Math.exp(-V));
        XGrad[I] += Sig * (1 + V * (1 - Sig)) * OutGrad[I];
      }
    };
  }
  return Out;
}
