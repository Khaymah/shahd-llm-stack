// LayerNorm per row: y = Gamma * (x - mu)/sqrt(var + Eps) + Beta, with Gamma/Beta as [1,N].
// Uses population variance (divide by N). Backward is the exact standard closed form:
//   dX_i = (invStd/N) * [N*dXHat_i - sum(dXHat) - xHat_i*sum(dXHat*xHat)]
// (the hardest op backward to get right — verified against nano-gpt.ts and by GradCheck).

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function LayerNorm(X: Tensor, Gamma: Tensor, Beta: Tensor, Eps: number): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  if (Gamma.Rows !== 1 || Gamma.Cols !== N || Beta.Rows !== 1 || Beta.Cols !== N) {
    throw new Error(`LayerNorm: Gamma/Beta must be [1,${N}]`);
  }
  const Out = new Tensor(M, N, undefined, [X, Gamma, Beta]);
  const XHat = new Float64Array(M * N);
  const InvStd = new Float64Array(M);
  const XData = X.Data;
  const GammaData = Gamma.Data;
  const BetaData = Beta.Data;
  const OutData = Out.Data;

  for (let I = 0; I < M; I++) {
    const Row = I * N;
    let Mu = 0;
    for (let J = 0; J < N; J++) Mu += XData[Row + J];
    Mu /= N;
    let Var = 0;
    for (let J = 0; J < N; J++) {
      const D = XData[Row + J] - Mu;
      Var += D * D;
    }
    Var /= N;
    const Is = 1 / Math.sqrt(Var + Eps);
    InvStd[I] = Is;
    for (let J = 0; J < N; J++) {
      const Xh = (XData[Row + J] - Mu) * Is;
      XHat[Row + J] = Xh;
      OutData[Row + J] = GammaData[J] * Xh + BetaData[J];
    }
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const GammaIn = Gamma.Data;
      const GammaGrad = Gamma.Grad;
      const BetaGrad = Beta.Grad;
      const XGrad = X.Grad;
      // One row-scratch per backward call (not per row): every row fully overwrites it before reading.
      const DXHat = new Float64Array(N);
      for (let I = 0; I < M; I++) {
        const Row = I * N;
        let SumDXHat = 0;
        let SumDXHatXHat = 0;
        for (let J = 0; J < N; J++) {
          const Dy = OutGrad[Row + J];
          const Dxh = Dy * GammaIn[J];
          DXHat[J] = Dxh;
          SumDXHat += Dxh;
          SumDXHatXHat += Dxh * XHat[Row + J];
          GammaGrad[J] += Dy * XHat[Row + J];
          BetaGrad[J] += Dy;
        }
        const Is = InvStd[I];
        for (let J = 0; J < N; J++) {
          XGrad[Row + J] += (Is / N) * (N * DXHat[J] - SumDXHat - XHat[Row + J] * SumDXHatXHat);
        }
      }
    };
  }
  return Out;
}
