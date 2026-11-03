// Rotary Position Embedding (RoPE). Rotates adjacent dim pairs (2i, 2i+1) of X[T,D] by an angle
// proportional to the token's absolute position and the pair's frequency. Applied to Q and K
// inside attention (D = HeadDim, even). RoPE encodes RELATIVE position in the dot product, so
// unlike learned absolute positions it extends past the training context — which also removes
// the KV-cache-beyond-BlockSize limitation (KvCache.ts note). The rotation is orthogonal, so its
// backward is the rotation by the negative angle.

import { Tensor } from "../Tensor/Tensor.ts";
import type { NumArray } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

/** The default RoPE frequency base (theta) — the value every attention call site uses. */
export const DefaultRopeBase = 10000;

/** Per-pair inverse frequencies Base^(-2i/D), i in [0, D/2). Shared by ApplyRope and the KV-cached
 *  decode path so both rotate by bit-identical angles. */
export function RopeFrequencies(D: number, Base: number = DefaultRopeBase): Float64Array {
  const Half = D / 2;
  const Freq = new Float64Array(Half);
  for (let I = 0; I < Half; I++) Freq[I] = Math.pow(Base, (-2 * I) / D);
  return Freq;
}

/** cos/sin of Position*Freq[i] for Rows consecutive positions from PositionStart, laid out [Row, Pair]. */
export function RopeCosSin(Freq: Float64Array, PositionStart: number, Rows: number): { Cos: Float64Array; Sin: Float64Array } {
  const Half = Freq.length;
  const Cos = new Float64Array(Rows * Half);
  const Sin = new Float64Array(Rows * Half);
  for (let R = 0; R < Rows; R++) {
    const Position = PositionStart + R;
    for (let I = 0; I < Half; I++) {
      const Angle = Position * Freq[I];
      Cos[R * Half + I] = Math.cos(Angle);
      Sin[R * Half + I] = Math.sin(Angle);
    }
  }
  return { Cos, Sin };
}

/** Rotate one row: each pair (x0, x1) = (Src[2i], Src[2i+1]) becomes (x0·c − x1·s, x0·s + x1·c).
 *  Src and Dst may be the same buffer and offset (each pair is read before it is written). */
export function RotateRow(
  Src: NumArray,
  SrcOffset: number,
  Dst: NumArray,
  DstOffset: number,
  Cos: Float64Array,
  Sin: Float64Array,
  CsOffset: number,
  Half: number,
): void {
  for (let I = 0; I < Half; I++) {
    const C = Cos[CsOffset + I];
    const S = Sin[CsOffset + I];
    const X0 = Src[SrcOffset + 2 * I];
    const X1 = Src[SrcOffset + 2 * I + 1];
    Dst[DstOffset + 2 * I] = X0 * C - X1 * S;
    Dst[DstOffset + 2 * I + 1] = X0 * S + X1 * C;
  }
}

/** Rotate X[T,D] (D even). Row r uses absolute position PositionStart+r. */
export function ApplyRope(X: Tensor, PositionStart: number, Base: number = DefaultRopeBase): Tensor {
  const T = X.Rows;
  const D = X.Cols;
  if (D % 2 !== 0) throw new Error(`ApplyRope: head dim ${D} must be even`);
  const Half = D / 2;
  const Out = new Tensor(T, D, undefined, [X]);

  // cos/sin per (row, pair), computed once and shared by forward and backward.
  const { Cos, Sin } = RopeCosSin(RopeFrequencies(D, Base), PositionStart, T);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let R = 0; R < T; R++) RotateRow(XData, R * D, OutData, R * D, Cos, Sin, R * Half, Half);

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const XGrad = X.Grad;
      for (let R = 0; R < T; R++) {
        for (let I = 0; I < Half; I++) {
          const C = Cos[R * Half + I];
          const S = Sin[R * Half + I];
          const G0 = OutGrad[R * D + 2 * I];
          const G1 = OutGrad[R * D + 2 * I + 1];
          // Backward = rotation by -angle applied to the upstream gradient.
          XGrad[R * D + 2 * I] += G0 * C + G1 * S;
          XGrad[R * D + 2 * I + 1] += -G0 * S + G1 * C;
        }
      }
    };
  }
  return Out;
}
