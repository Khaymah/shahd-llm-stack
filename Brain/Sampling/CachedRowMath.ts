// Single-row (one token) numerics for the KV-cached decoder — the plain-array twins of the tensor
// ops a block runs (norm, projection, attention over the cache, MLP, LM head). Each mirrors its op's
// formula AND summation order, so the cached logits agree with the tensor forward to the last bit
// in F64 (CachedForward.Test.ts holds them to 1e-9). Weight buffers arrive in the run's storage
// precision (NumArray); row scratch and outputs stay f64.

import type { Shahd } from "../Nn/Shahd.ts";
import type { NormLayer } from "../Nn/NormLayer.ts";
import type { Mlp } from "../Nn/Mlp.ts";
import type { NumArray } from "../Tensor/Tensor.ts";
import { GeluValue, SiluValue } from "../Ops/OpsBarrel.ts";

/** X[K] @ W[K,N] -> [N] (the ikj order of Ops/MatMul, including its zero skip). */
export function MatMulRow(X: Float64Array, W: NumArray, K: number, N: number): Float64Array {
  const Out = new Float64Array(N);
  for (let Kk = 0; Kk < K; Kk++) {
    const Xk = X[Kk];
    if (Xk === 0) continue;
    const Base = Kk * N;
    for (let Nn = 0; Nn < N; Nn++) Out[Nn] += Xk * W[Base + Nn];
  }
  return Out;
}

function LayerNormRow(X: Float64Array, Gamma: NumArray, Beta: NumArray, Eps: number, N: number): Float64Array {
  let Mu = 0;
  for (let J = 0; J < N; J++) Mu += X[J];
  Mu /= N;
  let Var = 0;
  for (let J = 0; J < N; J++) {
    const D = X[J] - Mu;
    Var += D * D;
  }
  Var /= N;
  const Is = 1 / Math.sqrt(Var + Eps);
  const Out = new Float64Array(N);
  for (let J = 0; J < N; J++) Out[J] = Gamma[J] * ((X[J] - Mu) * Is) + Beta[J];
  return Out;
}

function RmsNormRow(X: Float64Array, Gamma: NumArray, Eps: number, N: number): Float64Array {
  let SumSq = 0;
  for (let J = 0; J < N; J++) SumSq += X[J] * X[J];
  const Is = 1 / Math.sqrt(SumSq / N + Eps);
  const Out = new Float64Array(N);
  for (let J = 0; J < N; J++) Out[J] = Gamma[J] * X[J] * Is;
  return Out;
}

/** The row form of NormLayer.Forward (dispatches on its Kind exactly as the layer does). */
export function NormRow(Layer: NormLayer, X: Float64Array): Float64Array {
  const N = X.length;
  if (Layer.Kind === "RmsNorm") return RmsNormRow(X, Layer.Gamma.Data, Layer.Eps, N);
  if (Layer.Beta === null) throw new Error("NormRow: LayerNorm missing Beta");
  return LayerNormRow(X, Layer.Gamma.Data, Layer.Beta.Data, Layer.Eps, N);
}

/** One query head attending over every cached position: softmax(Scale · q·k_t) weighted sum of v_t.
 *  The cache only ever holds the past and the current token, so no mask is needed. */
export function AttendRow(Q: Float64Array, Keys: Float64Array, Values: Float64Array, HeadDim: number, Scale: number): Float64Array {
  const T = Keys.length / HeadDim;
  const Scores = new Float64Array(T);
  let Max = -Infinity;
  for (let Ti = 0; Ti < T; Ti++) {
    let Dot = 0;
    const KBase = Ti * HeadDim;
    for (let D = 0; D < HeadDim; D++) Dot += Q[D] * Keys[KBase + D];
    const Scaled = Dot * Scale;
    Scores[Ti] = Scaled;
    if (Scaled > Max) Max = Scaled;
  }
  let Sum = 0;
  for (let Ti = 0; Ti < T; Ti++) {
    const Ex = Math.exp(Scores[Ti] - Max);
    Scores[Ti] = Ex;
    Sum += Ex;
  }
  const Ctx = new Float64Array(HeadDim);
  for (let Ti = 0; Ti < T; Ti++) {
    const Weight = Scores[Ti] / Sum;
    if (Weight === 0) continue; // mirrors MatMul's zero skip on the attention weights
    const VBase = Ti * HeadDim;
    for (let D = 0; D < HeadDim; D++) Ctx[D] += Weight * Values[VBase + D];
  }
  return Ctx;
}

function Required<T>(Value: T | null, What: string): T {
  if (Value === null) throw new Error(`MlpRow: ${What} weights missing`);
  return Value;
}

/** The row form of Mlp.Forward for every MlpKind (Relu with biases; SwiGlu/GeGlu gated). */
export function MlpRow(Layer: Mlp, X: Float64Array, Hidden: number): Float64Array {
  const E = X.length;
  if (Layer.Kind === "Relu") {
    const H1 = MatMulRow(X, Required(Layer.Wfc, "Relu Wfc").Data, E, Hidden);
    const Bfc = Required(Layer.Bfc, "Relu Bfc").Data;
    for (let J = 0; J < Hidden; J++) {
      const Val = H1[J] + Bfc[J];
      H1[J] = Val > 0 ? Val : 0; // ReLU
    }
    const H2 = MatMulRow(H1, Required(Layer.Wproj, "Relu Wproj").Data, Hidden, E);
    const Bproj = Required(Layer.Bproj, "Relu Bproj").Data;
    for (let J = 0; J < E; J++) H2[J] += Bproj[J];
    return H2;
  }
  // Gated: (Act(x @ W1) ⊙ (x @ W3)) @ W2, Act = SiLU (SwiGlu) or GELU (GeGlu).
  const Gate = MatMulRow(X, Required(Layer.W1, "gated W1").Data, E, Hidden);
  const Up = MatMulRow(X, Required(Layer.W3, "gated W3").Data, E, Hidden);
  const Act = Layer.Kind === "SwiGlu" ? SiluValue : GeluValue;
  for (let J = 0; J < Hidden; J++) Gate[J] = Act(Gate[J]) * Up[J];
  return MatMulRow(Gate, Required(Layer.W2, "gated W2").Data, Hidden, E);
}

/** Final norm + LM head for one hidden row -> logits [VocabSize] (tied: X @ Wteᵀ; untied: X @ LmHead). */
export function HeadLogitsRow(Model: Shahd, X: Float64Array): Float64Array {
  const E = X.length;
  const Xf = NormRow(Model.LnFinal, X);
  const VocabSize = Model.Config.Model.VocabSize;
  const Bias = Model.LmHeadBias.Data;
  if (!Model.WeightTying) {
    if (Model.LmHead === null) throw new Error("HeadLogitsRow: untied model missing LmHead");
    const Logits = MatMulRow(Xf, Model.LmHead.Data, E, VocabSize);
    for (let Vi = 0; Vi < VocabSize; Vi++) Logits[Vi] += Bias[Vi];
    return Logits;
  }
  const Wte = Model.Embedding.Wte.Data; // [VocabSize, E]: logits[v] = sum_j Xf[j] * Wte[v, j]
  const Logits = new Float64Array(VocabSize);
  for (let Vi = 0; Vi < VocabSize; Vi++) {
    let S = 0;
    const WBase = Vi * E;
    for (let J = 0; J < E; J++) S += Xf[J] * Wte[WBase + J];
    Logits[Vi] = S + Bias[Vi];
  }
  return Logits;
}
