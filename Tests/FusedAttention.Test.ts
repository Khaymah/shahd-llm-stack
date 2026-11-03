import { test, expect } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { Tape } from "../Brain/Tensor/Tape.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import { Backward } from "../Brain/Autograd/Backward.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { CausalSoftmax, Scale, CausalMask, SoftmaxRows, MatMul, MatMulNT, Transpose } from "../Brain/Ops/OpsBarrel.ts";

const Tolerance = 1e-12;
const HeadDim = 4;
const Factor = 1 / Math.sqrt(HeadDim);

function RandTensor(Rows: number, Cols: number, Rng: SeededRng, Magnitude = 1): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian() * Magnitude;
  return T;
}

function ProjectScalar(X: Tensor, R: Float64Array): Tensor {
  const Out = new Tensor(1, 1, undefined, [X]);
  let Sum = 0;
  for (let I = 0; I < X.Size; I++) Sum += X.Data[I] * R[I];
  Out.Data[0] = Sum;
  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0];
      for (let I = 0; I < X.Size; I++) X.Grad[I] += G * R[I];
    };
  }
  return Out;
}

function RandVec(Count: number, Rng: SeededRng): Float64Array {
  return Float64Array.from({ length: Count }, () => Rng.NextGaussian());
}

const Unfused = (X: Tensor): Tensor => SoftmaxRows(CausalMask(Scale(X, Factor)));

/** Forward values + input gradient of `Op` under a fixed scalar projection. */
function Run(Op: (X: Tensor) => Tensor, X: Tensor, R: Float64Array): { Out: Float64Array; Dx: Float64Array } {
  X.ZeroGrad();
  const Y = Op(X);
  const Out = Float64Array.from(Y.Data);
  Backward(ProjectScalar(Y, R));
  return { Out, Dx: Float64Array.from(X.Grad) };
}

function ExpectClose(Got: Float64Array, Want: Float64Array): void {
  expect(Got.length).toBe(Want.length);
  for (let I = 0; I < Want.length; I++) expect(Math.abs(Got[I] - Want[I])).toBeLessThan(Tolerance);
}

for (const [T, Magnitude] of [[1, 1], [6, 1], [9, 30]] as const) {
  test(`CausalSoftmax == Scale→CausalMask→SoftmaxRows within 1e-12 (T=${T}, |scores|~${Magnitude})`, () => {
    const Rng = new SeededRng(100 + T);
    const X = RandTensor(T, T, Rng, Magnitude);
    const R = RandVec(T * T, Rng);
    const Fused = Run((In) => CausalSoftmax(In, Factor), X, R);
    const Reference = Run(Unfused, X, R);
    ExpectClose(Fused.Out, Reference.Out);
    ExpectClose(Fused.Dx, Reference.Dx);
    // Strict causality: the future is exactly zero and gets exactly zero gradient.
    for (let I = 0; I < T; I++) {
      for (let J = I + 1; J < T; J++) {
        expect(Fused.Out[I * T + J]).toBe(0);
        expect(Fused.Dx[I * T + J]).toBe(0);
      }
    }
  });
}

test("CausalSoftmax rows sum to 1 over the visible prefix", () => {
  const T = 7;
  const P = CausalSoftmax(RandTensor(T, T, new SeededRng(7)), Factor);
  for (let I = 0; I < T; I++) {
    let Sum = 0;
    for (let J = 0; J <= I; J++) Sum += P.Data[I * T + J];
    expect(Math.abs(Sum - 1)).toBeLessThan(1e-15);
  }
});

test("CausalSoftmax rejects a non-square score matrix", () => {
  expect(() => CausalSoftmax(new Tensor(3, 4), Factor)).toThrow();
});

test("CausalSoftmax gradient matches central finite differences", () => {
  const Rng = new SeededRng(8);
  const T = 6;
  const X = RandTensor(T, T, Rng);
  const R = RandVec(T * T, Rng);
  expect(GradCheck([X], () => ProjectScalar(CausalSoftmax(X, Factor), R)).Passed).toBe(true);
});

test("fused attention head (MatMulNT + CausalSoftmax) == unfused head (Transpose + MatMul + Scale + Mask + Softmax)", () => {
  const Rng = new SeededRng(9);
  const T = 8;
  const Q = RandTensor(T, HeadDim, Rng);
  const K = RandTensor(T, HeadDim, Rng);
  const V = RandTensor(T, HeadDim, Rng);
  const R = RandVec(T * HeadDim, Rng);
  const Leaves = [Q, K, V];

  const RunHead = (Head: () => Tensor): Float64Array[] => {
    for (const Leaf of Leaves) Leaf.ZeroGrad();
    const Out = Head();
    const Values = Float64Array.from(Out.Data);
    Backward(ProjectScalar(Out, R));
    return [Values, ...Leaves.map((Leaf) => Float64Array.from(Leaf.Grad))];
  };
  const Fused = RunHead(() => MatMul(CausalSoftmax(MatMulNT(Q, K), Factor), V));
  const Reference = RunHead(() => MatMul(Unfused(MatMul(Q, Transpose(K))), V));
  for (let I = 0; I < Fused.length; I++) ExpectClose(Fused[I], Reference[I]);
});
