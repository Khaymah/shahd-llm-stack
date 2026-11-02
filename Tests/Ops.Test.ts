import { test, expect } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { Tape } from "../Brain/Tensor/Tape.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import {
  MatMul,
  Add,
  AddBias,
  Scale,
  ReLU,
  Transpose,
  CausalMask,
  SoftmaxRows,
  LayerNorm,
  EmbeddingLookup,
  CrossEntropy,
  MaskedCrossEntropy,
  MatMulNT,
} from "../Brain/Ops/OpsBarrel.ts";

function RandTensor(Rows: number, Cols: number, Rng: SeededRng, Scale = 1): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian() * Scale;
  return T;
}

function RandVec(N: number, Rng: SeededRng): Float64Array {
  const Out = new Float64Array(N);
  for (let I = 0; I < N; I++) Out[I] = Rng.NextGaussian();
  return Out;
}

// Reduce any tensor to a scalar via a FIXED random projection, giving non-trivial gradients for
// every element (constant R is captured, not gradient-checked).
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

test("MatMul gradient is correct", () => {
  const Rng = new SeededRng(1);
  const A = RandTensor(3, 4, Rng);
  const B = RandTensor(4, 2, Rng);
  const R = RandVec(3 * 2, Rng);
  expect(GradCheck([A, B], () => ProjectScalar(MatMul(A, B), R)).Passed).toBe(true);
});

test("Add gradient is correct", () => {
  const Rng = new SeededRng(2);
  const A = RandTensor(3, 4, Rng);
  const B = RandTensor(3, 4, Rng);
  const R = RandVec(12, Rng);
  expect(GradCheck([A, B], () => ProjectScalar(Add(A, B), R)).Passed).toBe(true);
});

test("AddBias gradient is correct", () => {
  const Rng = new SeededRng(3);
  const X = RandTensor(3, 4, Rng);
  const Bias = RandTensor(1, 4, Rng);
  const R = RandVec(12, Rng);
  expect(GradCheck([X, Bias], () => ProjectScalar(AddBias(X, Bias), R)).Passed).toBe(true);
});

test("Scale gradient is correct", () => {
  const Rng = new SeededRng(4);
  const X = RandTensor(3, 4, Rng);
  const R = RandVec(12, Rng);
  expect(GradCheck([X], () => ProjectScalar(Scale(X, 2.5), R)).Passed).toBe(true);
});

test("ReLU gradient is correct (inputs bounded away from 0)", () => {
  const Rng = new SeededRng(5);
  const X = new Tensor(3, 4);
  X.Data.set([0.5, -1.2, 0.3, -0.7, 0.9, -0.4, 1.1, -0.8, 0.6, -1.5, 0.2, -0.9]);
  const R = RandVec(12, Rng);
  expect(GradCheck([X], () => ProjectScalar(ReLU(X), R)).Passed).toBe(true);
});

test("Transpose gradient is correct", () => {
  const Rng = new SeededRng(6);
  const X = RandTensor(3, 4, Rng);
  const R = RandVec(12, Rng);
  expect(GradCheck([X], () => ProjectScalar(Transpose(X), R)).Passed).toBe(true);
});

test("CausalMask gradient is correct (composed with softmax, as used in attention)", () => {
  // CausalMask emits -1e9 on the future, which is designed to be consumed by softmax (-> ~0),
  // not summed linearly. Testing it in isolation would hit catastrophic cancellation of the
  // huge constants; composing with SoftmaxRows mirrors real attention usage.
  const Rng = new SeededRng(7);
  const X = RandTensor(4, 4, Rng);
  const R = RandVec(16, Rng);
  expect(GradCheck([X], () => ProjectScalar(SoftmaxRows(CausalMask(X)), R)).Passed).toBe(true);
});

test("SoftmaxRows gradient is correct", () => {
  const Rng = new SeededRng(8);
  const X = RandTensor(3, 5, Rng);
  const R = RandVec(15, Rng);
  expect(GradCheck([X], () => ProjectScalar(SoftmaxRows(X), R)).Passed).toBe(true);
});

test("LayerNorm gradient is correct (X, Gamma, Beta)", () => {
  const Rng = new SeededRng(9);
  const X = RandTensor(3, 6, Rng);
  const Gamma = new Tensor(1, 6);
  Gamma.Data.fill(1);
  for (let I = 0; I < 6; I++) Gamma.Data[I] += 0.1 * Rng.NextGaussian();
  const Beta = RandTensor(1, 6, Rng, 0.1);
  const R = RandVec(18, Rng);
  const Result = GradCheck([X, Gamma, Beta], () => ProjectScalar(LayerNorm(X, Gamma, Beta, 1e-5), R));
  expect(Result.Passed).toBe(true);
});

test("EmbeddingLookup gradient is correct (repeated ids accumulate)", () => {
  const Rng = new SeededRng(10);
  const Table = RandTensor(5, 3, Rng);
  const Ids = [0, 2, 2, 4, 1];
  const R = RandVec(Ids.length * 3, Rng);
  expect(GradCheck([Table], () => ProjectScalar(EmbeddingLookup(Table, Ids), R)).Passed).toBe(true);
});

test("CrossEntropy gradient is correct", () => {
  const Rng = new SeededRng(11);
  const Logits = RandTensor(4, 6, Rng);
  const Targets = [0, 3, 5, 1];
  expect(GradCheck([Logits], () => CrossEntropy(Logits, Targets)).Passed).toBe(true);
});

test("CrossEntropy loss is exact log-sum-exp (no 1e-12 floor) for a very unlikely target", () => {
  // Target logit 40 below its competitor: -log p = log(1 + e^40) ≈ 40. The old -log(p + 1e-12)
  // capped this at -log(1e-12) ≈ 27.63 while the gradient kept saying p ≈ 4e-18.
  const Logits = new Tensor(1, 2);
  Logits.Data.set([0, -40]);
  const Loss = CrossEntropy(Logits, [1]);
  expect(Math.abs(Loss.Data[0] - (40 + Math.log1p(Math.exp(-40))))).toBeLessThan(1e-12);
  const Masked = MaskedCrossEntropy(Logits, [1], [true]);
  expect(Masked.Data[0]).toBe(Loss.Data[0]);
});

test("MaskedCrossEntropy gradient is correct and ignores masked-out rows", () => {
  const Rng = new SeededRng(12);
  const Logits = RandTensor(4, 5, Rng);
  const Targets = [1, 0, 4, 2];
  const Mask = [true, false, true, false];
  expect(GradCheck([Logits], () => MaskedCrossEntropy(Logits, Targets, Mask)).Passed).toBe(true);
  // Masked-out targets are never validated or read — an out-of-range id there is harmless.
  expect(() => MaskedCrossEntropy(Logits, [1, 99, 4, 2], Mask)).not.toThrow();
});

test("MatMulNT gradient is correct", () => {
  const Rng = new SeededRng(13);
  const A = RandTensor(3, 4, Rng);
  const B = RandTensor(5, 4, Rng);
  const R = RandVec(15, Rng);
  expect(GradCheck([A, B], () => ProjectScalar(MatMulNT(A, B), R)).Passed).toBe(true);
});
