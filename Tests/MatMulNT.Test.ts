import { test, expect, afterEach } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { Tape } from "../Brain/Tensor/Tape.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import { Backward } from "../Brain/Autograd/Backward.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { MatMul, MatMulNT, Transpose } from "../Brain/Ops/OpsBarrel.ts";
import { SetActiveBackend } from "../Brain/ComputeBackend/BackendSelector.ts";
import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";

// M=11 crosses MatMulNT's 8-row block boundary, so both the full and the ragged block run.
const M = 11;
const K = 5;
const N = 7;

function RandTensor(Rows: number, Cols: number, Rng: SeededRng): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian();
  return T;
}

// Fixed random projection to a scalar so every output element gets a distinct, non-trivial gradient.
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

afterEach(() => SetActiveBackend(null));

test("MatMulNT forward computes A @ Bᵀ", () => {
  const A = new Tensor(2, 3);
  A.Data.set([1, 2, 3, 4, 5, 6]);
  const B = new Tensor(2, 3); // Bᵀ is [3,2]
  B.Data.set([1, 0, -1, 2, 1, 0]);
  const Out = MatMulNT(A, B);
  expect(Out.Rows).toBe(2);
  expect(Out.Cols).toBe(2);
  expect(Array.from(Out.Data)).toEqual([1 - 3, 2 + 2, 4 - 6, 8 + 5]);
});

test("MatMulNT rejects mismatched inner dims", () => {
  expect(() => MatMulNT(new Tensor(2, 3), new Tensor(4, 2))).toThrow();
});

test("MatMulNT gradient matches central finite differences", () => {
  const Rng = new SeededRng(21);
  const A = RandTensor(M, K, Rng);
  const B = RandTensor(N, K, Rng);
  const R = RandVec(M * N, Rng);
  const Result = GradCheck([A, B], () => ProjectScalar(MatMulNT(A, B), R));
  expect(Result.Passed).toBe(true);
  expect(Result.MaxAbsError).toBeLessThan(1e-7);
});

test("MatMulNT(X, X) (aliased operands) is gradient-correct", () => {
  const Rng = new SeededRng(22);
  const X = RandTensor(M, K, Rng);
  const R = RandVec(M * M, Rng);
  expect(GradCheck([X], () => ProjectScalar(MatMulNT(X, X), R)).Passed).toBe(true);
});

test("MatMulNT is bit-identical to MatMul(A, Transpose(B)) in F64, forward and backward", () => {
  const Rng = new SeededRng(23);
  const A = RandTensor(M, K, Rng);
  const B = RandTensor(N, K, Rng);
  const R = RandVec(M * N, Rng);

  const Fused = MatMulNT(A, B);
  const FusedOut = Float64Array.from(Fused.Data);
  Backward(ProjectScalar(Fused, R));
  const FusedDa = Float64Array.from(A.Grad);
  const FusedDb = Float64Array.from(B.Grad);

  A.ZeroGrad();
  B.ZeroGrad();
  const Reference = MatMul(A, Transpose(B));
  const ReferenceOut = Float64Array.from(Reference.Data);
  Backward(ProjectScalar(Reference, R));

  expect(Array.from(FusedOut)).toEqual(Array.from(ReferenceOut));
  expect(Array.from(FusedDa)).toEqual(Array.from(A.Grad));
  expect(Array.from(FusedDb)).toEqual(Array.from(B.Grad));
});

test("MatMulNT through a compute backend matches the inline path", () => {
  const Rng = new SeededRng(24);
  const A = RandTensor(M, K, Rng);
  const B = RandTensor(N, K, Rng);
  const R = RandVec(M * N, Rng);

  const Inline = MatMulNT(A, B);
  const InlineOut = Float64Array.from(Inline.Data);
  Backward(ProjectScalar(Inline, R));
  const InlineDa = Float64Array.from(A.Grad);
  const InlineDb = Float64Array.from(B.Grad);

  A.ZeroGrad();
  B.ZeroGrad();
  SetActiveBackend(new TsBackend());
  const Routed = MatMulNT(A, B);
  const RoutedOut = Float64Array.from(Routed.Data);
  Backward(ProjectScalar(Routed, R));
  SetActiveBackend(null);

  const ExpectClose = (Got: ArrayLike<number>, Want: ArrayLike<number>): void => {
    for (let I = 0; I < Want.length; I++) expect(Math.abs(Got[I] - Want[I])).toBeLessThan(1e-12);
  };
  ExpectClose(RoutedOut, InlineOut);
  ExpectClose(A.Grad, InlineDa);
  ExpectClose(B.Grad, InlineDb);
});
