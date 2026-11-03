// Code-eval harness (Phase 5). A problem is a set of tests; a candidate solution is run against them.
// Computes correct-count and the unbiased pass@k — the go/no-go metric for Phase 4/5. The verifier
// here is ALSO the reward signal for RL (Rl/RejectionSampling.ts), so it is built to resist reward
// hacking, not only to grade honest code.
//
// Why not "candidate + tests in one file, exit 0 = pass": candidate code runs FIRST, so a candidate of
// `process.exit(0)` (or one that patches whatever the tests rely on) "passes" every problem. Instead:
//   • the candidate is its own module (Candidate.ts); the tests live in a separate trusted runner
//     (Runner.ts) that imports it;
//   • the runner reads a per-run random nonce from stdin BEFORE any candidate code runs, and prints it
//     ONLY after every assertion passed. Pass = exit 0 AND stdout contains the nonce, so exiting early,
//     patching process.exit, `process.on("exit")` tricks and printing a guess all fail;
//   • process.exit/reallyExit/abort are replaced with throwing functions before the candidate loads;
//   • the tests resolve built-ins (and a trusted `assert`) from a snapshot taken BEFORE the candidate
//     loaded, and the prototypes they iterate/compare with are frozen, so patching globalThis.assert,
//     Array.prototype[Symbol.iterator] & co. can't turn a failing assertion into a pass.
// Residual (honest): the nonce lives in the runner's memory while the candidate runs, so a candidate
// that scans its own process memory (/proc/self/mem, bun:ffi) could still recover it. Closing that
// needs the tests to run out-of-process from the candidate, or the OS sandbox CodeExecutor describes.

import { randomUUID } from "node:crypto";
import { RunProgram, RunProgramAsync, DefaultExecTimeoutMs } from "./CodeExecutor.ts";
import type { ExecProgram, ExecResult } from "./CodeExecutor.ts";
import { PassAtK } from "./PassAtK.ts";

export type EvalProblem = {
  Name: string;
  Tests: string; // statements that reference the candidate's definitions and throw on failure (no imports)
};

export type ProblemEval = { Name: string; Correct: number; N: number; PassAt1: number };

// Standard built-ins the tests resolve from the pre-candidate snapshot (a candidate can reassign the
// global bindings, but the tests never see the reassignment). `assert` is the trusted helper.
const SnapshotNames: readonly string[] = [
  "Object", "Array", "String", "Number", "Boolean", "Symbol", "BigInt", "Math", "JSON", "Reflect", "Proxy",
  "Map", "Set", "WeakMap", "WeakSet", "Promise", "RegExp", "Date", "Error", "TypeError", "RangeError",
  "SyntaxError", "ReferenceError", "EvalError", "URIError", "AggregateError", "parseInt", "parseFloat",
  "isNaN", "isFinite", "NaN", "Infinity", "undefined", "encodeURIComponent", "decodeURIComponent",
  "encodeURI", "decodeURI", "ArrayBuffer", "DataView", "Uint8Array", "Int8Array", "Uint16Array",
  "Int16Array", "Uint32Array", "Int32Array", "Float32Array", "Float64Array", "BigInt64Array",
  "BigUint64Array", "structuredClone", "console",
];

// Appended to the candidate module: a resolver that reads the module's OWN bindings by name (direct
// eval in module scope), so plain `function add(...)` needs no export for the tests to call it. If the
// candidate's source swallows this footer (unterminated comment/string) the module fails to load or
// lacks the resolver — a fail, never a pass.
const CandidateFooter = "\n;export const __ShahdResolve = (Name: string): unknown => eval(Name);\n";

// The trusted runner. Order matters: everything it needs after the candidate loads (nonce, real exit,
// writer, snapshot, compiled tests) is captured BEFORE the dynamic import of the candidate.
function RunnerSource(Tests: string): string {
  return `import { writeSync } from "node:fs";
const Nonce = (await Bun.stdin.text()).trim();
// reallyExit (what process.exit calls after the "exit" handlers) — so candidate-registered exit
// handlers never run after the verdict, and our replacement below can't intercept our own exit.
const RealExit = (typeof process.reallyExit === "function" ? process.reallyExit : process.exit).bind(process);
const SafeError = Error;
const Fail = (Why) => {
  try { writeSync(2, "verifier: " + Why + "\\n"); } finally { RealExit(1); }
};
const Blocked = () => { throw new SafeError("process.exit is disabled inside the verifier"); };
process.exit = Blocked;
process.reallyExit = Blocked;
process.abort = Blocked;

const Snapshot = Object.create(null);
for (const Name of ${JSON.stringify(SnapshotNames)}) Snapshot[Name] = globalThis[Name];
Snapshot.assert = function assert(Condition, Message) {
  if (!Condition) throw new SafeError(Message === undefined ? "assertion failed" : "" + Message);
};

const ArrayIterator = Object.getPrototypeOf([][Symbol.iterator]());
for (const Target of [
  Array.prototype, ArrayIterator, Object.getPrototypeOf(ArrayIterator), String.prototype,
  Object.getPrototypeOf(""[Symbol.iterator]()), Map.prototype, Object.getPrototypeOf(new Map()[Symbol.iterator]()),
  Set.prototype, Object.getPrototypeOf(new Set()[Symbol.iterator]()), Function.prototype, Number.prototype,
  Boolean.prototype, Symbol.prototype, RegExp.prototype, Promise.prototype,
  Object, Array, String, Number, Boolean, Symbol, Math, JSON, Reflect, Map, Set, Promise, RegExp, Function,
]) Object.freeze(Target);

let RunTests;
try {
  const TestsJs = new Bun.Transpiler({ loader: "ts" }).transformSync(${JSON.stringify(Tests)});
  RunTests = new Function("__ShahdScope", "with (__ShahdScope) {\\n" + TestsJs + "\\n}");
} catch {
  Fail("tests failed to compile");
}

let Resolve;
try {
  Resolve = (await import("./Candidate.ts")).__ShahdResolve;
} catch {
  Fail("candidate failed to load");
}
if (typeof Resolve !== "function") Fail("candidate module did not expose its bindings");

const Scope = new Proxy(Object.create(null), {
  has(_Target, Name) {
    if (typeof Name !== "string") return false;
    if (Name in Snapshot) return true;
    try { Resolve(Name); return true; } catch { return false; }
  },
  get(_Target, Name) {
    if (typeof Name !== "string") return undefined;
    if (Name in Snapshot) return Snapshot[Name];
    return Resolve(Name);
  },
});

let Passed = false;
try {
  RunTests(Scope);
  Passed = true;
} catch {
  Passed = false;
}
if (!Passed) Fail("an assertion failed or the candidate threw");
writeSync(1, "\\n" + Nonce + "\\n");
RealExit(0);
`;
}

/** The program + nonce for ONE verification run. Single home of the verifier layout, shared by the sync
 *  and async runners below (and through them by EvaluateProblem and RejectionSampling). */
export function BuildVerifierRun(Problem: EvalProblem, Candidate: string): { Program: ExecProgram; Nonce: string } {
  const Nonce = randomUUID();
  return {
    Nonce,
    Program: {
      Files: { "Candidate.ts": Candidate + CandidateFooter, "Runner.ts": RunnerSource(Problem.Tests) },
      Entry: "Runner.ts",
      Stdin: Nonce,
    },
  };
}

// The ONLY pass criterion: a clean exit within the caps AND the runner's post-assertion nonce line.
function Verdict(Result: ExecResult, Nonce: string): boolean {
  return Result.Passed && Result.Stdout.split("\n").includes(Nonce);
}

/** Execute ONE candidate against a problem's tests (synchronous). Pass requires the per-run nonce —
 *  exit status alone never counts. */
export function RunAgainstTests(Problem: EvalProblem, Candidate: string, TimeoutMs = DefaultExecTimeoutMs): boolean {
  const Run = BuildVerifierRun(Problem, Candidate);
  return Verdict(RunProgram(Run.Program, TimeoutMs), Run.Nonce);
}

/** Async (non-blocking) sibling of RunAgainstTests — same verifier, same verdict. */
export async function RunAgainstTestsAsync(Problem: EvalProblem, Candidate: string, TimeoutMs = DefaultExecTimeoutMs): Promise<boolean> {
  const Run = BuildVerifierRun(Problem, Candidate);
  return Verdict(await RunProgramAsync(Run.Program, TimeoutMs), Run.Nonce);
}

/** Run every candidate against the problem's tests; count how many pass. */
export function EvaluateProblem(Problem: EvalProblem, Candidates: string[], TimeoutMs = DefaultExecTimeoutMs): ProblemEval {
  let Correct = 0;
  for (const Candidate of Candidates) {
    if (RunAgainstTests(Problem, Candidate, TimeoutMs)) Correct++;
  }
  return {
    Name: Problem.Name,
    Correct,
    N: Candidates.length,
    PassAt1: Candidates.length > 0 ? PassAtK(Candidates.length, Correct, 1) : 0,
  };
}
