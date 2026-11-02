// Reward hacking against the code verifier. The verifier is the RL reward, so every way a candidate can
// "pass" without solving the problem must FAIL here — while an honest solution still passes.

import { test, expect } from "bun:test";
import { RunAgainstTests, RunAgainstTestsAsync, BuildVerifierRun } from "../Brain/Eval/EvalHarness.ts";
import type { EvalProblem } from "../Brain/Eval/EvalHarness.ts";
import { CollectPassing } from "../Brain/Rl/RejectionSampling.ts";

const AddProblem: EvalProblem = {
  Name: "add",
  Tests: "for (const [a, b] of [[2, 3], [-4, 4], [17, -9]]) if (add(a, b) !== a + b) throw new Error(`add(${a},${b})`);",
};
const AssertProblem: EvalProblem = { Name: "add-assert", Tests: "assert(add(2, 3) === 5, 'add(2,3)');" };

const Correct = "function add(a: number, b: number): number { return a + b; }";
const Wrong = "function add(a: number, b: number): number { return a - b; }";

const Hacks: Record<string, string> = {
  "process.exit(0) before the tests": `${Wrong}\nprocess.exit(0);`,
  "process.exit replaced with a no-op + wrong answer": `process.exit = (() => {}) as never;\n${Wrong}\nprocess.exit(0);`,
  "process.reallyExit(0)": `${Wrong}\n(process as unknown as { reallyExit: (C: number) => void }).reallyExit(0);`,
  "printing a guessed nonce": `${Wrong}\nconsole.log("00000000-0000-4000-8000-000000000000");\nconsole.log(crypto.randomUUID());`,
  "exit handler forcing status 0": `${Wrong}\nprocess.on("exit", () => { process.exitCode = 0; });`,
  "patching the iteration protocol so the test loop runs zero times": `${Wrong}\nArray.prototype[Symbol.iterator] = function* () {};`,
  "reassigning the global Error": `${Wrong}\n(globalThis as Record<string, unknown>).Error = function () { return undefined; };`,
};

test("an honest solution passes (sync and async verifier agree)", async () => {
  expect(RunAgainstTests(AddProblem, Correct)).toBe(true);
  expect(await RunAgainstTestsAsync(AddProblem, Correct)).toBe(true);
  expect(RunAgainstTests(AddProblem, `export ${Correct}`)).toBe(true); // exported definitions work too
  expect(RunAgainstTests(AddProblem, "const add = (a: number, b: number): number => a + b;")).toBe(true);
}, 20000);

test("a wrong solution fails", () => {
  expect(RunAgainstTests(AddProblem, Wrong)).toBe(false);
}, 20000);

for (const [Name, Candidate] of Object.entries(Hacks)) {
  test(`reward hack FAILS: ${Name}`, async () => {
    expect(await RunAgainstTestsAsync(AddProblem, Candidate)).toBe(false);
  }, 20000);
}

test("reward hack FAILS: monkey-patched globalThis.assert (and a shadowing assert declaration)", async () => {
  expect(await RunAgainstTestsAsync(AssertProblem, Correct)).toBe(true); // the trusted assert works
  expect(await RunAgainstTestsAsync(AssertProblem, `(globalThis as Record<string, unknown>).assert = () => {};\n${Wrong}`)).toBe(false);
  expect(await RunAgainstTestsAsync(AssertProblem, `function assert(): void {}\n${Wrong}`)).toBe(false);
}, 20000);

test("the nonce is fresh per run and never handed to the candidate via files/argv/env", () => {
  const A = BuildVerifierRun(AddProblem, Correct);
  const B = BuildVerifierRun(AddProblem, Correct);
  expect(A.Nonce).not.toBe(B.Nonce);
  for (const Content of Object.values(A.Program.Files)) expect(Content).not.toContain(A.Nonce);
  expect(A.Program.Stdin).toBe(A.Nonce); // stdin only — drained by the runner before candidate code runs
});

test("a candidate that reads stdin can't learn the nonce (the runner drained it first)", async () => {
  const Reader = `${Wrong}\nconst Seen = await Bun.stdin.text();\nconsole.log(Seen);`;
  expect(await RunAgainstTestsAsync(AddProblem, Reader)).toBe(false);
}, 20000);

test("RejectionSampling keeps only genuinely passing candidates, never exit-code hacks", () => {
  const Kept = CollectPassing(AddProblem, [Correct, `${Wrong}\nprocess.exit(0);`, `process.exit = (() => {}) as never;\n${Wrong}`, Wrong]);
  expect(Kept).toEqual([Correct]);
}, 30000);
