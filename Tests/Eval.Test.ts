import { test, expect } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { PassAtK } from "../Brain/Eval/PassAtK.ts";
import { RunCode, RunCodeAsync, RunProgramAsync } from "../Brain/Eval/CodeExecutor.ts";
import { EvaluateProblem, RunAgainstTests } from "../Brain/Eval/EvalHarness.ts";
import { CollectPassing } from "../Brain/Rl/RejectionSampling.ts";
import { CodingProblems } from "../Brain/Eval/ProblemSet.ts";

test("PassAtK matches known values", () => {
  expect(PassAtK(5, 0, 1)).toBe(0);
  expect(PassAtK(5, 5, 1)).toBe(1);
  expect(PassAtK(2, 1, 1)).toBeCloseTo(0.5, 10);
  expect(PassAtK(4, 2, 2)).toBeCloseTo(1 - 1 / 6, 10); // 1 - C(2,2)/C(4,2)
});

test("CodeExecutor runs passing/failing code and enforces a timeout", () => {
  expect(RunCode("if (1 + 1 !== 2) throw new Error('bad');").Passed).toBe(true);
  expect(RunCode("throw new Error('boom');").Passed).toBe(false);
  const Timed = RunCode("while (true) {}", 800);
  expect(Timed.Passed).toBe(false); // killed by the timeout
}, 20000);

test("EvaluateProblem and CollectPassing count correct candidates", () => {
  const Problem = { Name: "add", Tests: "if (add(2, 3) !== 5) throw new Error('fail');" };
  const Good = "function add(a, b) { return a + b; }";
  const Bad = "function add(a, b) { return a - b; }";
  const Result = EvaluateProblem(Problem, [Good, Bad, Good]);
  expect(Result.Correct).toBe(2);
  expect(Result.PassAt1).toBeCloseTo(PassAtK(3, 2, 1), 10);
  expect(CollectPassing(Problem, [Good, Bad]).length).toBe(1);
  expect(CollectPassing(Problem, [Good, Good, Good]).length).toBe(1); // distinct passers only (dedup)
}, 20000);

test("every ProblemSet reference solution passes its own tests through the nonce verifier", () => {
  for (const P of CodingProblems) {
    expect(RunAgainstTests(P, P.Reference)).toBe(true);
  }
}, 60000);

test("RunCodeAsync: private temp cwd (not the repo), scrubbed env, no .env auto-load", async () => {
  const Result = await RunCodeAsync("console.log(JSON.stringify({ Cwd: process.cwd(), Env: Object.keys(process.env) }));");
  expect(Result.Passed).toBe(true);
  const Seen = JSON.parse(Result.Stdout.trim()) as { Cwd: string; Env: string[] };
  expect(Seen.Cwd).not.toBe(process.cwd());
  expect(Seen.Cwd).toContain("shahd-exec-");
  for (const Name of Seen.Env) expect(["PATH", "HOME", "TMPDIR", "TEMP", "TMP"]).toContain(Name);
}, 20000);

// A killed grandchild whose parent also died may linger as a ZOMBIE (in a container whose pid 1 does not
// reap), and kill(pid, 0) succeeds on zombies — so on Linux read its state instead of probing.
function IsRunning(Pid: number): boolean {
  if (existsSync("/proc/self/stat")) {
    let Stat: string;
    try {
      Stat = readFileSync(`/proc/${Pid}/stat`, "utf8");
    } catch {
      return false; // gone (or reaped between checks)
    }
    const State = Stat.split(") ")[1]?.[0];
    return State !== "Z" && State !== "X";
  }
  try {
    process.kill(Pid, 0);
    return true;
  } catch {
    return false;
  }
}

test("RunCodeAsync: timeout kills the whole process group (no orphaned grandchild)", async () => {
  // The candidate spawns a background child that would outlive it, then spins. After the timeout the
  // group kill must have taken the grandchild down too: its pid no longer exists.
  const Code = `const Child = Bun.spawn(["sleep", "30"]); console.log(Child.pid); while (true) {}`;
  const Result = await RunCodeAsync(Code, 1500);
  expect(Result.Passed).toBe(false);
  expect(Result.TimedOut).toBe(true);
  const Pid = Number(Result.Stdout.trim());
  expect(Number.isInteger(Pid) && Pid > 0).toBe(true);
  expect(IsRunning(Pid)).toBe(false);
}, 20000);

test("RunCodeAsync caps output and the timeout ceiling; unsafe file names are rejected", async () => {
  const Flood = await RunCodeAsync(`const Line = "x".repeat(4096); while (true) console.log(Line);`, 8000);
  expect(Flood.Passed).toBe(false);
  expect(Flood.OutputCapped).toBe(true);
  const Clamped = await RunCodeAsync("while (true) {}", Number.POSITIVE_INFINITY); // non-finite -> default
  expect(Clamped.TimedOut).toBe(true);
  await expect(RunProgramAsync({ Files: { "../Escape.ts": "1" }, Entry: "../Escape.ts" })).rejects.toThrow("unsafe file name");
}, 30000);
