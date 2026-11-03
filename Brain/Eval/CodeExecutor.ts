// Resource-capped code RUNNER for eval, RL verifiable reward (Phase 5) and the opt-in `run_code` tool.
//
// ⚠️ THIS IS NOT A SANDBOX. Read this before pointing it at untrusted input.
// What it DOES enforce, in one controllable place:
//   • a fresh subprocess per run, in its own process group (detached), whose WHOLE group is SIGKILLed
//     on timeout, on output overflow, and after the main process exits (so a background grandchild
//     it spawned does not outlive the run);
//   • a wall-clock timeout, clamped to HardMaxTimeoutMs no matter what the caller/model asks for;
//   • a cap on captured stdout+stderr (MaxOutputBytes), so a print loop can't grow host memory;
//   • a fresh PRIVATE temp directory (mode 0700) as the working directory — never the repo root where
//     `.env` lives — removed after the run, plus `bun run --no-env-file` so no .env is auto-loaded;
//   • an ALLOWLISTED environment (secrets dropped by default, not pattern-matched out).
// What it does NOT do: restrict the filesystem (the child can read/write anything this user can,
// including the repo and `~`), restrict the network, cap memory/CPU/pids, or hide other processes:
// `/proc/<ancestor pid>/environ` of the HOST process (with its unscrubbed secrets) is readable by a
// same-uid child, and a child that calls setsid()/spawns detached escapes the group kill. The real
// isolation boundary — a container/VM/gVisor with no network, a read-only fs, cgroup limits and a
// separate uid — is out of scope here; this module is the seam where it is plugged in. Do NOT enable
// Tools.ExecEnabled in a deployment reachable by untrusted input without that OS-level isolation.
//
// Two entry styles share ONE hardened spawn path (SpawnCapped):
//   RunProgramAsync / RunCodeAsync — async, non-blocking; the preferred API.
//   RunProgram / RunCode            — synchronous, for callers that need a sync result (the regex and
//     run_code tools, the sync eval/RL verifiers). It re-runs THIS file as a tiny supervisor process
//     (spawnSync + SIGKILL backstop) which performs the async path, so sync callers get the same
//     process-group kill, private cwd and caps instead of a weaker spawnSync-only copy.

import { spawn, spawnSync } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

export type ExecResult = {
  Passed: boolean; // exit code 0, not timed out, output not capped, spawned successfully
  ExitCode: number | null;
  Stdout: string;
  Stderr: string;
  DurationMs: number;
  TimedOut: boolean;
  OutputCapped: boolean;
};

// A multi-file program: every file is written into the run's private directory, Entry is executed and
// Stdin (if any) is piped to it. File names are bare names (no paths) — validated before anything runs.
export type ExecProgram = { Files: Readonly<Record<string, string>>; Entry: string; Stdin?: string };

export const DefaultExecTimeoutMs = 5000;

// A hard ceiling on the (possibly model-supplied) timeout: a caller/model can shorten it but never make
// one execution hold the host (or, on the sync path, block the event loop) for an unbounded window.
const HardMaxTimeoutMs = 10000;

// A hard ceiling on captured stdout+stderr: the timeout bounds wall-clock only — a candidate that prints
// in a tight loop could otherwise grow host memory without limit inside that window. On overflow the
// group is killed and the run fails.
const MaxOutputBytes = 1024 * 1024; // 1 MB

// After the main process exits (or is killed), how long to wait for its pipes to close. A grandchild
// that escaped the process group may hold them open forever; we stop waiting instead of hanging.
const PipeCloseGraceMs = 250;

// Sync path: the supervisor gets the run's own timeout plus this slack before spawnSync SIGKILLs IT
// (a backstop only — the supervisor's internal timer always fires first).
const SupervisorSlackMs = 5000;

// Sync path: the supervisor returns the ExecResult as JSON. JSON escaping can grow a captured byte to at
// most 6 chars (\u00XX), for stdout + stderr, plus a little envelope.
const SupervisorEnvelopeBytes = 64 * 1024;
const SupervisorMaxBuffer = MaxOutputBytes * 2 * 6 + SupervisorEnvelopeBytes;
const SupervisorFlag = "--shahd-exec-supervisor";

// Bare file names only: no separators, no leading dot (no "..", no hidden files).
const SafeFileName = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;

// Pass through ONLY the handful of vars the bun runtime needs to start (PATH resolution, temp dir,
// Windows shell plumbing) — everything else, secrets included, is dropped by default.
const AllowedEnvVars = new Set([
  "PATH", "Path", "SystemRoot", "windir", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
  "ProgramFiles", "ProgramData", "PATHEXT", "COMSPEC",
]);

function ScrubbedEnv(): Record<string, string> {
  const Out: Record<string, string> = {};
  for (const [Name, Value] of Object.entries(process.env)) {
    if (Value !== undefined && AllowedEnvVars.has(Name)) Out[Name] = Value;
  }
  return Out;
}

function ClampTimeout(TimeoutMs: number): number {
  if (!Number.isFinite(TimeoutMs)) return DefaultExecTimeoutMs;
  return Math.min(Math.max(1, Math.floor(TimeoutMs)), HardMaxTimeoutMs);
}

function ValidateProgram(Program: ExecProgram): void {
  const Names = Object.keys(Program.Files);
  for (const Name of Names) {
    if (!SafeFileName.test(Name)) throw new Error(`CodeExecutor: unsafe file name ${JSON.stringify(Name)}`);
  }
  if (!Names.includes(Program.Entry)) throw new Error(`CodeExecutor: entry ${JSON.stringify(Program.Entry)} is not one of the program files`);
}

// The command that runs one file: the SAME bun binary as the host (not a PATH lookup a planted `bun`
// could shadow), with automatic .env loading disabled.
function BunArgs(File: string): string[] {
  return ["run", "--no-env-file", File];
}

// SIGKILL the child's whole process group (it was spawned detached, so its pid is the group id).
// ESRCH (group already gone) is expected and ignored. Windows has no process groups: kill the child.
function KillGroup(Child: ChildProcessWithoutNullStreams): void {
  const Pid = Child.pid;
  if (Pid === undefined) return;
  try {
    if (process.platform === "win32") Child.kill("SIGKILL");
    else process.kill(-Pid, "SIGKILL");
  } catch {
    // already exited — nothing left to kill
  }
}

function FailedResult(Stderr: string, Start: number, ExitCode: number | null = null, TimedOut = false): ExecResult {
  return { Passed: false, ExitCode, Stdout: "", Stderr, DurationMs: Date.now() - Start, TimedOut, OutputCapped: false };
}

// The one hardened spawn: detached group, private cwd, scrubbed env, capped output, group-kill timer.
function SpawnCapped(EntryPath: string, Cwd: string, Stdin: string, Timeout: number): Promise<ExecResult> {
  const Start = Date.now();
  return new Promise<ExecResult>((Resolve) => {
    let Child: ChildProcessWithoutNullStreams;
    try {
      Child = spawn(process.execPath, BunArgs(EntryPath), {
        cwd: Cwd,
        env: ScrubbedEnv(),
        detached: true,
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (Err) {
      Resolve(FailedResult(`spawn failed: ${(Err as Error).message}`, Start));
      return;
    }
    const Out: Buffer[] = [];
    const ErrOut: Buffer[] = [];
    let Captured = 0;
    let TimedOut = false;
    let OutputCapped = false;
    let ExitCode: number | null = null;
    let SpawnError: string | null = null;
    let Settled = false;
    let GraceTimer: ReturnType<typeof setTimeout> | null = null;
    let DeadlineTimer: ReturnType<typeof setTimeout> | null = null;

    const Finish = (): void => {
      if (Settled) return;
      Settled = true;
      if (DeadlineTimer !== null) clearTimeout(DeadlineTimer);
      if (GraceTimer !== null) clearTimeout(GraceTimer);
      Child.stdout.destroy();
      Child.stderr.destroy();
      Child.stdin.destroy();
      const Stderr = Buffer.concat(ErrOut).toString("utf8");
      Resolve({
        Passed: ExitCode === 0 && !TimedOut && !OutputCapped && SpawnError === null,
        ExitCode,
        Stdout: Buffer.concat(Out).toString("utf8"),
        Stderr: SpawnError === null ? Stderr : `${Stderr}spawn failed: ${SpawnError}`,
        DurationMs: Date.now() - Start,
        TimedOut,
        OutputCapped,
      });
    };
    const FinishAfterGrace = (): void => {
      if (GraceTimer === null) GraceTimer = setTimeout(Finish, PipeCloseGraceMs);
    };
    const Collect = (Into: Buffer[]) => (Chunk: Buffer): void => {
      if (OutputCapped) return;
      const Room = MaxOutputBytes - Captured;
      if (Chunk.length > Room) {
        Into.push(Chunk.subarray(0, Room));
        Captured = MaxOutputBytes;
        OutputCapped = true;
        KillGroup(Child);
        FinishAfterGrace();
        return;
      }
      Into.push(Chunk);
      Captured += Chunk.length;
    };

    DeadlineTimer = setTimeout(() => {
      TimedOut = true;
      KillGroup(Child);
      FinishAfterGrace();
    }, Timeout);
    Child.stdout.on("data", Collect(Out));
    Child.stderr.on("data", Collect(ErrOut));
    Child.on("exit", (Code) => {
      ExitCode = Code;
      KillGroup(Child); // reap any background process the run left in its group
      FinishAfterGrace();
    });
    Child.on("close", (Code) => {
      if (ExitCode === null) ExitCode = Code;
      Finish();
    });
    Child.on("error", (Err) => {
      SpawnError = Err.message;
      Finish();
    });
    Child.stdin.on("error", () => {
      // the program may exit without reading its stdin (EPIPE) — not an execution failure by itself
    });
    Child.stdin.end(Stdin);
  });
}

/** Run a multi-file program asynchronously (non-blocking) under the caps described in the header. */
export async function RunProgramAsync(Program: ExecProgram, TimeoutMs = DefaultExecTimeoutMs): Promise<ExecResult> {
  ValidateProgram(Program);
  const Timeout = ClampTimeout(TimeoutMs);
  const Dir = mkdtempSync(join(tmpdir(), "shahd-exec-")); // mode 0700: private to this user
  try {
    for (const [Name, Content] of Object.entries(Program.Files)) writeFileSync(join(Dir, Name), Content, { mode: 0o600 });
    return await SpawnCapped(join(Dir, Program.Entry), Dir, Program.Stdin ?? "", Timeout);
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
}

/** Run one code string asynchronously (non-blocking). */
export function RunCodeAsync(Code: string, TimeoutMs = DefaultExecTimeoutMs): Promise<ExecResult> {
  return RunProgramAsync({ Files: { "Candidate.ts": Code }, Entry: "Candidate.ts" }, TimeoutMs);
}

function IsExecResult(Value: unknown): Value is ExecResult {
  if (typeof Value !== "object" || Value === null) return false;
  const V = Value as Record<string, unknown>;
  return typeof V["Passed"] === "boolean" && typeof V["Stdout"] === "string" && typeof V["Stderr"] === "string"
    && (V["ExitCode"] === null || typeof V["ExitCode"] === "number") && typeof V["DurationMs"] === "number"
    && typeof V["TimedOut"] === "boolean" && typeof V["OutputCapped"] === "boolean";
}

/** Synchronous sibling of RunProgramAsync (same caps). Blocks the calling thread for up to the clamped
 *  timeout — prefer the async API anywhere an event loop serves other work. */
export function RunProgram(Program: ExecProgram, TimeoutMs = DefaultExecTimeoutMs): ExecResult {
  ValidateProgram(Program);
  const Timeout = ClampTimeout(TimeoutMs);
  const Start = Date.now();
  const Proc = spawnSync(process.execPath, [...BunArgs(fileURLToPath(import.meta.url)), SupervisorFlag], {
    input: JSON.stringify({ Program, TimeoutMs: Timeout }),
    cwd: tmpdir(),
    env: ScrubbedEnv(),
    timeout: Timeout + SupervisorSlackMs,
    killSignal: "SIGKILL",
    maxBuffer: SupervisorMaxBuffer,
    windowsHide: true,
  });
  if (Proc.status === 0 && Proc.stdout !== null) {
    try {
      const Parsed: unknown = JSON.parse(Proc.stdout.toString("utf8"));
      if (IsExecResult(Parsed)) return Parsed;
    } catch {
      // fall through to the failure result below
    }
  }
  const Reason = Proc.error !== undefined ? Proc.error.message : `supervisor exited with ${Proc.status ?? Proc.signal}`;
  return FailedResult(`exec supervisor failed: ${Reason}`, Start, null, Proc.error !== undefined);
}

/** Run one code string synchronously. Passed = exit 0 within the caps. NOTE: exit status alone is not
 *  a trustworthy verdict for model-written code (it can call process.exit(0)) — verifiers must use
 *  EvalHarness.RunAgainstTests, which requires a per-run nonce printed after all assertions. */
export function RunCode(Code: string, TimeoutMs = DefaultExecTimeoutMs): ExecResult {
  return RunProgram({ Files: { "Candidate.ts": Code }, Entry: "Candidate.ts" }, TimeoutMs);
}

// Supervisor mode (sync path only): read { Program, TimeoutMs } from stdin, run it through the async
// hardened path, print the ExecResult as JSON. Exit 2 on a malformed request.
async function Supervise(): Promise<void> {
  const Request = JSON.parse(await Bun.stdin.text()) as { Program?: ExecProgram; TimeoutMs?: unknown };
  if (Request.Program === undefined || typeof Request.TimeoutMs !== "number") {
    process.exitCode = 2;
    return;
  }
  const Result = await RunProgramAsync(Request.Program, Request.TimeoutMs);
  await Bun.write(Bun.stdout, JSON.stringify(Result));
}

if (import.meta.main && process.argv.includes(SupervisorFlag)) {
  Supervise().catch((Err: unknown) => {
    process.stderr.write(`exec supervisor: ${(Err as Error).message}\n`);
    process.exitCode = 2;
  });
}
