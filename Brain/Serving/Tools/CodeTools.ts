// Code-execution tool. Registered ONLY when Config.Tools.ExecEnabled is true (off by default —
// absolute safety). Routes through the resource-capped CodeExecutor (subprocess + timeout + private temp cwd; NOT a sandbox);
// see CodeExecutor's own note that production use needs OS-level container isolation on top.

import type { Tool } from "./ToolTypes.ts";
import { OptionalNumber, OptionalString } from "./ToolArgs.ts";
import { RunCode } from "../../Eval/CodeExecutor.ts";

// Execute code in a resource-capped subprocess (not a sandbox). Args: { code: string, timeoutMs?: number }.
export const RunCodeTool: Tool = {
  Name: "run_code",
  Description: "Execute code in a resource-capped subprocess (not a sandbox) and return stdout/stderr/exit.",
  Args: "{ code: string, timeoutMs?: number }",
  Execute: (Arguments) => {
    const Code = OptionalString(Arguments, "code", "");
    const TimeoutMs = OptionalNumber(Arguments, "timeoutMs", 5000);
    const Result = RunCode(Code, TimeoutMs);
    return {
      passed: Result.Passed,
      exitCode: Result.ExitCode,
      stdout: Result.Stdout,
      stderr: Result.Stderr,
    };
  },
};
