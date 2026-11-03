// String / JSON / regex tools. The regex tool screens for catastrophic-backtracking patterns and
// caps input length (safety: a user-supplied regex is untrusted and could otherwise ReDoS-hang the
// server) — the same intent-oriented, ReDoS-safe discipline the content filter uses.

import type { Tool, ToolResult } from "./ToolTypes.ts";
import { Err, RequireString, OptionalString, OptionalBool } from "./ToolArgs.ts";
import { RunCode } from "../../Eval/CodeExecutor.ts";

const MaxRegexInput = 100_000;
const RegexTimeoutMs = 2000; // hard cap for the isolated subprocess that runs quantifier-bearing patterns

// Heuristic FAST-REJECT for the classic obvious ReDoS shapes — a clear early error, NOT the safety
// boundary (it is provably incomplete, e.g. it misses (a|aa)+). The real safety comes from running any
// quantifier-bearing pattern in an isolated, timeout-killed subprocess (RunFencedRegex) below.
function LooksLikeReDoS(Pattern: string): boolean {
  return /(\([^)]*[+*][^)]*\)[+*])|(\(\.\*\)[+*])|(\[[^\]]*\][+*])[+*]/.test(Pattern);
}

// Only a pattern with an unbounded/large quantifier (+ * {) can catastrophically backtrack; those are
// executed in a killable subprocess so a crafted pattern is terminated at the timeout instead of
// freezing the single-threaded server forever. Patterns without one can't ReDoS and run in-process.
function HasQuantifier(Pattern: string): boolean {
  return /[+*{]/.test(Pattern);
}

// Run the regex in an isolated subprocess with a hard timeout (reusing the resource-capped executor). A
// catastrophic pattern is KILLED at RegexTimeoutMs; the worst case is a bounded stall, never a hang.
function RunFencedRegex(Pattern: string, Flags: string, Text: string, Action: string, Replacement: string): ToolResult {
  const Script =
    `const Re = new RegExp(${JSON.stringify(Pattern)}, ${JSON.stringify(Flags)});\n` +
    `const Text = ${JSON.stringify(Text)};\n` +
    `let Out;\n` +
    `if (${JSON.stringify(Action)} === "replace") { Out = { text: Text.replace(Re, ${JSON.stringify(Replacement)}) }; }\n` +
    `else { const G = Re.global ? Re : new RegExp(${JSON.stringify(Pattern)}, ${JSON.stringify(Flags)} + "g"); const M = [...Text.matchAll(G)].map((x) => x[0]); Out = { matches: M, count: M.length }; }\n` +
    `console.log(JSON.stringify(Out));`;
  const Result = RunCode(Script, RegexTimeoutMs);
  if (!Result.Passed) return Err("pattern rejected: possible catastrophic backtracking (timed out or failed in the subprocess)");
  try {
    return JSON.parse(Result.Stdout.trim()) as ToolResult;
  } catch {
    return Err("regex subprocess produced no parseable result");
  }
}

// Parse or stringify JSON. Args: { action: 'parse'|'stringify', input: string, pretty?: boolean }.
export const JsonTool: Tool = {
  Name: "json",
  Description: "Parse a JSON string or stringify a value.",
  Args: "{ action: 'parse'|'stringify', input: string, pretty?: boolean }",
  Execute: (Arguments) => {
    const Action = RequireString(Arguments, "action");
    if (Action === "parse") {
      try {
        return { value: JSON.parse(RequireString(Arguments, "input")) };
      } catch (Error_) {
        return Err(`invalid JSON: ${(Error_ as Error).message}`);
      }
    }
    if (Action === "stringify") {
      const Pretty = OptionalBool(Arguments, "pretty", false);
      return { text: JSON.stringify(Arguments["input"], null, Pretty ? 2 : 0) };
    }
    return Err(`unknown action: ${Action}`);
  },
};

// Match or replace with a regex. Args: { pattern, text, flags?, action?: 'match'|'replace', replacement? }.
export const RegexTool: Tool = {
  Name: "regex",
  Description: "Regex match/replace over text (ReDoS-screened, input-capped).",
  Args: "{ pattern: string, text: string, flags?: string, action?: 'match'|'replace', replacement?: string }",
  Execute: (Arguments) => {
    const Pattern = RequireString(Arguments, "pattern");
    const Text = RequireString(Arguments, "text");
    if (Text.length > MaxRegexInput) return Err(`text exceeds ${MaxRegexInput} chars`);
    if (Pattern.length > MaxRegexInput) return Err(`pattern exceeds ${MaxRegexInput} chars`);
    if (LooksLikeReDoS(Pattern)) return Err("pattern rejected: possible catastrophic backtracking");
    const Flags = OptionalString(Arguments, "flags", "");
    const Action = OptionalString(Arguments, "action", "match");
    if (Action !== "match" && Action !== "replace") return Err(`unknown action: ${Action}`);
    const Replacement = OptionalString(Arguments, "replacement", "");
    if (Replacement.length > MaxRegexInput) return Err(`replacement exceeds ${MaxRegexInput} chars`);
    // Validate the pattern compiles (fast, clear error) before any execution.
    try {
      new RegExp(Pattern, Flags);
    } catch (Error_) {
      return Err(`invalid regex: ${(Error_ as Error).message}`);
    }
    // Quantifier-bearing patterns run fenced (subprocess + hard timeout); the rest run in-process.
    if (HasQuantifier(Pattern)) return RunFencedRegex(Pattern, Flags, Text, Action, Replacement);
    const Re = new RegExp(Pattern, Flags);
    if (Action === "replace") return { text: Text.replace(Re, Replacement) };
    const Matches = [...Text.matchAll(Re.global ? Re : new RegExp(Pattern, Flags + "g"))].map((M) => M[0]);
    return { matches: Matches, count: Matches.length };
  },
};

// Common string transforms. Args: { action, text, sep?, needle? }.
export const TextTool: Tool = {
  Name: "text",
  Description: "String transforms: upper/lower/trim/length/reverse/split/count.",
  Args: "{ action: 'upper'|'lower'|'trim'|'length'|'reverse'|'split'|'count', text: string, sep?: string, needle?: string }",
  Execute: (Arguments) => {
    const Action = RequireString(Arguments, "action");
    const Text = RequireString(Arguments, "text");
    switch (Action) {
      case "upper": return { text: Text.toUpperCase() };
      case "lower": return { text: Text.toLowerCase() };
      case "trim": return { text: Text.trim() };
      case "length": return { result: Text.length };
      case "reverse": return { text: [...Text].reverse().join("") };
      case "split": return { parts: Text.split(OptionalString(Arguments, "sep", "\n")) };
      case "count": {
        const Needle = RequireString(Arguments, "needle");
        return { result: Needle === "" ? 0 : Text.split(Needle).length - 1 };
      }
      default: return Err(`unknown action: ${Action}`);
    }
  },
};
