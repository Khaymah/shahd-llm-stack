// The fixed evaluation suite's contract (Plan: Brain-Strengthening, phase 1). Every metric produces
// a MetricResult; the runner (RunSuite.ts) assembles them into one SuiteReport per checkpoint, which
// is what "stronger" means from now on — a number against a frozen baseline, not an impression.
//
// Slices: the brain is grown as a curriculum (ts/js -> html/css -> go -> sql), so each case carries a
// Slice and the report groups metrics per slice as well as overall. The case sets live in
// Brain/Eval/Suite/Data/ and are FROZEN once a baseline is recorded — add a new versioned set rather
// than editing cases, or the numbers stop being comparable.

/** Domain slice of a case. "general" = not domain-specific (identity, format, tools). */
export type EvalSlice = "ts" | "js" | "html" | "css" | "go" | "sql" | "general";

export const EvalSlices: readonly EvalSlice[] = ["ts", "js", "html", "css", "go", "sql", "general"];

/** One number with its direction, so a report can be compared mechanically. */
export type MetricResult = {
  Name: string; // e.g. "bpb.code", "code.pass@1", "format.compliance", "tool.accuracy", "multiturn.coherence", "memorization.ngram"
  Value: number;
  Higher: "better" | "worse";
  N: number; // cases / tokens the number was computed over
  Details?: Record<string, unknown>; // per-case breakdown, failure examples (bounded), timings
};

/** A coding problem verified by executing tests (Bun runs TS/JS; other slices use StaticChecks). */
export type CodeCase = {
  Id: string; // stable, e.g. "ts-001"
  Slice: EvalSlice;
  Prompt: string; // the user message
  /** Statements that reference the candidate's definitions and throw on failure (EvalHarness contract). */
  Tests?: string;
  /** For non-executable slices (html/css/sql/go without a toolchain): regexes that must ALL match the answer. */
  StaticChecks?: readonly string[];
  Reference: string; // a known-good answer; used to self-check the case, never shown to the model
};

/** A prompt whose reply is judged on format only: replies, stops at end-of-turn, no leaked control tokens. */
export type FormatCase = { Id: string; Slice: EvalSlice; Messages: readonly { Role: "System" | "User" | "Assistant"; Content: string }[] };

/** A prompt that should produce exactly one tool call with these arguments (subset match on Arguments). */
export type ToolCase = { Id: string; Slice: EvalSlice; User: string; Expected: { Name: string; Arguments: Record<string, unknown> } };

/** A multi-turn conversation; the final user turn depends on an earlier one, and the reply must satisfy the checks. */
export type MultiTurnCase = {
  Id: string;
  Slice: EvalSlice;
  Turns: readonly { Role: "User" | "Assistant"; Content: string }[]; // ends with a User turn
  MustContain?: readonly string[]; // case-insensitive substrings expected in the reply
  MustNotContain?: readonly string[];
};

/** Everything the runner needs about a case set, frozen under a version tag. */
export type EvalCaseSet = {
  Version: string; // e.g. "v1" — bump when cases change
  Code: readonly CodeCase[];
  Format: readonly FormatCase[];
  Tools: readonly ToolCase[];
  MultiTurn: readonly MultiTurnCase[];
};

export type SuiteReport = {
  SuiteVersion: string; // EvalCaseSet.Version
  Checkpoint: string; // name or path
  ConfigHash: string;
  Params: number;
  RanAt: string; // ISO
  DurationMs: number;
  Metrics: MetricResult[]; // overall
  Slices: Partial<Record<EvalSlice, MetricResult[]>>; // per-slice breakdown of the case-based metrics
};
