// Builders for CodeCases. Static cases carry regex StaticChecks, applied with StaticCheckFlags.
// Executable cases build their `Tests` text here (EvalHarness contract: plain statements that
// reference the candidate's definitions by name and THROW on failure — no imports). Every case set
// file builds its tests here so the comparison semantics are identical across the whole suite:
//
//   • Expect(Actual, Expected, Label) compares a CANONICAL serialization: arrays are order-sensitive,
//     object keys are sorted (key order never decides a verdict), primitives use JSON. `undefined` is
//     distinct from `null`, and an object property holding `undefined` is NOT the same as a missing one.
//   • Expected values are embedded as JSON literals, so they are fixed text inside the frozen set.
//
// The prelude names (Canon, Expect) are block-scoped inside the test body, so a candidate that happens
// to declare the same names can neither shadow nor patch them.

import type { CodeCase } from "../SuiteTypes.ts";

/** Flags every StaticChecks regex is compiled with: case-insensitive, and `.` spans newlines so a
 *  check can look across a multi-line answer. The runner and the case-set test share this value. */
export const StaticCheckFlags = "is";

/** A raw statement, or an [expression, expected value] pair checked with Expect. */
export type TestStep = string | readonly [string, unknown];

const TestPrelude = [
  "const Canon = (V) => Array.isArray(V)",
  "  ? \"[\" + V.map(Canon).join(\",\") + \"]\"",
  "  : V !== null && typeof V === \"object\"",
  "    ? \"{\" + Object.keys(V).sort().map((K) => JSON.stringify(K) + \":\" + Canon(V[K])).join(\",\") + \"}\"",
  "    : V === undefined ? \"undefined\" : JSON.stringify(V);",
  "const Expect = (Actual, Expected, Label) => {",
  "  if (Canon(Actual) !== Canon(Expected)) throw new Error(Label + \" => got \" + Canon(Actual) + \", want \" + Canon(Expected));",
  "};",
].join("\n");

/** The source text of a call: `Name(arg1, arg2, …)` with every argument written as a JSON literal. */
export function CallExpr(Name: string, ...Args: readonly unknown[]): string {
  return `${Name}(${Args.map((Arg) => (Arg === undefined ? "undefined" : JSON.stringify(Arg))).join(", ")})`;
}

/** Turn steps into the Tests text: the shared prelude followed by one statement per step. */
export function BuildTests(Steps: readonly TestStep[]): string {
  const Lines = Steps.map((Step) =>
    typeof Step === "string"
      ? Step
      : `Expect(${Step[0]}, ${Step[1] === undefined ? "undefined" : JSON.stringify(Step[1])}, ${JSON.stringify(Step[0])});`,
  );
  return [TestPrelude, ...Lines].join("\n");
}

/** An executable (ts/js) case: Tests are built from Steps with the shared prelude. */
export function ExecutableCase(Id: string, Slice: "ts" | "js", Prompt: string, Reference: string, Steps: readonly TestStep[]): CodeCase {
  return { Id, Slice, Prompt, Reference, Tests: BuildTests(Steps) };
}

/** A static (html/css/go/sql) case: every regex in Checks must match the answer (StaticCheckFlags). */
export function StaticCase(Id: string, Slice: "html" | "css" | "go" | "sql", Prompt: string, Reference: string, Checks: readonly string[]): CodeCase {
  return { Id, Slice, Prompt, Reference, StaticChecks: Checks };
}
