// Self-checks for the frozen v1 eval case set (Brain/Eval/Suite/Data). A case that cannot be passed by
// a correct answer — or that a wrong answer passes — would silently corrupt every number recorded
// against it, so each case proves itself here: references pass through the REAL nonce verifier, a
// do-nothing stub fails it, every static regex compiles and matches its reference, tool names exist.

import { test, expect } from "bun:test";
import { EvalCaseSetV1 } from "../Brain/Eval/Suite/Data/CaseSetV1.ts";
import { StaticCheckFlags } from "../Brain/Eval/Suite/Data/CodeCaseBuilder.ts";
import { RunAgainstTestsAsync } from "../Brain/Eval/EvalHarness.ts";
import { BuildToolRegistry } from "../Brain/Serving/Tools/BuildRegistry.ts";
import type { CodeCase, EvalSlice } from "../Brain/Eval/Suite/SuiteTypes.ts";

// Spawning one verifier subprocess per reference (and per stub) takes a few seconds in total on CPU.
const VerifierBudgetMs = 120_000;

const ExecutableSlices: readonly EvalSlice[] = ["ts", "js"];
const StaticSlices: readonly EvalSlice[] = ["html", "css", "go", "sql"];

const ExpectedCodeCounts: Readonly<Partial<Record<EvalSlice, number>>> = { ts: 35, js: 15, html: 12, css: 10, go: 10, sql: 8 };
const ExpectedFormatCount = 60;
const ExpectedToolCount = 40;
const ExpectedMultiTurnCount = 30;
const MinFormatCasesWithHistory = 10;
const MinMultiTurnMessages = 3;
const MaxMultiTurnMessages = 5;

const CalculatorOps: readonly string[] = ["+", "-", "*", "/", "^", "%"];
const StatsOps: readonly string[] = ["sum", "mean", "min", "max", "count"];

const Code = EvalCaseSetV1.Code;
const ExecutableCases = Code.filter((C) => ExecutableSlices.includes(C.Slice));
const StaticCases = Code.filter((C) => StaticSlices.includes(C.Slice));

/** The function/class name a code prompt asks for: the first backticked identifier. */
function RequestedName(Case: CodeCase): string {
  const Match = /`(?:class\s+)?([A-Za-z_$][\w$]*)/.exec(Case.Prompt);
  if (Match === null || Match[1] === undefined) throw new Error(`${Case.Id}: prompt names no function`);
  return Match[1];
}

function CountBy<T>(Items: readonly T[], Key: (Item: T) => string): Record<string, number> {
  const Counts: Record<string, number> = {};
  for (const Item of Items) Counts[Key(Item)] = (Counts[Key(Item)] ?? 0) + 1;
  return Counts;
}

test("v1 has the frozen composition: per-slice code counts and per-category totals", () => {
  expect(EvalCaseSetV1.Version).toBe("v1");
  expect(CountBy(Code, (C) => C.Slice)).toEqual(ExpectedCodeCounts as Record<string, number>);
  expect(EvalCaseSetV1.Format.length).toBe(ExpectedFormatCount);
  expect(EvalCaseSetV1.Tools.length).toBe(ExpectedToolCount);
  expect(EvalCaseSetV1.MultiTurn.length).toBe(ExpectedMultiTurnCount);
  for (const Slice of [...ExecutableSlices, ...StaticSlices]) expect(Code.some((C) => C.Slice === Slice)).toBe(true);
});

test("ids are unique across the whole set and follow <prefix>-NNN (code prefix = its slice)", () => {
  const Ids = [
    ...Code.map((C) => C.Id),
    ...EvalCaseSetV1.Format.map((C) => C.Id),
    ...EvalCaseSetV1.Tools.map((C) => C.Id),
    ...EvalCaseSetV1.MultiTurn.map((C) => C.Id),
  ];
  expect(new Set(Ids).size).toBe(Ids.length);
  for (const C of Code) expect(C.Id).toMatch(new RegExp(`^${C.Slice}-\\d{3}$`));
  for (const C of EvalCaseSetV1.Format) expect(C.Id).toMatch(/^fmt-\d{3}$/);
  for (const C of EvalCaseSetV1.Tools) expect(C.Id).toMatch(/^tool-\d{3}$/);
  for (const C of EvalCaseSetV1.MultiTurn) expect(C.Id).toMatch(/^mt-\d{3}$/);
});

test("no prompt is duplicated within a category", () => {
  expect(new Set(Code.map((C) => C.Prompt)).size).toBe(Code.length);
  expect(new Set(EvalCaseSetV1.Tools.map((C) => C.User)).size).toBe(EvalCaseSetV1.Tools.length);
  const FormatKeys = EvalCaseSetV1.Format.map((C) => JSON.stringify(C.Messages));
  expect(new Set(FormatKeys).size).toBe(FormatKeys.length);
});

test("executable slices carry Tests (naming the requested function) and static slices carry StaticChecks", () => {
  for (const C of ExecutableCases) {
    expect(C.StaticChecks).toBeUndefined();
    expect(typeof C.Tests).toBe("string");
    expect(C.Tests ?? "").toContain(RequestedName(C));
  }
  for (const C of StaticCases) {
    expect(C.Tests).toBeUndefined();
    expect((C.StaticChecks ?? []).length).toBeGreaterThan(0);
  }
});

test("every executable reference passes its own tests through the nonce verifier (as-is and exported)", async () => {
  const Verdicts = await Promise.all(
    ExecutableCases.flatMap((C) => {
      const Problem = { Name: C.Id, Tests: C.Tests ?? "" };
      return [
        RunAgainstTestsAsync(Problem, C.Reference).then((Passed) => ({ Id: C.Id, Form: "plain", Passed })),
        RunAgainstTestsAsync(Problem, `export ${C.Reference}`).then((Passed) => ({ Id: C.Id, Form: "export", Passed })),
      ];
    }),
  );
  expect(Verdicts.filter((V) => !V.Passed)).toEqual([]);
}, VerifierBudgetMs);

test("a do-nothing stub with the requested name fails every executable case (tests are not vacuous)", async () => {
  const Verdicts = await Promise.all(
    ExecutableCases.map((C) => {
      const Name = RequestedName(C);
      const Stub = C.Prompt.includes("class `") ? `class ${Name} {}` : `function ${Name}(..._Args: unknown[]): unknown { return undefined; }`;
      return RunAgainstTestsAsync({ Name: C.Id, Tests: C.Tests ?? "" }, Stub).then((Passed) => ({ Id: C.Id, Passed }));
    }),
  );
  expect(Verdicts.filter((V) => V.Passed)).toEqual([]);
}, VerifierBudgetMs);

test("every StaticChecks regex compiles, matches its reference, and does not match an empty answer", () => {
  const Failures: string[] = [];
  for (const C of StaticCases) {
    for (const Pattern of C.StaticChecks ?? []) {
      let Compiled: RegExp;
      try {
        Compiled = new RegExp(Pattern, StaticCheckFlags);
      } catch {
        Failures.push(`${C.Id}: does not compile: ${Pattern}`);
        continue;
      }
      if (!Compiled.test(C.Reference)) Failures.push(`${C.Id}: reference misses ${Pattern}`);
      if (Compiled.test("")) Failures.push(`${C.Id}: vacuous (matches "") ${Pattern}`);
    }
  }
  expect(Failures).toEqual([]);
});

test("a static case's reference does not satisfy another case's checks in the same slice (checks are specific)", () => {
  const Leaks: string[] = [];
  for (const C of StaticCases) {
    for (const Other of StaticCases) {
      if (Other === C || Other.Slice !== C.Slice) continue;
      const AllMatch = (C.StaticChecks ?? []).every((Pattern) => new RegExp(Pattern, StaticCheckFlags).test(Other.Reference));
      if (AllMatch) Leaks.push(`${Other.Id} satisfies ${C.Id}`);
    }
  }
  expect(Leaks).toEqual([]);
});

test("tool cases name tools that exist and their expected arguments follow the tool's schema", async () => {
  const Registry = BuildToolRegistry({ FileAccess: "ReadWrite", ExecEnabled: true, WebSearchEnabled: true });
  for (const C of EvalCaseSetV1.Tools) {
    expect(Registry.Has(C.Expected.Name)).toBe(true);
    expect(C.Slice).toBe("general");
    const Args = C.Expected.Arguments;
    if (C.Expected.Name === "calculator") {
      expect(typeof Args["a"]).toBe("number");
      expect(typeof Args["b"]).toBe("number");
      expect(CalculatorOps).toContain(Args["op"] as string);
    }
    if (C.Expected.Name === "stats") {
      expect(StatsOps).toContain(Args["op"] as string);
      expect((Args["values"] as unknown[]).every((V) => typeof V === "number")).toBe(true);
    }
    if (C.Expected.Name === "calculator" || C.Expected.Name === "stats") {
      // Pure tools: the expected call must actually execute cleanly.
      const Result = await Registry.Run({ Name: C.Expected.Name, Arguments: Args });
      expect(Result).not.toHaveProperty("error");
    }
    for (const Key of ["code", "path", "query", "value"]) {
      if (Key in Args) expect(typeof Args[Key]).toBe("string");
    }
  }
});

test("format cases: system prompt only first, prior turns alternate, last turn is the User", () => {
  let WithHistory = 0;
  for (const C of EvalCaseSetV1.Format) {
    const Turns = C.Messages.filter((M) => M.Role !== "System");
    expect(C.Messages.findIndex((M) => M.Role === "System")).toBeLessThanOrEqual(0);
    expect(Turns.length).toBeGreaterThan(0);
    Turns.forEach((M, I) => expect(M.Role).toBe(I % 2 === 0 ? "User" : "Assistant"));
    expect(Turns[Turns.length - 1]?.Role).toBe("User");
    for (const M of C.Messages) expect(M.Content.trim().length).toBeGreaterThan(0);
    if (Turns.some((M) => M.Role === "Assistant")) WithHistory++;
  }
  expect(WithHistory).toBeGreaterThanOrEqual(MinFormatCasesWithHistory);
});

test("multi-turn cases alternate User/Assistant, end with a User turn, and have consistent checks", () => {
  for (const C of EvalCaseSetV1.MultiTurn) {
    expect(C.Turns.length).toBeGreaterThanOrEqual(MinMultiTurnMessages);
    expect(C.Turns.length).toBeLessThanOrEqual(MaxMultiTurnMessages);
    C.Turns.forEach((M, I) => expect(M.Role).toBe(I % 2 === 0 ? "User" : "Assistant"));
    expect(C.Turns[C.Turns.length - 1]?.Role).toBe("User");
    const Must = C.MustContain ?? [];
    const MustNot = C.MustNotContain ?? [];
    expect(Must.length).toBeGreaterThan(0);
    for (const S of [...Must, ...MustNot]) expect(S.trim().length).toBeGreaterThan(0);
    // A reply that satisfies MustContain must be able to avoid MustNotContain (case-insensitive).
    for (const Bad of MustNot) for (const Good of Must) expect(Good.toLowerCase().includes(Bad.toLowerCase())).toBe(false);
  }
});
