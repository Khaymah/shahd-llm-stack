// The FROZEN v1 evaluation case set (Plan: Brain-Strengthening §3, phase 1). Once a baseline is
// recorded against "v1", these cases are never edited: a change ships as a new versioned set (CaseSetV2)
// so numbers stay comparable across checkpoints. Self-checks live in Tests/EvalCaseSet.Test.ts.
// StaticChecks regexes are compiled with StaticCheckFlags (Data/CodeCaseBuilder.ts).
//
// Composition (counts asserted by the test):
//   Code      90 — ts 35 + js 15 (executable, Tests) · html 12 + css 10 + go 10 + sql 8 (StaticChecks)
//   Format    60 — single turns, system-prompt variations, 10 with prior Assistant turns
//   Tools     40 — calculator, stats, run_code, file_read/list/search, web_search, memory_*, current_time
//   MultiTurn 30 — renames, follow-up edits and recall questions that depend on earlier turns

import type { EvalCaseSet } from "../SuiteTypes.ts";
import { TsCodeCases } from "./CodeCasesTs.ts";
import { JsCodeCases } from "./CodeCasesJs.ts";
import { HtmlCodeCases, CssCodeCases } from "./CodeCasesWeb.ts";
import { GoCodeCases, SqlCodeCases } from "./CodeCasesGoSql.ts";
import { FormatCases } from "./FormatCases.ts";
import { ToolCases } from "./ToolCases.ts";
import { MultiTurnCases } from "./MultiTurnCases.ts";

export const EvalCaseSetV1: EvalCaseSet = {
  Version: "v1",
  Code: [...TsCodeCases, ...JsCodeCases, ...HtmlCodeCases, ...CssCodeCases, ...GoCodeCases, ...SqlCodeCases],
  Format: FormatCases,
  Tools: ToolCases,
  MultiTurn: MultiTurnCases,
};
