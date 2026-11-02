// What a synthetic tool conversation is ABOUT, decided by the pipeline before the teacher writes a word:
// the tool and the EXACT arguments (numbers, ops, paths, keys) are sampled here; the teacher only writes
// the user's phrasing and the scratchpad. That turns the eval's observed failure — right tool, wrong
// arguments — into a checkable property: a teacher call whose arguments differ from the intent is
// rejected, so every stored call maps a varied phrasing to exactly-right arguments. The checks below
// also require every number the user "gave" to appear in the user's text, and the final answer to echo
// the REAL result.

import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { SeededRng } from "../../Random/SeededRng.ts";
import type { ToolResult } from "../../Serving/Tools/ToolTypes.ts";
import { PickInt, PickOne } from "./SynthSampling.ts";

/** One planned call: the tool, its exact arguments, and what the user wants in words (for the teacher). */
export type ToolStepPlan = {
  Tool: string;
  Want: string;
  Arguments: Record<string, unknown>;
  Numbers: readonly number[]; // values the user must state literally in the request
  Hidden: readonly string[]; // strings the request must NOT contain (a recall must not restate the value)
};

export type ToolPlan = { Kind: string; Steps: readonly ToolStepPlan[] };

/** The fixture workspace as the file tools see it (relative, sorted). */
export type WorkspaceView = { Files: readonly string[]; Dirs: readonly string[] };

/** Walk a workspace root into a sorted view (bounded — the fixture is a handful of files). */
export function ReadWorkspaceView(Root: string): WorkspaceView {
  const Files: string[] = [];
  const Dirs: string[] = ["."];
  const Walk = (Prefix: string): void => {
    for (const Name of readdirSync(join(Root, Prefix)).sort()) {
      const Rel = Prefix === "" ? Name : `${Prefix}/${Name}`;
      if (statSync(join(Root, Rel)).isDirectory()) {
        Dirs.push(Rel);
        Walk(Rel);
      } else {
        Files.push(Rel);
      }
    }
  };
  Walk("");
  return { Files, Dirs };
}

const OpWords: Record<string, string> = { "+": "the sum of", "-": "the difference of", "*": "the product of", "/": "the quotient of", "^": "a power:", "%": "the remainder of" };
const StatsOps: readonly string[] = ["sum", "mean", "min", "max", "count"];
const DecimalChance = 0.25;
const CentsPerUnit = 100;

// A memory fact: the key Shahd stores it under, the value, and how the user refers to it.
const Facts: readonly [string, string, string][] = [
  ["api_base_url", "https://api.example.com/v2", "the API base URL"],
  ["db_port", "5432", "the database port"],
  ["deploy_branch", "release/2.4", "the deploy branch"],
  ["team_lead", "Mariam", "the team lead's name"],
  ["staging_host", "staging.internal.example.com", "the staging host"],
  ["node_version", "22.11.0", "the Node.js version the project uses"],
  ["release_date", "2026-03-15", "the release date"],
  ["cache_ttl", "300 seconds", "the cache TTL"],
  ["max_upload", "25 MB", "the maximum upload size"],
  ["primary_color", "#1f6feb", "the brand's primary color"],
];

function Operand(Rng: SeededRng, Min: number, Max: number): number {
  const Whole = PickInt(Rng, Min, Max);
  if (Rng.NextFloat() >= DecimalChance) return Whole;
  return Whole + PickInt(Rng, 1, CentsPerUnit - 1) / CentsPerUnit;
}

function CalculatorPlan(Rng: SeededRng): ToolPlan {
  const Op = PickOne(Rng, Object.keys(OpWords));
  let A: number;
  let B: number;
  if (Op === "^") [A, B] = [PickInt(Rng, 2, 12), PickInt(Rng, 2, 10)];
  else if (Op === "/" || Op === "%") [A, B] = [PickInt(Rng, 10, 9999), PickInt(Rng, 2, 60)];
  else [A, B] = [Operand(Rng, 2, 9999), Operand(Rng, 2, 999)];
  const Want = `to compute ${OpWords[Op]} ${A} and ${B} (operation "${Op}")`;
  return { Kind: "calculator", Steps: [{ Tool: "calculator", Want, Arguments: { a: A, op: Op, b: B }, Numbers: [A, B], Hidden: [] }] };
}

function StatsPlan(Rng: SeededRng): ToolPlan {
  const Values = Array.from({ length: PickInt(Rng, 3, 7) }, () => PickInt(Rng, 1, 500));
  const Op = PickOne(Rng, StatsOps);
  return { Kind: "stats", Steps: [{ Tool: "stats", Want: `the ${Op} of the numbers ${Values.join(", ")}`, Arguments: { values: Values, op: Op }, Numbers: Values, Hidden: [] }] };
}

function TimePlan(): ToolPlan {
  return { Kind: "current_time", Steps: [{ Tool: "current_time", Want: "to know the current date or time (for example for a log entry, a deadline or a timestamp)", Arguments: {}, Numbers: [], Hidden: [] }] };
}

function StoreStep(Fact: [string, string, string]): ToolStepPlan {
  const [Key, Value, About] = Fact;
  return { Tool: "memory_store", Want: `Shahd to remember ${About}: ${Value}`, Arguments: { key: Key, value: Value }, Numbers: [], Hidden: [] };
}

function MemoryStorePlan(Rng: SeededRng): ToolPlan {
  return { Kind: "memory_store", Steps: [StoreStep(PickOne(Rng, Facts))] };
}

function MemoryRecallPlan(Rng: SeededRng): ToolPlan {
  const Fact = PickOne(Rng, Facts);
  const Recall: ToolStepPlan = { Tool: "memory_recall", Want: `Shahd to tell them ${Fact[2]} they saved earlier (without restating it)`, Arguments: { key: Fact[0] }, Numbers: [], Hidden: [Fact[1]] };
  return { Kind: "memory_recall", Steps: [StoreStep(Fact), Recall] };
}

function FileReadPlan(Rng: SeededRng, View: WorkspaceView): ToolPlan {
  const Path = PickOne(Rng, View.Files);
  return { Kind: "file_read", Steps: [{ Tool: "file_read", Want: `to see or ask about the file ${Path}`, Arguments: { path: Path }, Numbers: [], Hidden: [] }] };
}

function FileListPlan(Rng: SeededRng, View: WorkspaceView): ToolPlan {
  const Path = PickOne(Rng, View.Dirs);
  const Where = Path === "." ? "the project root" : `the ${Path} folder`;
  return { Kind: "file_list", Steps: [{ Tool: "file_list", Want: `to know what is in ${Where}`, Arguments: { path: Path }, Numbers: [], Hidden: [] }] };
}

/** Substrings worth searching the fixture for: file stems and extensions. */
function SearchQueries(View: WorkspaceView): string[] {
  const Out = new Set<string>();
  for (const File of View.Files) {
    const Name = File.split("/").pop() ?? File;
    const Dot = Name.lastIndexOf(".");
    if (Dot > 0) {
      Out.add(Name.slice(0, Dot));
      Out.add(Name.slice(Dot));
    }
  }
  return [...Out].sort();
}

function FileSearchPlan(Rng: SeededRng, View: WorkspaceView): ToolPlan {
  const Query = PickOne(Rng, SearchQueries(View));
  return { Kind: "file_search", Steps: [{ Tool: "file_search", Want: `to find the files whose path contains "${Query}"`, Arguments: { query: Query }, Numbers: [], Hidden: [] }] };
}

// Scenario weights: arithmetic and data dominate (the eval's largest tool groups), every tool appears.
const Scenarios: readonly [number, (Rng: SeededRng, View: WorkspaceView) => ToolPlan][] = [
  [0.3, CalculatorPlan],
  [0.15, StatsPlan],
  [0.08, TimePlan],
  [0.1, MemoryStorePlan],
  [0.1, MemoryRecallPlan],
  [0.12, FileReadPlan],
  [0.07, FileListPlan],
  [0.08, FileSearchPlan],
];

/** Sample one tool conversation plan. */
export function PlanToolConversation(Rng: SeededRng, View: WorkspaceView): ToolPlan {
  const Total = Scenarios.reduce((Sum, [Weight]) => Sum + Weight, 0);
  let Draw = Rng.NextFloat() * Total;
  for (const [Weight, Plan] of Scenarios) {
    if (Draw < Weight) return Plan(Rng, View);
    Draw -= Weight;
  }
  return Scenarios[Scenarios.length - 1][1](Rng, View);
}

// Decimal places tried when matching a number written in text (12.5 may appear as 12.50 or 12.5).
const MaxDecimals = 4;

/** The ways a number may legitimately be written: plain, with thousands separators, or rounded to
 *  1..4 decimals (a quotient like 3.3333… is fairly reported as 3.33). */
export function NumberSpellings(Value: number): string[] {
  const Out = new Set<string>([String(Value), Value.toLocaleString("en-US", { maximumFractionDigits: MaxDecimals })]);
  for (let Places = 1; Places <= MaxDecimals; Places++) Out.add(Value.toFixed(Places));
  return [...Out];
}

function EscapeRegExp(Text: string): string {
  return Text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** True when Text states the number (any spelling, not as part of a longer number). */
export function MentionsNumber(Text: string, Value: number): boolean {
  return NumberSpellings(Value).some((Spelling) => new RegExp(`(?<![\\d.])${EscapeRegExp(Spelling)}(?![\\d])`).test(Text));
}

/** Strings of which the final answer must contain at least one (empty = no echo requirement). */
export function ResultEchoes(Tool: string, Result: ToolResult): { Numbers: number[]; Texts: string[] } {
  const Value = Result["result"];
  if ((Tool === "calculator" || Tool === "stats") && typeof Value === "number") return { Numbers: [Value], Texts: [] };
  if (Tool === "current_time" && typeof Result["iso"] === "string") {
    const Iso = Result["iso"];
    return { Numbers: [], Texts: [Iso.slice(0, 10), Iso.slice(11, 16), Iso.slice(0, 4)] };
  }
  if (Tool === "memory_recall" && typeof Result["value"] === "string") return { Numbers: [], Texts: [Result["value"]] };
  if (Tool === "file_list" && Array.isArray(Result["entries"])) {
    return { Numbers: [], Texts: (Result["entries"] as { name?: unknown }[]).map((Entry) => String(Entry.name ?? "")).filter((Name) => Name !== "") };
  }
  if (Tool === "file_search" && Array.isArray(Result["matches"])) {
    const Matches = (Result["matches"] as unknown[]).map(String);
    return { Numbers: [Matches.length], Texts: Matches.flatMap((Match) => [Match, Match.split("/").pop() ?? Match]) };
  }
  return { Numbers: [], Texts: [] };
}

/** True when the answer echoes the real result (or the tool has no echo requirement). */
export function AnswerEchoesResult(Answer: string, Tool: string, Result: ToolResult): boolean {
  const Echo = ResultEchoes(Tool, Result);
  if (Echo.Numbers.length === 0 && Echo.Texts.length === 0) return true;
  return Echo.Numbers.some((Value) => MentionsNumber(Answer, Value)) || Echo.Texts.some((Text) => Answer.includes(Text));
}
