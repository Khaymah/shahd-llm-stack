// Human-readable tables for SuiteReports: one report (overall + per-slice), and a comparison of a
// candidate against a baseline with deltas and direction-aware arrows — the "stronger = a number, not
// an impression" view. A metric with N = 0 was not measured; it prints "n/a" and is never compared.

import type { MetricResult, SuiteReport } from "./SuiteTypes.ts";
import { EvalSlices } from "./SuiteTypes.ts";

const ValueDigits = 4;
// Deltas smaller than this are treated as no change (float noise in a deterministic re-run).
const DeltaEpsilon = 1e-9;
const NotMeasured = "n/a";

export type Direction = "better" | "worse" | "same" | "n/a";
export type ComparisonRow = { Metric: string; Baseline: number | null; Candidate: number | null; Delta: number | null; Direction: Direction };
export type Comparison = { Rows: ComparisonRow[]; Table: string };

const Arrow: Record<Direction, string> = { better: "▲ better", worse: "▼ worse", same: "= same", "n/a": NotMeasured };

function Measured(Metric: MetricResult | undefined): number | null {
  return Metric !== undefined && Metric.N > 0 ? Metric.Value : null;
}

function Show(Value: number | null): string {
  return Value === null ? NotMeasured : Value.toFixed(ValueDigits);
}

function ShowDelta(Delta: number | null): string {
  if (Delta === null) return NotMeasured;
  return `${Delta >= 0 ? "+" : ""}${Delta.toFixed(ValueDigits)}`;
}

/** Render rows as an aligned text table. */
function Render(Header: readonly string[], Rows: readonly (readonly string[])[]): string {
  const Widths = Header.map((H, I) => Math.max(H.length, ...Rows.map((R) => R[I].length)));
  const Line = (Cells: readonly string[]): string => Cells.map((C, I) => C.padEnd(Widths[I])).join("  ").trimEnd();
  return [Line(Header), Widths.map((W) => "-".repeat(W)).join("  "), ...Rows.map(Line)].join("\n");
}

/** Every metric of a report as (label, metric): the overall list, then each slice's as "name [slice]". */
function Labeled(Report: SuiteReport): [string, MetricResult][] {
  const Out: [string, MetricResult][] = Report.Metrics.map((M) => [M.Name, M]);
  for (const Slice of EvalSlices) for (const M of Report.Slices[Slice] ?? []) Out.push([`${M.Name} [${Slice}]`, M]);
  return Out;
}

/** One report as a table (overall metrics, then per-slice), with a header line. */
export function FormatReport(Report: SuiteReport): string {
  const Rows = Labeled(Report).map(([Label, M]) => [Label, Show(Measured(M)), M.N > 0 ? String(M.N) : "0", M.Higher === "better" ? "higher" : "lower"]);
  const Title = `suite ${Report.SuiteVersion} · ${Report.Checkpoint} · config ${Report.ConfigHash} · ${Report.Params.toLocaleString("en-US")} params · ${(Report.DurationMs / 1000).toFixed(1)}s`;
  return `${Title}\n${Render(["metric", "value", "N", "is better"], Rows)}`;
}

function DirectionOf(Higher: MetricResult["Higher"], Delta: number | null): Direction {
  if (Delta === null) return "n/a";
  if (Math.abs(Delta) < DeltaEpsilon) return "same";
  return (Delta > 0) === (Higher === "better") ? "better" : "worse";
}

/** Compare Candidate against Baseline metric by metric (Delta = candidate - baseline). */
export function CompareReports(Baseline: SuiteReport, Candidate: SuiteReport): Comparison {
  const Base = new Map(Labeled(Baseline));
  const Cand = new Map(Labeled(Candidate));
  const Labels = [...Cand.keys(), ...[...Base.keys()].filter((L) => !Cand.has(L))];
  const Rows: ComparisonRow[] = Labels.map((Label) => {
    const B = Measured(Base.get(Label));
    const C = Measured(Cand.get(Label));
    const Delta = B !== null && C !== null ? C - B : null;
    const Higher = (Cand.get(Label) ?? Base.get(Label))?.Higher ?? "better";
    return { Metric: Label, Baseline: B, Candidate: C, Delta, Direction: DirectionOf(Higher, Delta) };
  });
  const Header = ["metric", `baseline (${Baseline.Checkpoint})`, `candidate (${Candidate.Checkpoint})`, "delta", ""];
  const Notes = Baseline.SuiteVersion === Candidate.SuiteVersion ? "" : `\n⚠ suite versions differ (${Baseline.SuiteVersion} vs ${Candidate.SuiteVersion}) — numbers are not directly comparable`;
  const Table = Render(Header, Rows.map((R) => [R.Metric, Show(R.Baseline), Show(R.Candidate), ShowDelta(R.Delta), Arrow[R.Direction]])) + Notes;
  return { Rows, Table };
}

