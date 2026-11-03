// Building MetricResults from per-case outcomes — the one place that turns "case X scored S" into the
// overall number and its per-slice breakdown, so every case-based metric aggregates identically.

import type { EvalSlice, MetricResult } from "./SuiteTypes.ts";
import { EvalSlices } from "./SuiteTypes.ts";

/** One case's result for one metric: Score in [0, 1] (1 = pass), plus why it failed (if it did). */
export type CaseOutcome = { Id: string; Slice: EvalSlice; Score: number; Reasons: readonly string[]; Excerpt?: string };

/** A metric computed over a list of case outcomes. */
export type CaseMetric = { Name: string; Higher: MetricResult["Higher"]; Outcomes: readonly CaseOutcome[] };

/** Failure examples kept in Details — bounded so a report stays small enough for checkpoint Meta. */
export const MaxFailureExamples = 5;
/** Characters of a reply kept in a failure example. */
export const MaxExcerptChars = 160;

/** Shorten a reply for a failure example. */
export function Excerpt(Text: string): string {
  return Text.length > MaxExcerptChars ? `${Text.slice(0, MaxExcerptChars)}…` : Text;
}

/** How many failing cases carry each failure reason. */
export function CountReasons(Outcomes: readonly CaseOutcome[]): Record<string, number> {
  const Counts: Record<string, number> = {};
  for (const O of Outcomes) for (const Reason of O.Reasons) Counts[Reason] = (Counts[Reason] ?? 0) + 1;
  return Counts;
}

function MeanScore(Outcomes: readonly CaseOutcome[]): number {
  if (Outcomes.length === 0) return 0;
  let Sum = 0;
  for (const O of Outcomes) Sum += O.Score;
  return Sum / Outcomes.length;
}

/** The overall MetricResult: mean score over the cases, failure-reason counts and a few examples. */
export function AggregateMetric(Metric: CaseMetric, Extra: Record<string, unknown> = {}): MetricResult {
  const Failures = Metric.Outcomes.filter((O) => O.Score < 1);
  return {
    Name: Metric.Name,
    Value: MeanScore(Metric.Outcomes),
    Higher: Metric.Higher,
    N: Metric.Outcomes.length,
    Details: {
      ...Extra,
      FailureReasons: CountReasons(Metric.Outcomes),
      FailureExamples: Failures.slice(0, MaxFailureExamples).map((O) => ({ Id: O.Id, Reasons: O.Reasons, ...(O.Excerpt !== undefined ? { Excerpt: O.Excerpt } : {}) })),
    },
  };
}

/** A metric that was NOT measured (N = 0); Details.Skipped says why. Never read Value when N is 0. */
export function SkippedMetric(Name: string, Higher: MetricResult["Higher"], Reason: string): MetricResult {
  return { Name, Value: 0, Higher, N: 0, Details: { Skipped: Reason } };
}

/** Per-slice breakdown of case metrics: each slice gets the same metric names over its own cases. */
export function SliceBreakdown(Metrics: readonly CaseMetric[]): Partial<Record<EvalSlice, MetricResult[]>> {
  const Out: Partial<Record<EvalSlice, MetricResult[]>> = {};
  for (const Slice of EvalSlices) {
    for (const Metric of Metrics) {
      const Outcomes = Metric.Outcomes.filter((O) => O.Slice === Slice);
      if (Outcomes.length === 0) continue;
      const List = Out[Slice] ?? [];
      List.push({ Name: Metric.Name, Value: MeanScore(Outcomes), Higher: Metric.Higher, N: Outcomes.length });
      Out[Slice] = List;
    }
  }
  return Out;
}
