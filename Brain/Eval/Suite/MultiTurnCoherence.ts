// Multi-turn coherence: does the reply use what an EARLIER turn established? The case's turns are fed
// as history (assistant turns base-encoded, as a replayed history is at serving time), the final reply
// is generated greedily, and the case passes when every MustContain substring is present and no
// MustNotContain substring is (both case-insensitive, judged on the VISIBLE answer only — thinking
// is never the reply).

import type { MultiTurnCase } from "./SuiteTypes.ts";
import type { CaseMetric, CaseOutcome } from "./SuiteMetrics.ts";
import { Excerpt } from "./SuiteMetrics.ts";
import { CapCases } from "./SuiteSettings.ts";
import { CaseRng, ChatReply, ErrorReason, SamplingAt, StartDeadline } from "./SuiteGeneration.ts";
import type { SuiteContext } from "./SuiteGeneration.ts";

/** The coherence failures of Reply for a case (empty = pass). */
export function CoherenceFailures(Case: MultiTurnCase, Reply: string): string[] {
  const Lower = Reply.toLowerCase();
  const Reasons: string[] = [];
  if ((Case.MustContain ?? []).some((Needle) => !Lower.includes(Needle.toLowerCase()))) Reasons.push("missing-expected");
  if ((Case.MustNotContain ?? []).some((Needle) => Lower.includes(Needle.toLowerCase()))) Reasons.push("contains-forbidden");
  return Reasons;
}

/** multiturn.coherence over the (capped) multi-turn cases. */
export async function MultiTurnCoherenceMetric(Ctx: SuiteContext, Cases: readonly MultiTurnCase[]): Promise<CaseMetric> {
  const Outcomes: CaseOutcome[] = [];
  const Selected = CapCases(Cases, Ctx.Settings.MaxCases);
  for (const Case of Selected) {
    const Deadline = StartDeadline(Ctx.Settings.CaseTimeoutMs);
    const Settings = { MaxTokens: Ctx.Settings.ReplyMaxTokens, Sampling: SamplingAt(0), Rng: CaseRng(Ctx.Settings.Seed, Cases.indexOf(Case), 0), ShouldStop: Deadline.ShouldStop };
    try {
      const Turn = await ChatReply(Ctx.Factory, Ctx.Runnable.Tokenizer, Case.Turns, Settings);
      const Reasons = CoherenceFailures(Case, Turn.Answer);
      Outcomes.push({ Id: Case.Id, Slice: Case.Slice, Score: Reasons.length === 0 ? 1 : 0, Reasons, Excerpt: Excerpt(Turn.Answer) });
    } catch (Caught) {
      Outcomes.push({ Id: Case.Id, Slice: Case.Slice, Score: 0, Reasons: [ErrorReason(Caught)] });
    }
  }
  Ctx.Log(`[eval] multi-turn: ${Outcomes.filter((O) => O.Score === 1).length}/${Outcomes.length} coherent`);
  return { Name: "multiturn.coherence", Higher: "better", Outcomes };
}
