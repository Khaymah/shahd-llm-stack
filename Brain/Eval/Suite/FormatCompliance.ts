// Format compliance: is the reply a well-formed assistant turn? Judged on format only, never content.
// A case passes when ALL hold:
//   (a) the reply is non-empty (a visible answer, or a genuine tool call);
//   (b) the turn ended at the <|endofturn|> id within the token budget — not cut by ReplyMaxTokens
//       (or the case deadline), and not "ended" by the model starting another role's turn;
//   (c) no control-token text (chat or tool sentinels) appears in the VISIBLE answer — spelled in
//       ordinary tokens or as a stray special id, the user would see it either way;
//   (d) every <|think|> span is closed by <|endthink|>.
// Replies are greedy (temperature 0) so the number is reproducible.

import type { FormatCase } from "./SuiteTypes.ts";
import type { CaseMetric, CaseOutcome } from "./SuiteMetrics.ts";
import { Excerpt } from "./SuiteMetrics.ts";
import { CapCases } from "./SuiteSettings.ts";
import { CaseRng, ChatReply, ErrorReason, SamplingAt, StartDeadline } from "./SuiteGeneration.ts";
import type { ReadTurn, SuiteContext } from "./SuiteGeneration.ts";

/** Every format rule the turn breaks (empty = compliant). */
export function FormatFailures(Turn: ReadTurn): string[] {
  const Reasons: string[] = [];
  if (Turn.Answer === "" && Turn.ToolCallCount === 0) Reasons.push("empty-reply");
  if (Turn.End === "no-end-of-turn") Reasons.push("no-end-of-turn");
  if (Turn.End === "role-marker") Reasons.push("ended-by-role-marker");
  if (Turn.LeakedLiterals.length > 0) Reasons.push("leaked-control-token");
  if (!Turn.ThinkingClosed) Reasons.push("unclosed-thinking");
  return Reasons;
}

/** format.compliance over the (capped) format cases. */
export async function FormatComplianceMetric(Ctx: SuiteContext, Cases: readonly FormatCase[]): Promise<CaseMetric> {
  const Outcomes: CaseOutcome[] = [];
  const Selected = CapCases(Cases, Ctx.Settings.MaxCases);
  for (const Case of Selected) {
    const Deadline = StartDeadline(Ctx.Settings.CaseTimeoutMs);
    const Settings = { MaxTokens: Ctx.Settings.ReplyMaxTokens, Sampling: SamplingAt(0), Rng: CaseRng(Ctx.Settings.Seed, Cases.indexOf(Case), 0), ShouldStop: Deadline.ShouldStop };
    try {
      const Turn = await ChatReply(Ctx.Factory, Ctx.Runnable.Tokenizer, Case.Messages, Settings);
      const Reasons = FormatFailures(Turn);
      Outcomes.push({ Id: Case.Id, Slice: Case.Slice, Score: Reasons.length === 0 ? 1 : 0, Reasons, Excerpt: Excerpt(Turn.Answer) });
    } catch (Caught) {
      Outcomes.push({ Id: Case.Id, Slice: Case.Slice, Score: 0, Reasons: [ErrorReason(Caught)] });
    }
  }
  Ctx.Log(`[eval] format: ${Outcomes.filter((O) => O.Score === 1).length}/${Outcomes.length} compliant`);
  return { Name: "format.compliance", Higher: "better", Outcomes };
}
