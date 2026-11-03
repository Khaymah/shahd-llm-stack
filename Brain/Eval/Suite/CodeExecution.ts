// Code correctness by EXECUTION: pass@1 (one greedy answer, temperature 0) and pass@k (k samples at
// SampleTemperature, seeded per case + sample), per case, overall and per slice. A case with Tests runs
// the answer through the nonce-verified harness (EvalHarness.RunAgainstTestsAsync — exiting early or
// patching the runner cannot pass); a case with StaticChecks (html/css/sql/go without a toolchain)
// passes when EVERY regex matches the answer.
//
// StaticChecks are compiled with StaticCheckFlags (Data/CodeCaseBuilder.ts — case-insensitive, `.`
// spans newlines), the one flag set the case authors and the case-set self-test also use.
//
// Chat models answer the case Prompt as a user turn through the serving generator; base models get a
// plain completion of the prompt written as a code comment in the slice's language. The answer's code
// is the first ``` fenced block (an unclosed fence runs to the end), else the whole answer.
//
// Time is bounded twice: every case has a wall-clock budget (CaseTimeoutMs) that stops generation and
// shortens test runs — work past it counts as a fail — and --MaxCases caps how many cases run.

import { RunAgainstTestsAsync } from "../EvalHarness.ts";
import { DefaultExecTimeoutMs } from "../CodeExecutor.ts";
import { PassAtK } from "../PassAtK.ts";
import { StaticCheckFlags } from "./Data/CodeCaseBuilder.ts";
import type { CodeCase, EvalSlice } from "./SuiteTypes.ts";
import type { CaseMetric, CaseOutcome } from "./SuiteMetrics.ts";
import { Excerpt } from "./SuiteMetrics.ts";
import { CapCases } from "./SuiteSettings.ts";
import { CaseRng, ChatReply, CompletePlain, ErrorReason, SamplingAt, StartDeadline } from "./SuiteGeneration.ts";
import type { CaseDeadline, SuiteContext } from "./SuiteGeneration.ts";

// Below this much remaining budget a test run is not started (it could not finish meaningfully).
const MinExecBudgetMs = 250;

const FencedBlock = /```[\w+#.-]*[^\S\n]*\n([\s\S]*?)```/;
const OpenFence = /```[\w+#.-]*[^\S\n]*\n([\s\S]*)$/;

/** The code inside the first ``` fence (an unclosed fence runs to the end), else the whole reply. */
export function ExtractCode(Reply: string): string {
  const Closed = FencedBlock.exec(Reply);
  if (Closed !== null) return Closed[1];
  const Open = OpenFence.exec(Reply);
  return Open !== null ? Open[1] : Reply;
}

/** Compile one StaticChecks entry with the shared flags. Throws on an invalid pattern. */
export function CompileStaticCheck(Pattern: string): RegExp {
  return new RegExp(Pattern, StaticCheckFlags);
}

// The comment syntax a base-model prompt is written in, per slice.
const CommentFor: Record<EvalSlice, (Text: string) => string> = {
  ts: (Text) => `// ${Text}\n`,
  js: (Text) => `// ${Text}\n`,
  go: (Text) => `// ${Text}\n`,
  general: (Text) => `// ${Text}\n`,
  css: (Text) => `/* ${Text} */\n`,
  html: (Text) => `<!-- ${Text} -->\n`,
  sql: (Text) => `-- ${Text}\n`,
};

/** Base-model prompt: the instruction as a comment in the slice's language (exported for tests). */
export function BaseCodePrompt(Case: CodeCase): string {
  return CommentFor[Case.Slice](Case.Prompt.replace(/\s*\n\s*/g, " "));
}

type CaseCheck = { Kind: "Tests"; Tests: string } | { Kind: "Static"; Patterns: RegExp[] };

function CheckFor(Case: CodeCase): CaseCheck {
  if (Case.Tests !== undefined && Case.Tests.trim() !== "") return { Kind: "Tests", Tests: Case.Tests };
  if (Case.StaticChecks !== undefined && Case.StaticChecks.length > 0) return { Kind: "Static", Patterns: Case.StaticChecks.map(CompileStaticCheck) };
  throw new Error(`code case ${Case.Id} has neither Tests nor StaticChecks`);
}

/** Does Code pass the case's check? Tests run with at most TimeoutMs. */
async function Verify(Case: CodeCase, Check: CaseCheck, Code: string, TimeoutMs: number): Promise<boolean> {
  if (Code.trim() === "") return false;
  if (Check.Kind === "Static") return Check.Patterns.every((Pattern) => Pattern.test(Code));
  return RunAgainstTestsAsync({ Name: Case.Id, Tests: Check.Tests }, Code, TimeoutMs);
}

/** Self-check a case: its Reference must pass its own check (a case that fails this is broken). */
export async function SelfCheckCodeCase(Case: CodeCase): Promise<boolean> {
  return Verify(Case, CheckFor(Case), Case.Reference, DefaultExecTimeoutMs);
}

type AttemptResult = { Passed: boolean; Reason: string | null; Answer: string };

async function RunAttempt(Ctx: SuiteContext, Case: CodeCase, Check: CaseCheck, Deadline: CaseDeadline, Temperature: number, CaseIndex: number, Sample: number): Promise<AttemptResult> {
  if (Deadline.Expired()) return { Passed: false, Reason: "case-timeout", Answer: "" };
  const Settings = { MaxTokens: Ctx.Settings.CodeMaxTokens, Sampling: SamplingAt(Temperature), Rng: CaseRng(Ctx.Settings.Seed, CaseIndex, Sample), ShouldStop: Deadline.ShouldStop };
  let Answer: string;
  try {
    Answer = Ctx.Runnable.Chat
      ? (await ChatReply(Ctx.Factory, Ctx.Runnable.Tokenizer, [{ Role: "User", Content: Case.Prompt }], Settings)).Answer
      : await CompletePlain(Ctx.Runnable, BaseCodePrompt(Case), Settings);
  } catch (Caught) {
    return { Passed: false, Reason: ErrorReason(Caught), Answer: "" };
  }
  const Code = ExtractCode(Answer);
  if (Code.trim() === "") return { Passed: false, Reason: "empty-answer", Answer };
  const Budget = Math.min(DefaultExecTimeoutMs, Deadline.RemainingMs());
  if (Check.Kind === "Tests" && Budget < MinExecBudgetMs) return { Passed: false, Reason: "case-timeout", Answer };
  const Passed = await Verify(Case, Check, Code, Budget);
  return { Passed, Reason: Passed ? null : Check.Kind === "Tests" ? "tests-failed" : "static-check-failed", Answer };
}

export type CodeExecutionResult = { Metrics: CaseMetric[]; InvalidCases: { Id: string; Error: string }[] };

/** code.pass@1 and code.pass@k over the (capped) code cases. */
export async function CodeExecutionMetrics(Ctx: SuiteContext, Cases: readonly CodeCase[]): Promise<CodeExecutionResult> {
  const K = Ctx.Settings.PassAtKSamples;
  const PassAt1: CaseOutcome[] = [];
  const PassAtKOutcomes: CaseOutcome[] = [];
  const InvalidCases: { Id: string; Error: string }[] = [];
  const Selected = CapCases(Cases, Ctx.Settings.MaxCases);
  for (const Case of Selected) {
    let Check: CaseCheck;
    try {
      Check = CheckFor(Case);
    } catch (Caught) {
      InvalidCases.push({ Id: Case.Id, Error: (Caught as Error).message });
      continue;
    }
    const Deadline = StartDeadline(Ctx.Settings.CaseTimeoutMs);
    const Greedy = await RunAttempt(Ctx, Case, Check, Deadline, 0, Cases.indexOf(Case), 0);
    let Correct = 0;
    const SampleReasons = new Set<string>();
    for (let S = 1; S <= K; S++) {
      const Sampled = await RunAttempt(Ctx, Case, Check, Deadline, Ctx.Settings.SampleTemperature, Cases.indexOf(Case), S);
      if (Sampled.Passed) Correct++;
      else if (Sampled.Reason !== null) SampleReasons.add(Sampled.Reason);
    }
    const Base = { Id: Case.Id, Slice: Case.Slice, Excerpt: Excerpt(Greedy.Answer) };
    PassAt1.push({ ...Base, Score: Greedy.Passed ? 1 : 0, Reasons: Greedy.Reason === null ? [] : [Greedy.Reason] });
    const Score = PassAtK(K, Correct, K);
    PassAtKOutcomes.push({ ...Base, Score, Reasons: Score < 1 ? [...SampleReasons].sort() : [] });
    Ctx.Log(`[eval] code ${Case.Id}: greedy ${Greedy.Passed ? "pass" : `fail (${Greedy.Reason ?? "?"})`}, ${Correct}/${K} samples`);
  }
  return {
    Metrics: [
      { Name: "code.pass@1", Higher: "better", Outcomes: PassAt1 },
      { Name: `code.pass@${K}`, Higher: "better", Outcomes: PassAtKOutcomes },
    ],
    InvalidCases,
  };
}
