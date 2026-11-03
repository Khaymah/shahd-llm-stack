// Tool-call accuracy: given a user request, does ONE agent step produce the right call? The step runs
// through the real agent loop (RunAgent, MaxSteps 1) so the call is parsed exactly as at serving time
// (ParseToolCallIds on special ids), against a FAKE registry with the full tool surface whose tools
// return canned results — the calculator computes, every other tool echoes its arguments — so a case
// is judged on the CALL, never on a tool's side effects (nothing touches fs, network or a subprocess).
//
//   tool.accuracy      — exactly one call parsed, Name matches, and every key of Expected.Arguments
//                        deep-equals the produced argument (extra produced keys are allowed);
//   tool.name_accuracy — exactly one call parsed and Name matches.
// Greedy (temperature 0).

import { isDeepStrictEqual } from "node:util";
import { RunAgent } from "../../Serving/AgentLoop.ts";
import type { Generator } from "../../Serving/AgentLoop.ts";
import type { ToolCall } from "../../Serving/ToolProtocol.ts";
import { ToolRegistry } from "../../Serving/Tools/ToolRegistry.ts";
import { BuildToolRegistry } from "../../Serving/Tools/BuildRegistry.ts";
import { CalculatorTool } from "../../Serving/Tools/MathTools.ts";
import type { ToolCase } from "./SuiteTypes.ts";
import type { CaseMetric, CaseOutcome } from "./SuiteMetrics.ts";
import { Excerpt } from "./SuiteMetrics.ts";
import { CapCases } from "./SuiteSettings.ts";
import { BuildSession, CaseRng, ErrorReason, ReadGenerated, SamplingAt, StartDeadline } from "./SuiteGeneration.ts";
import type { ReadTurn, SuiteContext } from "./SuiteGeneration.ts";

// One agent step: the first generation either is the call or is not.
const OneStep = 1;

/** The full tool surface (every gate open, so every tool NAME exists) with canned behavior: the pure
 *  calculator computes; every other tool echoes { tool, arguments } instead of executing. */
export function EvalToolRegistry(): ToolRegistry {
  const Surface = BuildToolRegistry({ FileAccess: "ReadWrite", ExecEnabled: true, WebSearchEnabled: true });
  const Registry = new ToolRegistry();
  for (const Tool of Surface.List()) {
    Registry.Register(Tool.Name === CalculatorTool.Name ? CalculatorTool : { ...Tool, Execute: (Arguments) => ({ tool: Tool.Name, arguments: Arguments }) });
  }
  return Registry;
}

export type ToolJudgement = { NameOk: boolean; ArgumentsOk: boolean; Reasons: string[] };

/** Judge one step's call against the expectation (see the header for the rules). */
export function JudgeToolCall(Expected: ToolCase["Expected"], Call: ToolCall | null, CallCount: number): ToolJudgement {
  if (Call === null || CallCount === 0) return { NameOk: false, ArgumentsOk: false, Reasons: ["no-tool-call"] };
  if (CallCount > 1) return { NameOk: false, ArgumentsOk: false, Reasons: ["multiple-tool-calls"] };
  if (Call.Name !== Expected.Name) return { NameOk: false, ArgumentsOk: false, Reasons: ["wrong-tool-name"] };
  const Wrong = Object.keys(Expected.Arguments).filter((Key) => !isDeepStrictEqual(Call.Arguments[Key], Expected.Arguments[Key]));
  return { NameOk: true, ArgumentsOk: Wrong.length === 0, Reasons: Wrong.length === 0 ? [] : ["wrong-arguments"] };
}

/** tool.accuracy and tool.name_accuracy over the (capped) tool cases. */
export async function ToolCallMetrics(Ctx: SuiteContext, Cases: readonly ToolCase[]): Promise<CaseMetric[]> {
  const Registry = EvalToolRegistry();
  const Full: CaseOutcome[] = [];
  const NameOnly: CaseOutcome[] = [];
  const Selected = CapCases(Cases, Ctx.Settings.MaxCases);
  for (const Case of Selected) {
    const Deadline = StartDeadline(Ctx.Settings.CaseTimeoutMs);
    // Filled from inside the generator / step callbacks (a holder, so closure writes stay typed).
    const Seen: { Raw: readonly number[] | null; Turn: ReadTurn | null; Call: ToolCall | null } = { Raw: null, Turn: null, Call: null };
    const Inner = Ctx.Factory({
      MaxTokens: Ctx.Settings.ToolMaxTokens, Sampling: SamplingAt(0), Rng: CaseRng(Ctx.Settings.Seed, Cases.indexOf(Case), 0), ShouldStop: Deadline.ShouldStop,
      OnPassEnd: (Ids) => { Seen.Raw = Ids; },
    });
    // Read the generated turn on the way through, for the call COUNT (the loop parses only the first).
    const Capture: Generator = async (Session) => {
      const Output = await Inner(Session);
      Seen.Turn = ReadGenerated(Output, Seen.Raw, Ctx.Runnable.Tokenizer);
      return Output;
    };
    let Judgement: ToolJudgement;
    try {
      await RunAgent(BuildSession([{ Role: "User", Content: Case.User }]), Capture, Registry, OneStep, undefined, (Step) => { Seen.Call = Step.Call; });
      Judgement = JudgeToolCall(Case.Expected, Seen.Call, Seen.Turn?.ToolCallCount ?? 0);
    } catch (Caught) {
      Judgement = { NameOk: false, ArgumentsOk: false, Reasons: [ErrorReason(Caught)] };
    }
    const Shown = Seen.Call !== null ? JSON.stringify(Seen.Call) : Seen.Turn?.Answer ?? "";
    const Base = { Id: Case.Id, Slice: Case.Slice, Excerpt: Excerpt(Shown) };
    Full.push({ ...Base, Score: Judgement.NameOk && Judgement.ArgumentsOk ? 1 : 0, Reasons: Judgement.Reasons });
    NameOnly.push({ ...Base, Score: Judgement.NameOk ? 1 : 0, Reasons: Judgement.NameOk ? [] : Judgement.Reasons });
  }
  Ctx.Log(`[eval] tools: ${Full.filter((O) => O.Score === 1).length}/${Full.length} exact, ${NameOnly.filter((O) => O.Score === 1).length}/${NameOnly.length} name`);
  return [
    { Name: "tool.accuracy", Higher: "better", Outcomes: Full },
    { Name: "tool.name_accuracy", Higher: "better", Outcomes: NameOnly },
  ];
}
