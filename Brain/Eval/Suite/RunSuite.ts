// The suite runner: one checkpoint + one frozen case set -> one SuiteReport. Order: held-out BPB,
// code execution, then the chat-only metrics (format, tools, multi-turn), then the memorization probe.
//
// BASE (non-chat) models run only what a plain language model can be asked: held-out BPB, the
// memorization probe, and code pass@k through plain completion (the prompt as a code comment). Format
// compliance, tool-call accuracy and multi-turn coherence are SKIPPED for them (reported with N = 0 and
// the reason): all three are defined over the chat template — end-of-turn ids, tool-call sentinels,
// role-delimited history — which a base model was never trained to produce, so scoring it would only
// measure "not a chat model", a fact already known from its Meta.Format.

import type { RunnableModel } from "../../Checkpoint/LoadRunnableModel.ts";
import type { EvalCaseSet, MetricResult, SuiteReport } from "./SuiteTypes.ts";
import { AggregateMetric, SkippedMetric, SliceBreakdown } from "./SuiteMetrics.ts";
import type { CaseMetric } from "./SuiteMetrics.ts";
import { CapCases, ResolveSuiteSettings } from "./SuiteSettings.ts";
import type { SuiteSettings } from "./SuiteSettings.ts";
import { ModelGeneratorFactory } from "./SuiteGeneration.ts";
import type { GeneratorFactory, SuiteContext } from "./SuiteGeneration.ts";
import { LoadHeldOutFromSeed } from "./SuiteCorpora.ts";
import type { HeldOutSet, TrainCorpus } from "./SuiteCorpora.ts";
import { HeldOutBpbMetrics } from "./HeldOutBpb.ts";
import { CodeExecutionMetrics } from "./CodeExecution.ts";
import { FormatComplianceMetric } from "./FormatCompliance.ts";
import { ToolCallMetrics } from "./ToolCallAccuracy.ts";
import { MultiTurnCoherenceMetric } from "./MultiTurnCoherence.ts";
import { MemorizationMetrics } from "./MemorizationProbe.ts";

export type SuiteRunOptions = {
  Checkpoint: string; // name or path, recorded in the report
  Settings?: Partial<SuiteSettings>; // overrides of SuiteDefaults (validated)
  HeldOut?: HeldOutSet; // default: the built-in seed corpus (last resort — see SuiteCorpora.ts)
  TrainCorpus?: TrainCorpus | null; // the memorization probe is skipped without it
  Generators?: GeneratorFactory; // default: the guarded serving generator (tests inject stubs)
  Log?: (Line: string) => void;
  Now?: () => number; // clock (ms) for RanAt/DurationMs; injectable for tests
};

const ChatOnlySkipReason = "base (non-chat) model: format, tool and multi-turn metrics are defined over the chat template, which a base model was never trained to produce";
const ChatOnlyMetrics: readonly string[] = ["format.compliance", "tool.accuracy", "tool.name_accuracy", "multiturn.coherence"];

function ParamCount(Runnable: RunnableModel): number {
  return Runnable.Model.Parameters().reduce((Sum, P) => Sum + P.Size, 0);
}

/** Run the whole suite on a loaded model. */
export async function RunSuite(Runnable: RunnableModel, CaseSet: EvalCaseSet, Options: SuiteRunOptions): Promise<SuiteReport> {
  const Now = Options.Now ?? Date.now;
  const Started = Now();
  const Settings = ResolveSuiteSettings(Options.Settings);
  const Log = Options.Log ?? ((): void => undefined);
  const Ctx: SuiteContext = { Runnable, Factory: Options.Generators ?? ModelGeneratorFactory(Runnable), Settings, Log };
  const HeldOut = Options.HeldOut ?? LoadHeldOutFromSeed();

  const Metrics: MetricResult[] = [];
  const CaseMetrics: CaseMetric[] = [];
  const Push = (List: readonly CaseMetric[], Extra: Record<string, unknown> = {}): void => {
    for (const M of List) {
      CaseMetrics.push(M);
      Metrics.push(AggregateMetric(M, Extra));
    }
  };

  Log(`[eval] held-out BPB (${HeldOut.Origin}: ${HeldOut.Docs.length} docs)`);
  Metrics.push(...HeldOutBpbMetrics(Runnable, HeldOut, Settings.BpbMaxWindows));

  const Mode = Runnable.Chat ? "chat reply" : "plain completion (base model)";
  Log(`[eval] code execution: ${CapCases(CaseSet.Code, Settings.MaxCases).length}/${CaseSet.Code.length} cases, ${Mode}`);
  const Code = await CodeExecutionMetrics(Ctx, CaseSet.Code);
  Push(Code.Metrics, { Mode, ...(Code.InvalidCases.length > 0 ? { InvalidCases: Code.InvalidCases } : {}) });

  if (Runnable.Chat) {
    Push([await FormatComplianceMetric(Ctx, CaseSet.Format)]);
    Push(await ToolCallMetrics(Ctx, CaseSet.Tools));
    Push([await MultiTurnCoherenceMetric(Ctx, CaseSet.MultiTurn)]);
  } else {
    for (const Name of ChatOnlyMetrics) Metrics.push(SkippedMetric(Name, "better", ChatOnlySkipReason));
  }

  Log(`[eval] memorization probe`);
  Metrics.push(...(await MemorizationMetrics(Runnable, HeldOut, Options.TrainCorpus ?? null, Settings)));

  return {
    SuiteVersion: CaseSet.Version,
    Checkpoint: Options.Checkpoint,
    ConfigHash: Runnable.Config.ConfigHash,
    Params: ParamCount(Runnable),
    RanAt: new Date(Started).toISOString(),
    DurationMs: Now() - Started,
    Metrics,
    Slices: SliceBreakdown(CaseMetrics),
  };
}
