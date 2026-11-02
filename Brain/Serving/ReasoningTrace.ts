// Make the model's reasoning observable (Phase 8). RunAgent emits one AgentStep per iteration; this
// turns those raw steps into an ordered, human-readable trace of WHAT THE MODEL ACTUALLY DID: its
// hidden thinking, each tool call + result, and the final answer. This is the development lens —
// watch how the model reasons, see where it is weak, and target training/tools accordingly. Pure
// (no I/O): the dashboard/console/tests all render the same trace.

import type { AgentStep } from "./AgentLoop.ts";
import { SplitThinking } from "../Reasoning/ThinkingMode.ts";

export type TraceKind = "think" | "tool" | "answer";
export type TraceLine = { Step: number; Kind: TraceKind; Text: string; Detail?: string };

/** Flatten agent steps into ordered trace lines: a thinking line when the model reasoned, a tool line
 *  (name + args -> result) for each call, and an answer line for the final plain reply. */
export function BuildTrace(Steps: AgentStep[]): TraceLine[] {
  const Lines: TraceLine[] = [];
  for (const Step of Steps) {
    // Prefer the loop's own split (made on special ids for a real model turn, so echoed "<|think|>" text
    // is not mistaken for reasoning); hand-built steps fall back to the string split.
    const Split = Step.Split ?? SplitThinking(Step.Generated);
    if (Split.HadThinking && Split.Thinking.length > 0) {
      Lines.push({ Step: Step.Index, Kind: "think", Text: Split.Thinking });
    }
    if (Step.Call !== null) {
      Lines.push({
        Step: Step.Index,
        Kind: "tool",
        Text: `${Step.Call.Name}(${JSON.stringify(Step.Call.Arguments)})`,
        Detail: JSON.stringify(Step.Result),
      });
    } else if (Step.Terminal) {
      const Answer = Split.Answer.length > 0 ? Split.Answer : Step.Generated.trim();
      if (Answer.length > 0) Lines.push({ Step: Step.Index, Kind: "answer", Text: Answer });
    }
  }
  return Lines;
}

const Glyph: Record<TraceKind, string> = { think: "🧠 think ", tool: "🛠 tool  ", answer: "💬 answer" };

/** Render the trace as a console/log-friendly block (one line per reasoning step). */
export function FormatTrace(Steps: AgentStep[]): string {
  return BuildTrace(Steps)
    .map((Line) => {
      const Body = Line.Text.replace(/\s+/g, " ").trim();
      const Arrow = Line.Detail !== undefined ? ` -> ${Line.Detail}` : "";
      return `  [${Line.Step}] ${Glyph[Line.Kind]}: ${Body}${Arrow}`;
    })
    .join("\n");
}
