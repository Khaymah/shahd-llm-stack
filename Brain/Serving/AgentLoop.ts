// Multi-step agent loop (Phase 6, extended in Phase 7): generate -> if the model emitted a tool
// call, run it against the injected ToolContext and feed the result back -> repeat, until the model
// produces a plain (non-tool) answer, calls the Terminal `finish` tool, or the step budget is hit.
// Async throughout so tools that await (user_ask, web_search) and async generators compose. The
// step budget + never-throwing registry are the guardrails; `Generate` is injected (GuardedGenerate
// at serving time, a mock in tests) so the loop stays model-agnostic.

import type { ChatSession } from "./ChatSession.ts";
import type { ToolRegistry } from "./Tools/ToolRegistry.ts";
import type { ToolContext } from "./Tools/ToolTypes.ts";
import type { ToolCall } from "./ToolProtocol.ts";
import { ParseToolCall, ParseToolCallIds, FormatToolResult } from "./ToolProtocol.ts";
import { SplitThinking } from "../Reasoning/ThinkingMode.ts";
import type { SplitThought } from "../Reasoning/ThinkingMode.ts";
import type { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";
import { TruncateAtTurnEnd, SplitThinkingIds } from "../Sft/ChatTemplate.ts";

// Host-enforced backstop, independent of the model ever choosing to call the `compact` tool: once
// the running session grows past this many messages, RunAgent compacts it unconditionally so a long
// or looping conversation can never grow the context/prompt without bound.
const HardMessageCap = 40;
const HardMessageKeep = 20;

/** Compact Session if it has grown past the hard cap, regardless of whether the model requested it. */
function EnforceHardCap(Session: ChatSession, Context?: ToolContext): void {
  if (Session.Messages.length > HardMessageCap) Session.Compact(HardMessageKeep, Context?.Summarize);
}

// The generator is handed the live ChatSession (not a pre-rendered string) so it renders the prompt
// however its tokenizer requires: a special-token (chat) model MUST render to ids via
// Session.RenderPromptIds (base-encoding untrusted content so it can't smuggle control tokens); tests
// can inspect Session.RenderPrompt()/Messages directly. This keeps the encode step where the tokenizer
// lives, and closes the control-token smuggling that a re-encoded rendered string allowed.
//
// A REAL model generator returns a GeneratedTurn: the newly generated ids (the completion only, NOT the
// prompt) plus the tokenizer that produced them. Every control decision — end of turn, thinking spans,
// tool calls — is then made on SPECIAL IDS, and the turn is stored with its ids, so a model that echoes
// user text like `<|tool_call|>{...}` as ordinary byte tokens neither triggers a tool nor has that text
// promoted into real control tokens next turn. A plain string return is the TRUSTED path for scripted
// demos and test mocks only (its sentinels are parsed from text, and it is stored without ids, so it is
// base-encoded on the next render).
export type GeneratedTurn = { Ids: readonly number[]; Tok: SpecialTokenizer };
export type Generator = (Session: ChatSession) => string | GeneratedTurn | Promise<string | GeneratedTurn>;

// One observable step of the agent's reasoning — what it generated, whether that was a tool call,
// the tool's result, and whether the step ended the loop. This is the raw material of a thinking
// trace: it exposes exactly what the model did at each step so the process can be watched + improved.
export type AgentStep = {
  Index: number;
  Generated: string; // the raw text the model produced this step (decoded, up to its end-of-turn)
  Call: ToolCall | null; // the parsed tool call, if the step was one
  Result: unknown | null; // the tool's result, if a call ran
  Terminal: boolean; // did this step end the loop (plain answer or a successful finish)
  // Thinking/answer split as the loop decided it (by special ids for a GeneratedTurn). Optional so
  // hand-built steps still work: consumers fall back to SplitThinking(Generated).
  Split?: SplitThought;
};

export type AgentResult = {
  FinalText: string;
  Steps: number;
  ToolCalls: ToolCall[];
  HitStepLimit: boolean;
};

// One generation, normalized: the decoded turn text, its thinking/answer split, the tool call (if any)
// and — for a real model turn — the ids to store for the whole turn and for the visible answer.
type ParsedTurn = { Text: string; Split: SplitThought; Call: ToolCall | null; TurnIds?: number[]; AnswerIds?: number[] };

function ParseTurn(Output: string | GeneratedTurn): ParsedTurn {
  if (typeof Output === "string") return { Text: Output, Split: SplitThinking(Output), Call: ParseToolCall(Output) };
  const { Tok } = Output;
  const TurnIds = TruncateAtTurnEnd(Output.Ids, Tok);
  const Thinking = SplitThinkingIds(TurnIds, Tok);
  const Split: SplitThought = {
    Thinking: Thinking.ThinkingIds.map((Span) => Tok.Decode(Span)).join("\n\n").trim(),
    Answer: Tok.Decode(Thinking.AnswerIds).trim(),
    HadThinking: Thinking.HadThinking,
  };
  return { Text: Tok.Decode(TurnIds), Split, Call: ParseToolCallIds(TurnIds, Tok), TurnIds, AnswerIds: Thinking.AnswerIds };
}

export async function RunAgent(
  Session: ChatSession,
  Generate: Generator,
  Registry: ToolRegistry,
  MaxSteps = 6,
  Context?: ToolContext,
  OnStep?: (Step: AgentStep) => void, // fired once per loop iteration with the full step detail (trace hook)
): Promise<AgentResult> {
  const ToolCalls: ToolCall[] = [];
  for (let Step = 0; Step < MaxSteps; Step++) {
    const Turn = ParseTurn(await Generate(Session));
    const { Call, Split } = Turn;
    const Generated = Turn.Text;
    if (Call === null) {
      // Plain text is the canonical final answer. Strip the private <|think|> scratchpad before it
      // reaches the user or the conversation history — the raw text stays in the trace step (below) so
      // the reasoning is still observable, but it must never leak into the visible reply or context.
      const Answer = Split.Answer;
      Session.AddAssistant(Answer, Turn.AnswerIds);
      EnforceHardCap(Session, Context);
      OnStep?.({ Index: Step, Generated, Call: null, Result: null, Terminal: true, Split });
      return { FinalText: Answer, Steps: Step + 1, ToolCalls, HitStepLimit: false };
    }
    ToolCalls.push(Call);
    Session.AddAssistant(Generated, Turn.TurnIds); // record the tool-call turn (with its exact ids)
    EnforceHardCap(Session, Context);
    const Result = await Registry.Run(Call, Context); // never throws
    Session.AddToolResult(FormatToolResult(Result));
    EnforceHardCap(Session, Context);
    // The one other terminal path: a successful `finish` (Terminal) tool call ends the loop.
    const Tool = Registry.Get(Call.Name);
    const Terminal = Tool?.Terminal === true && !("error" in Result);
    OnStep?.({ Index: Step, Generated, Call, Result, Terminal, Split });
    if (Terminal) {
      const Answer = typeof Result["answer"] === "string" ? Result["answer"] : Split.Answer;
      return { FinalText: Answer, Steps: Step + 1, ToolCalls, HitStepLimit: false };
    }
  }
  return { FinalText: "(step budget exhausted)", Steps: MaxSteps, ToolCalls, HitStepLimit: true };
}
