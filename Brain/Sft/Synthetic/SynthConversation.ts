// Building blocks every generator shares when it turns teacher output into a stored conversation: the
// document envelope (OwnedSystemPrompt first — the ONE prompt serving presents — and provenance Meta),
// assistant turns that always think first (the unified recipe of OwnedSftData), and the per-slice names
// used in prompts and code fences.

import type { ChatDocument, ChatDocumentMessage, ChatDocumentToolCall } from "../ChatDocument.ts";
import { ChatDocumentFormat } from "../ChatDocument.ts";
import { OwnedSystemPrompt } from "../OwnedSftData.ts";
import type { EvalSlice } from "../../Eval/Suite/SuiteTypes.ts";
import type { CodeSlice, SynthGenerator } from "./SynthTypes.ts";

/** A scratchpad longer than this is not "short thinking" — the tiny model should learn brief plans. */
export const MaxThinkingChars = 400;

/** Human language names per slice (prompts, user messages). */
export const SliceLanguageName: Record<CodeSlice, string> = { ts: "TypeScript", js: "JavaScript", html: "HTML", css: "CSS", go: "Go", sql: "SQL" };

/** Markdown fence tags per slice. */
export const SliceFenceTag: Record<CodeSlice, string> = { ts: "ts", js: "js", html: "html", css: "css", go: "go", sql: "sql" };

export type SynthEnvelope = {
  Generator: SynthGenerator;
  Teacher: string;
  SourceId: string;
  Slice: EvalSlice;
  Task?: string;
  ClockMs?: number;
};

/** The stored document: OwnedSystemPrompt, then the turns, with provenance Meta. */
export function BuildSynthDocument(Envelope: SynthEnvelope, Turns: readonly ChatDocumentMessage[]): ChatDocument {
  const Meta: ChatDocument["Meta"] = { Generator: Envelope.Generator, Teacher: Envelope.Teacher, SourceId: Envelope.SourceId, Slice: Envelope.Slice };
  if (Envelope.Task !== undefined) Meta.Task = Envelope.Task;
  if (Envelope.ClockMs !== undefined) Meta.ClockMs = Envelope.ClockMs;
  return { Format: ChatDocumentFormat, Messages: [{ Role: "System", Content: OwnedSystemPrompt }, ...Turns], Meta };
}

/** An answering assistant turn: short thinking, then the visible answer. */
export function AnswerTurn(Thinking: string, Answer: string): ChatDocumentMessage {
  return { Role: "Assistant", Thinking: Thinking.trim(), Content: Answer.trim() };
}

/** A tool-calling assistant turn: short thinking, then only the call. */
export function CallTurn(Thinking: string, Call: ChatDocumentToolCall): ChatDocumentMessage {
  return { Role: "Assistant", Thinking: Thinking.trim(), Content: "", ToolCall: Call };
}

/** Null when every scratchpad is short enough, else the rejection detail. */
export function ThinkingTooLong(...Thoughts: readonly string[]): string | null {
  const Long = Thoughts.find((Thought) => Thought.trim().length > MaxThinkingChars);
  return Long === undefined ? null : `${Long.trim().length} > ${MaxThinkingChars} chars`;
}

// Leading blank lines (never the first code line's indentation — it is part of the code).
const LeadingBlankLines = /^(?:[ \t]*\r?\n)+/;

/** Wrap code in a fenced block for the slice (surrounding blank lines dropped, indentation kept). */
export function Fenced(Slice: CodeSlice, Code: string): string {
  return `\`\`\`${SliceFenceTag[Slice]}\n${Code.replace(LeadingBlankLines, "").trimEnd()}\n\`\`\``;
}
