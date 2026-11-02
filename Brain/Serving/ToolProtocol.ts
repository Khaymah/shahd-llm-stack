// Tool-calling protocol (Phase 6). The model signals a tool call by emitting a JSON object between
// sentinel tokens; the host parses it, runs the tool, and feeds the result back as another sentinel
// block. This is a MECHANISM that works at any model scale — the model's competence at CHOOSING the
// right tool/args is the scale-dependent part (CAPABILITIES.md).
//
// Two parsers, one JSON core: ParseToolCallIds is the SERVING path — it locates the sentinels by their
// special-token IDS, so text that merely spells "<|tool_call|>" with ordinary byte tokens (e.g. the
// model echoing a user message) is never a tool call. ParseToolCall works on a string and is only for
// trusted/authored text (scripted demos, test mocks, compaction of already-recorded turns).

import type { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";

export const ToolTokens = {
  CallStart: "<|tool_call|>",
  CallEnd: "<|end_tool_call|>",
  ResultStart: "<|tool_result|>",
  ResultEnd: "<|end_tool_result|>",
} as const;

export const ToolTokenList: readonly string[] = Object.values(ToolTokens);

export type ToolCall = { Name: string; Arguments: Record<string, unknown> };

// The shared JSON core. Unparseable JSON returns "Retry" so the caller can try a LATER CallEnd (the
// argument text itself may contain something resembling the sentinel); well-formed JSON without a
// string "name" is definitively not a call (null).
function ParseCallJson(Json: string): ToolCall | null | "Retry" {
  let Parsed: { name?: unknown; arguments?: unknown };
  try {
    Parsed = JSON.parse(Json.trim()) as { name?: unknown; arguments?: unknown };
  } catch {
    return "Retry";
  }
  if (typeof Parsed !== "object" || Parsed === null || typeof Parsed.name !== "string") return null;
  const Arguments = typeof Parsed.arguments === "object" && Parsed.arguments !== null && !Array.isArray(Parsed.arguments)
    ? (Parsed.arguments as Record<string, unknown>)
    : {};
  return { Name: Parsed.name, Arguments };
}

/** Parse the first tool call (JSON with a "name") from TRUSTED text, or null if none/invalid. If the
 *  JSON up to the first CallEnd fails to parse, retry against each LATER CallEnd occurrence in turn —
 *  only giving up once there are no further occurrences to try. Never use this on decoded model output
 *  at serving time: use ParseToolCallIds (a decoded string cannot tell a real sentinel from an echo). */
export function ParseToolCall(Text: string): ToolCall | null {
  const Start = Text.indexOf(ToolTokens.CallStart);
  if (Start === -1) return null;
  const From = Start + ToolTokens.CallStart.length;
  let SearchFrom = From;
  for (;;) {
    const End = Text.indexOf(ToolTokens.CallEnd, SearchFrom);
    const Call = ParseCallJson(Text.slice(From, End === -1 ? undefined : End));
    if (Call !== "Retry") return Call;
    if (End === -1) return null; // no further CallEnd occurrences left to retry against
    SearchFrom = End + ToolTokens.CallEnd.length;
  }
}

/** Ids-level tool-call parser (the serving path). Finds the first <|tool_call|> SPECIAL id, then decodes
 *  ONLY the span up to a <|end_tool_call|> special id (retrying later end ids exactly like the string
 *  parser). Byte tokens that happen to spell a sentinel are just argument/answer text here, so an echoed
 *  user string can never become a tool call. Null when the tokenizer has no tool sentinels at all. */
export function ParseToolCallIds(Ids: readonly number[], Tok: SpecialTokenizer): ToolCall | null {
  const StartId = Tok.TryId(ToolTokens.CallStart);
  if (StartId === null) return null;
  const Start = Ids.indexOf(StartId);
  if (Start === -1) return null;
  const EndId = Tok.TryId(ToolTokens.CallEnd);
  const From = Start + 1;
  let SearchFrom = From;
  for (;;) {
    const End = EndId === null ? -1 : Ids.indexOf(EndId, SearchFrom);
    const Call = ParseCallJson(Tok.Decode(Ids.slice(From, End === -1 ? undefined : End)));
    if (Call !== "Retry") return Call;
    if (End === -1) return null;
    SearchFrom = End + 1;
  }
}

/** Format a tool result block to append to the conversation. */
export function FormatToolResult(Result: unknown): string {
  return `${ToolTokens.ResultStart}${JSON.stringify(Result)}${ToolTokens.ResultEnd}`;
}
