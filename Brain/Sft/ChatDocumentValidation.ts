// Validation of a shahd-chat-v1 document (ChatDocument.ts) before it may become training data. Two
// layers, so a cheap structural check can run anywhere (training reads, imports) and the expensive one
// only where tools can actually run:
//
//   ValidateChatStructure (sync)  — roles in a valid order (optional System first; User -> Assistant;
//     a tool-call turn -> its Tool result -> Assistant; ends on an answering Assistant turn), at least
//     one assistant turn, no reserved control-token literal in ANY text (system, user, thinking,
//     answers, tool arguments and results — the renderer is the only producer of control tokens),
//     non-empty answers and scratchpads, every <|think|> closed, every call naming a registered tool.
//   ReplayToolCalls (async)       — re-executes every call, in order, through the tool's OWN Execute
//     (its own argument parsing) against a fresh deterministic context: a call whose tool returns an
//     { error } is "bad-arguments"; a recorded result that differs from the replayed one is
//     "result-mismatch" (the stored result must be the REAL tool output, not a claim).
// Problems carry a short Code (the acceptance statistics group by it) and a human Detail.

import { isDeepStrictEqual } from "node:util";
import { ChatTokenList, ChatTokens } from "./ChatTemplate.ts";
import type { ChatDocument, ChatDocumentMessage } from "./ChatDocument.ts";
import { RenderChatDocument } from "./ChatDocument.ts";
import { ToolTokenList } from "../Serving/ToolProtocol.ts";
import type { ToolRegistry } from "../Serving/Tools/ToolRegistry.ts";
import type { ToolContext, ToolResult } from "../Serving/Tools/ToolTypes.ts";
import { SpecialTokenValues } from "../Tokenizer/SpecialTokens.ts";
import { FimTokens } from "../Data/FimReformat.ts";

export type ChatProblemCode =
  | "role-order"
  | "no-assistant"
  | "control-token"
  | "unknown-tool"
  | "bad-arguments"
  | "empty-answer"
  | "empty-thinking"
  | "unclosed-thinking"
  | "call-with-text"
  | "tool-mismatch"
  | "result-mismatch";

export type ChatProblem = { Code: ChatProblemCode; Detail: string };

/** Every reserved token string of the stack (chat, tool, document and FIM sentinels). None may appear
 *  as literal text inside a stored conversation — at training time the assistant side is encoded WITH
 *  specials, so a literal there would become a real control token the model learns to emit. */
export const ReservedTokenLiterals: readonly string[] = [...ChatTokenList, ...ToolTokenList, ...SpecialTokenValues, ...Object.values(FimTokens)];

/** The reserved literals contained in Text (empty = clean). */
export function ReservedLiteralsIn(Text: string): string[] {
  return ReservedTokenLiterals.filter((Literal) => Text.includes(Literal));
}

// Every text field of a message, labeled for the problem detail.
function MessageTexts(Message: ChatDocumentMessage): [string, string][] {
  switch (Message.Role) {
    case "System":
    case "User":
      return [[Message.Role, Message.Content]];
    case "Assistant": {
      const Texts: [string, string][] = [["Assistant", Message.Content]];
      if (Message.Thinking !== undefined) Texts.push(["Thinking", Message.Thinking]);
      if (Message.ToolCall !== undefined) Texts.push(["ToolCall", JSON.stringify(Message.ToolCall)]);
      return Texts;
    }
    case "Tool":
      return [["Tool", JSON.stringify(Message.Result)]];
  }
}

// What may follow each state. "Answer" = an assistant turn without a call; "Call" = with one.
type TurnState = "Start" | "User" | "Call" | "Tool" | "Answer";

function StateOf(Message: ChatDocumentMessage): TurnState {
  if (Message.Role === "Assistant") return Message.ToolCall !== undefined ? "Call" : "Answer";
  if (Message.Role === "Tool") return "Tool";
  return "User"; // System is handled before the walk
}

const Allowed: Record<TurnState, readonly TurnState[]> = {
  Start: ["User"],
  User: ["Call", "Answer"],
  Call: ["Tool"],
  Tool: ["Call", "Answer"], // a result may be followed by another call (chained tools) or the answer
  Answer: ["User"],
};

function RoleOrderProblems(Doc: ChatDocument): ChatProblem[] {
  const Problems: ChatProblem[] = [];
  let State: TurnState = "Start";
  let PendingCall: string | null = null;
  for (let Index = 0; Index < Doc.Messages.length; Index++) {
    const Message = Doc.Messages[Index];
    if (Message.Role === "System") {
      if (Index !== 0) Problems.push({ Code: "role-order", Detail: `System message at position ${Index} (only the first message may be System)` });
      continue;
    }
    const Next = StateOf(Message);
    if (!Allowed[State].includes(Next)) Problems.push({ Code: "role-order", Detail: `${Message.Role} (${Next}) cannot follow ${State} at position ${Index}` });
    if (Message.Role === "Tool" && PendingCall !== null && Message.Name !== PendingCall) {
      Problems.push({ Code: "tool-mismatch", Detail: `result of "${Message.Name}" answers a call to "${PendingCall}"` });
    }
    PendingCall = Message.Role === "Assistant" && Message.ToolCall !== undefined ? Message.ToolCall.name : null;
    State = Next;
  }
  if (State !== "Answer") Problems.push({ Code: "role-order", Detail: `conversation must end on an answering Assistant turn (ends on ${State})` });
  return Problems;
}

function AssistantProblems(Doc: ChatDocument, ToolNames: ReadonlySet<string>): ChatProblem[] {
  const Problems: ChatProblem[] = [];
  for (const Message of Doc.Messages) {
    if (Message.Role !== "Assistant") continue;
    if (Message.Thinking !== undefined && Message.Thinking.trim() === "") Problems.push({ Code: "empty-thinking", Detail: "a <|think|> scratchpad is empty" });
    if (Message.ToolCall !== undefined) {
      if (Message.Content !== "") Problems.push({ Code: "call-with-text", Detail: "a tool-call turn must carry only the call (Content must be empty)" });
      if (!ToolNames.has(Message.ToolCall.name)) Problems.push({ Code: "unknown-tool", Detail: `tool "${Message.ToolCall.name}" is not registered` });
    } else if (Message.Content.trim() === "") {
      Problems.push({ Code: "empty-answer", Detail: "an answering Assistant turn is empty" });
    }
  }
  return Problems;
}

// Every <|think|> in a rendered assistant turn is closed by an <|endthink|> before the next one opens.
function ThinkingClosed(Content: string): boolean {
  let Open = false;
  let Cursor = 0;
  for (;;) {
    const Think = Content.indexOf(ChatTokens.Think, Cursor);
    const End = Content.indexOf(ChatTokens.EndThink, Cursor);
    if (Think === -1 && End === -1) return !Open;
    if (Think !== -1 && (End === -1 || Think < End)) {
      if (Open) return false;
      Open = true;
      Cursor = Think + ChatTokens.Think.length;
    } else {
      if (!Open) return false;
      Open = false;
      Cursor = End + ChatTokens.EndThink.length;
    }
  }
}

/** The structural problems of a document (empty = structurally valid). ToolNames = the registered tools. */
export function ValidateChatStructure(Doc: ChatDocument, ToolNames: ReadonlySet<string>): ChatProblem[] {
  const Problems: ChatProblem[] = [];
  if (!Doc.Messages.some((Message) => Message.Role === "Assistant")) Problems.push({ Code: "no-assistant", Detail: "no Assistant turn" });
  for (const Message of Doc.Messages) {
    for (const [Field, Text] of MessageTexts(Message)) {
      const Found = ReservedLiteralsIn(Text);
      if (Found.length > 0) Problems.push({ Code: "control-token", Detail: `${Field} text contains ${Found.join(", ")}` });
    }
  }
  Problems.push(...RoleOrderProblems(Doc), ...AssistantProblems(Doc, ToolNames));
  const Rendered = RenderChatDocument(Doc);
  if (Rendered.some((Message) => Message.Role === "Assistant" && !ThinkingClosed(Message.Content))) {
    Problems.push({ Code: "unclosed-thinking", Detail: "a <|think|> span is not closed" });
  }
  return Problems;
}

/** Where calls are replayed: the registry (whose names are also the "registered" set) and a FRESH context
 *  per document (memory starts empty, the clock comes from the document's Meta). */
export type ToolReplay = { Registry: ToolRegistry; Context: (Doc: ChatDocument) => ToolContext };

// Order-insensitive canonical JSON: object keys sorted, arrays sorted by their canonical form. Directory
// listings and search matches come back in filesystem order, which differs across machines; a recorded
// result is the same result when it holds the same facts.
function Canonical(Value: unknown): unknown {
  if (Array.isArray(Value)) return Value.map(Canonical).sort((A, B) => (JSON.stringify(A) < JSON.stringify(B) ? -1 : 1));
  if (typeof Value === "object" && Value !== null) {
    return Object.fromEntries(Object.keys(Value).sort().map((Key) => [Key, Canonical((Value as Record<string, unknown>)[Key])]));
  }
  return Value;
}

/** True when two tool results hold the same facts (JSON-normalized, order-insensitive). */
export function SameToolResult(A: ToolResult, B: ToolResult): boolean {
  return isDeepStrictEqual(Canonical(JSON.parse(JSON.stringify(A))), Canonical(JSON.parse(JSON.stringify(B))));
}

/** Re-run every call of the document in order (one fresh context for the whole conversation). */
export async function ReplayToolCalls(Doc: ChatDocument, Replay: ToolReplay): Promise<ChatProblem[]> {
  const Problems: ChatProblem[] = [];
  const Context = Replay.Context(Doc);
  for (let Index = 0; Index < Doc.Messages.length; Index++) {
    const Message = Doc.Messages[Index];
    if (Message.Role !== "Assistant" || Message.ToolCall === undefined) continue;
    const Call = Message.ToolCall;
    if (!Replay.Registry.Has(Call.name)) {
      Problems.push({ Code: "unknown-tool", Detail: `tool "${Call.name}" is not registered` });
      continue;
    }
    const Result = await Replay.Registry.Run({ Name: Call.name, Arguments: Call.arguments }, Context);
    if ("error" in Result) {
      Problems.push({ Code: "bad-arguments", Detail: `${Call.name}(${JSON.stringify(Call.arguments)}) -> ${String(Result["error"])}` });
      continue;
    }
    const Recorded = Doc.Messages[Index + 1];
    if (Recorded?.Role === "Tool" && !SameToolResult(Result, Recorded.Result)) {
      Problems.push({ Code: "result-mismatch", Detail: `${Call.name}: recorded ${JSON.stringify(Recorded.Result)} but the tool returns ${JSON.stringify(Result)}` });
    }
  }
  return Problems;
}

/** Full validation: structure (tool names = the replay registry's), then — only when the structure is
 *  sound — the tool replay. Empty = valid. */
export async function ValidateChatDocument(Doc: ChatDocument, Replay: ToolReplay): Promise<ChatProblem[]> {
  const Names = new Set(Replay.Registry.List().map((Tool) => Tool.Name));
  const Structural = ValidateChatStructure(Doc, Names);
  if (Structural.length > 0) return Structural;
  return ReplayToolCalls(Doc, Replay);
}
