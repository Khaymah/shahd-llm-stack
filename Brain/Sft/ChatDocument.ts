// The canonical STORED form of one whole SFT conversation — "shahd-chat-v1" — including the thinking
// scratchpads and the tool steps, so a conversation can live in documents_conversation as data and be
// rendered for training without any re-interpretation:
//
//   {"Format":"shahd-chat-v1","Messages":[…],"Meta":{…}}
//
// Messages are STRUCTURED, never pre-rendered: an assistant turn keeps its Thinking, its visible
// Content and its ToolCall as separate fields; a tool step is its own { Role: "Tool", Name, Result }
// message. RenderChatDocument turns that into the exact ChatMessage[] the authored mix builds
// (BuildToolConversation / OwnedSftData): thinking as <|think|>…<|endthink|> before the turn's body, a
// tool call as ToolCallText, a tool result as a User turn carrying FormatToolResult — so a stored
// conversation and an in-code exemplar of the same content train identically. Because the control
// tokens are produced only here, by construction, a text field that itself contains one is invalid
// (see ChatDocumentValidation.ts).
//
// The legacy collected-dialogue shape ("User: …\n\nAssistant: …") is still accepted by
// StoredConversations.ts, the single reader TrainSftChat uses for documents_conversation.

import { createHash } from "node:crypto";
import { z } from "zod";
import type { ChatMessage } from "./ChatTemplate.ts";
import { ToolCallText } from "./ToolUseExamples.ts";
import { WrapThinking } from "../Reasoning/ThinkingMode.ts";
import { FormatToolResult } from "../Serving/ToolProtocol.ts";

export const ChatDocumentFormat = "shahd-chat-v1";

// A tool call exactly as the model emits it inside <|tool_call|>…: lowercase JSON keys by protocol.
const ToolCallSchema = z.object({
  name: z.string().min(1),
  arguments: z.record(z.string(), z.unknown()),
});

const SystemMessageSchema = z.object({ Role: z.literal("System"), Content: z.string() });
const UserMessageSchema = z.object({ Role: z.literal("User"), Content: z.string() });
const AssistantMessageSchema = z.object({
  Role: z.literal("Assistant"),
  Thinking: z.string().optional(), // the <|think|> scratchpad that opens the turn
  Content: z.string(), // the visible answer; "" on a tool-call turn
  ToolCall: ToolCallSchema.optional(),
});
const ToolMessageSchema = z.object({
  Role: z.literal("Tool"),
  Name: z.string().min(1), // the tool that produced Result (must match the preceding call)
  Result: z.record(z.string(), z.unknown()), // the REAL tool output, as JSON
});

export const ChatDocumentMessageSchema = z.discriminatedUnion("Role", [SystemMessageSchema, UserMessageSchema, AssistantMessageSchema, ToolMessageSchema]);

// Provenance + replay facts. The named keys are the ones the pipeline reads; anything else is kept as
// opaque scalar annotations (forward compatible, still JSON-flat).
const ChatDocumentMetaSchema = z
  .object({
    Generator: z.string().optional(),
    Teacher: z.string().optional(),
    SourceId: z.string().optional(),
    Slice: z.string().optional(),
    Task: z.string().optional(),
    ClockMs: z.number().int().nonnegative().optional(), // the clock the tool steps ran under (replay)
  })
  .catchall(z.union([z.string(), z.number(), z.boolean()]));

export const ChatDocumentSchema = z.object({
  Format: z.literal(ChatDocumentFormat),
  Messages: z.array(ChatDocumentMessageSchema).min(1),
  Meta: ChatDocumentMetaSchema,
});

export type ChatDocument = z.infer<typeof ChatDocumentSchema>;
export type ChatDocumentMessage = z.infer<typeof ChatDocumentMessageSchema>;
export type ChatDocumentAssistant = z.infer<typeof AssistantMessageSchema>;
export type ChatDocumentToolCall = z.infer<typeof ToolCallSchema>;
export type ChatDocumentMeta = z.infer<typeof ChatDocumentMetaSchema>;

/** Thrown by ParseChatDocument on text that is not a well-formed shahd-chat-v1 document. */
export class ChatDocumentError extends Error {}

// Canonical key order per message, so equal conversations serialize to equal bytes (exact dedup and
// the store's content hash both depend on it).
function CanonicalMessage(Message: ChatDocumentMessage): ChatDocumentMessage {
  switch (Message.Role) {
    case "System":
    case "User":
      return { Role: Message.Role, Content: Message.Content };
    case "Assistant":
      return {
        Role: "Assistant",
        ...(Message.Thinking !== undefined ? { Thinking: Message.Thinking } : {}),
        Content: Message.Content,
        ...(Message.ToolCall !== undefined ? { ToolCall: { name: Message.ToolCall.name, arguments: Message.ToolCall.arguments } } : {}),
      };
    case "Tool":
      return { Role: "Tool", Name: Message.Name, Result: Message.Result };
  }
}

/** The stored text of a document (one JSON line, canonical key order). */
export function SerializeChatDocument(Doc: ChatDocument): string {
  return JSON.stringify({ Format: ChatDocumentFormat, Messages: Doc.Messages.map(CanonicalMessage), Meta: Doc.Meta });
}

/** Decode stored text into a document, or null when it is not valid shahd-chat-v1 JSON. */
export function DecodeChatDocument(Text: string): ChatDocument | null {
  let Json: unknown;
  try {
    Json = JSON.parse(Text);
  } catch {
    return null;
  }
  const Parsed = ChatDocumentSchema.safeParse(Json);
  return Parsed.success ? Parsed.data : null;
}

/** One stored message as the ChatMessage the trainer renders (see the header for the mapping). */
function RenderMessage(Message: ChatDocumentMessage): ChatMessage {
  switch (Message.Role) {
    case "System":
      return { Role: "System", Content: Message.Content };
    case "User":
      return { Role: "User", Content: Message.Content };
    case "Assistant": {
      const Body = Message.ToolCall !== undefined ? ToolCallText(Message.ToolCall) : Message.Content;
      return { Role: "Assistant", Content: Message.Thinking !== undefined ? WrapThinking(Message.Thinking, Body) : Body };
    }
    case "Tool":
      return { Role: "User", Content: FormatToolResult(Message.Result) };
  }
}

/** The conversation as ChatMessage[] — exactly what OwnedSftData / BuildToolConversation produce for
 *  the same content, ready for RenderForTraining. */
export function RenderChatDocument(Doc: ChatDocument): ChatMessage[] {
  return Doc.Messages.map(RenderMessage);
}

/** Parse stored shahd-chat-v1 text straight to ChatMessage[]; throws ChatDocumentError when malformed. */
export function ParseChatDocument(Text: string): ChatMessage[] {
  const Doc = DecodeChatDocument(Text);
  if (Doc === null) throw new ChatDocumentError(`not a ${ChatDocumentFormat} document`);
  return RenderChatDocument(Doc);
}

/** The exact-dedup key of a conversation: sha256 over its canonical MESSAGES only (Meta excluded, so the
 *  same conversation produced from two sources or by two runs is one conversation). */
export function ConversationHash(Doc: ChatDocument): string {
  return createHash("sha256").update(JSON.stringify(Doc.Messages.map(CanonicalMessage))).digest("hex");
}

/** All rendered message text joined — what decontamination and prose-quality scoring read. */
export function ChatDocumentText(Doc: ChatDocument): string {
  return RenderChatDocument(Doc).map((Message) => Message.Content).join("\n");
}
