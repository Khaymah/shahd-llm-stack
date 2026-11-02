// The single reader of documents_conversation rows for chat SFT (Scripts/TrainSftChat.ts). Two stored
// shapes are accepted:
//   • legacy collected dialogue, "User: …\n\nAssistant: …" (OASST / StackExchange providers) — one
//     exchange under the given system prompt, with NO fabricated <|think|> span (we do not invent
//     reasoning for text we did not author);
//   • shahd-chat-v1 documents (Brain/Sft/ChatDocument.ts) — whole conversations with thinking and tool
//     steps; their own System turn is kept (the given prompt is prepended when they have none), and a
//     document that fails ValidateChatStructure is skipped rather than trained on.

import type { ChatMessage } from "./ChatTemplate.ts";
import type { ChatDocument } from "./ChatDocument.ts";
import { ChatDocumentFormat, DecodeChatDocument, RenderChatDocument } from "./ChatDocument.ts";
import { ValidateChatStructure } from "./ChatDocumentValidation.ts";

export type StoredConversation = { Format: typeof ChatDocumentFormat | "legacy"; Messages: ChatMessage[]; Document: ChatDocument | null };

// The collected-dialogue shape written by the OASST / StackExchange providers.
const LegacyDialogue = /^User: ([\s\S]*?)\n\nAssistant: ([\s\S]*)$/;

/** Read one row: a shahd-chat-v1 document or a legacy exchange; null when it is neither. */
export function StoredConversationMessages(Content: string, SystemPrompt: string): StoredConversation | null {
  if (Content.trimStart().startsWith("{")) {
    const Doc = DecodeChatDocument(Content);
    if (Doc === null) return null;
    const Messages = RenderChatDocument(Doc);
    if (Messages[0]?.Role !== "System") Messages.unshift({ Role: "System", Content: SystemPrompt });
    return { Format: ChatDocumentFormat, Messages, Document: Doc };
  }
  const Match = LegacyDialogue.exec(Content);
  if (Match === null) return null;
  return {
    Format: "legacy",
    Messages: [{ Role: "System", Content: SystemPrompt }, { Role: "User", Content: Match[1] }, { Role: "Assistant", Content: Match[2] }],
    Document: null,
  };
}

export type StoredConversationTally = { Legacy: number; Chat: number; Skipped: number };

/** Every readable, structurally valid conversation among the rows (in order), with a tally. ToolNames are
 *  the tools a stored call may name. */
export function ReadStoredConversations(Contents: readonly string[], SystemPrompt: string, ToolNames: ReadonlySet<string>): { Conversations: ChatMessage[][]; Tally: StoredConversationTally } {
  const Conversations: ChatMessage[][] = [];
  const Tally: StoredConversationTally = { Legacy: 0, Chat: 0, Skipped: 0 };
  for (const Content of Contents) {
    const Stored = StoredConversationMessages(Content, SystemPrompt);
    if (Stored === null || (Stored.Document !== null && ValidateChatStructure(Stored.Document, ToolNames).length > 0)) {
      Tally.Skipped++;
      continue;
    }
    Conversations.push(Stored.Messages);
    if (Stored.Format === "legacy") Tally.Legacy++;
    else Tally.Chat++;
  }
  return { Conversations, Tally };
}
