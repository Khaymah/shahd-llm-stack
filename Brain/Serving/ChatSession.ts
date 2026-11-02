// Multi-turn chat state (Phase 6). Holds the running conversation and renders the model prompt via
// the chat template. Tool results are appended as user-visible turns so the model can read them.

import type { ChatMessage } from "../Sft/ChatTemplate.ts";
import { RenderChat, RenderChatToIds } from "../Sft/ChatTemplate.ts";
import type { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";
import { ToolTokens } from "./ToolProtocol.ts";
import type { Summarizer } from "./Compaction.ts";

export class ChatSession {
  Messages: ChatMessage[] = [];

  constructor(SystemPrompt?: string) {
    if (SystemPrompt !== undefined) this.Messages.push({ Role: "System", Content: SystemPrompt });
  }

  AddUser(Content: string): void {
    this.Messages.push({ Role: "User", Content });
  }

  /** Record an assistant turn. Pass Ids (the model's generated content ids) whenever the turn came from
   *  the model: RenderPromptIds then splices them verbatim, so its control tokens stay exactly the ones
   *  it emitted. Without Ids (a client-replayed history, a scripted string) the text is base-encoded at
   *  render time and can never be promoted into control tokens. */
  AddAssistant(Content: string, Ids?: readonly number[]): void {
    this.Messages.push(Ids === undefined ? { Role: "Assistant", Content } : { Role: "Assistant", Content, Ids: [...Ids] });
  }

  /** Feed a tool result back into the conversation (as a user turn the model reads next). */
  AddToolResult(Content: string): void {
    this.Messages.push({ Role: "User", Content });
  }

  /** Render the prompt string for the model, cueing it to produce the next assistant turn.
   *  NOTE: for a special-token model this string must NOT be re-encoded with SpecialTokenizer.Encode
   *  (that re-scans content and lets it smuggle control tokens) — use RenderPromptIds instead. Kept
   *  for base models and prompt inspection. */
  RenderPrompt(): string {
    return RenderChat(this.Messages, true);
  }

  /** Render the prompt directly to token ids — the SAFE serving path for special-token (chat) models.
   *  Content is base-encoded (never re-scanned for specials), so untrusted user/tool text can't forge
   *  role/turn boundaries; assistant turns recorded with Ids are spliced verbatim. Uses the same
   *  per-message core as training, so no train/serve skew. */
  RenderPromptIds(Tok: SpecialTokenizer): number[] {
    return RenderChatToIds(this.Messages, Tok, true);
  }

  /**
   * Compact the conversation: keep the leading system message (if any) and the last `Keep` non-system
   * turns, collapsing everything dropped into ONE note. When a Summarizer is provided the note is a
   * summary of the dropped turns (key points preserved); otherwise it falls back to a structural
   * elision marker. Returns how many turns were dropped.
   */
  Compact(Keep: number, Summarize?: Summarizer): number {
    const System = this.Messages[0]?.Role === "System" ? [this.Messages[0]] : [];
    const Body = this.Messages.slice(System.length);
    if (Body.length <= Keep) return 0;
    let Dropped = Body.length - Keep;
    // Don't strand a tool RESULT without its preceding tool CALL: if the first kept turn is a tool
    // result, keep one more turn (its call) so the call/result pair stays intact in the retained window.
    if (Dropped > 0 && (Body[Dropped]?.Content.includes(ToolTokens.ResultStart) ?? false)) Dropped -= 1;
    if (Dropped <= 0) return 0;
    const DroppedMessages = Body.slice(0, Dropped);
    const Recent = Body.slice(Dropped);
    const Content = Summarize ? Summarize(DroppedMessages) : `[${Dropped} earlier turn(s) elided to save context]`;
    const Note: ChatMessage = { Role: "System", Content };
    this.Messages = [...System, Note, ...Recent];
    return Dropped;
  }
}
