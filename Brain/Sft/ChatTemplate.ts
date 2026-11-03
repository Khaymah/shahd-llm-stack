// Chat template + special tokens (Phase 4). Renders a system/user/assistant conversation into
// the token format the model is SFT'd on. RenderForTraining also returns a loss mask so training
// counts ONLY the assistant's response tokens (and its end-of-turn), never the prompt.

import type { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";

export const ChatTokens = {
  System: "<|system|>",
  User: "<|user|>",
  Assistant: "<|assistant|>",
  EndOfTurn: "<|endofturn|>",
  Think: "<|think|>",
  EndThink: "<|endthink|>",
} as const;

export const ChatTokenList: readonly string[] = Object.values(ChatTokens);

export type ChatRole = "System" | "User" | "Assistant";
// Ids (Assistant turns only): the EXACT content ids the model generated for this turn, without the
// role marker or end-of-turn. When present they are spliced into the stream verbatim, so a turn's
// control tokens are exactly the ones the model emitted as special ids — never a re-scan of its
// decoded text (where an echoed "<|tool_call|>" made of ordinary byte tokens is indistinguishable
// from the real sentinel). Content stays the decoded text for display/compaction/inspection.
export type ChatMessage = { Role: ChatRole; Content: string; Ids?: readonly number[] };

const RoleToken: Record<ChatRole, string> = {
  System: ChatTokens.System,
  User: ChatTokens.User,
  Assistant: ChatTokens.Assistant,
};

/** Render a conversation to a plain string; AddAssistantCue appends the assistant token to prompt a reply. */
export function RenderChat(Messages: ChatMessage[], AddAssistantCue = true): string {
  let Out = "";
  for (const Message of Messages) Out += RoleToken[Message.Role] + Message.Content + ChatTokens.EndOfTurn;
  if (AddAssistantCue) Out += ChatTokens.Assistant;
  return Out;
}

export type TrainingSequence = { Ids: number[]; LossMask: boolean[] };

// The SINGLE id-building core, shared by training (RenderForTraining) and serving (RenderChatToIds)
// so their tokenization can NEVER drift (DRY-strict + no train/serve skew — a mismatch would silently
// break the model). One message becomes: role marker (a special id) + content + end-of-turn (a special
// id). The role/turn markers are the ONLY boundary tokens, produced only via Id().
//
// Content encoding is TRUST-based (the control/data channel separation):
//   • Assistant content WITH Ids is the model's own generation, kept as ids: spliced verbatim.
//   • Assistant content as TEXT is trusted only when TrustAssistantText is set — i.e. AUTHORED training
//     data (tool-use exemplars, SFT sets), whose control strings (<|tool_call|>, <|think|>…) MUST become
//     atomic ids. At SERVING time assistant text without ids is a client-supplied history replay or a
//     decoded string, so it is base-encoded: an echoed "<|tool_call|>" can never be promoted to a real
//     control token on the next turn.
//   • System + User content is EXTERNAL/untrusted (system prompt we author, but crucially USER messages
//     and TOOL RESULTS which arrive as User turns). It is encoded with EncodeBase() — the base
//     tokenizer only — so a reserved string like "<|assistant|>" inside it stays ordinary text and can
//     NEVER smuggle a real control token / forge a turn boundary in the stream.
// ContentTrainable gates the loss mask on the content + its end-of-turn.
export function AppendChatMessage(
  Ids: number[],
  LossMask: boolean[],
  Message: ChatMessage,
  Tok: SpecialTokenizer,
  ContentTrainable: boolean,
  TrustAssistantText = true,
): void {
  Ids.push(Tok.Id(RoleToken[Message.Role]));
  LossMask.push(false); // role marker is never trained
  for (const Id of MessageContentIds(Message, Tok, TrustAssistantText)) {
    Ids.push(Id);
    LossMask.push(ContentTrainable);
  }
  Ids.push(Tok.Id(ChatTokens.EndOfTurn)); // teach the model to stop after replying
  LossMask.push(ContentTrainable);
}

// The content ids of one message under the trust rules above (Ids first, then trusted/untrusted text).
function MessageContentIds(Message: ChatMessage, Tok: SpecialTokenizer, TrustAssistantText: boolean): readonly number[] {
  if (Message.Role !== "Assistant") return Tok.EncodeBase(Message.Content);
  if (Message.Ids !== undefined) return Message.Ids;
  return TrustAssistantText ? Tok.Encode(Message.Content) : Tok.EncodeBase(Message.Content);
}

/** Token ids + per-token loss mask (true only on assistant content + its end-of-turn). */
export function RenderForTraining(Messages: ChatMessage[], Tok: SpecialTokenizer): TrainingSequence {
  const Ids: number[] = [];
  const LossMask: boolean[] = [];
  for (const Message of Messages) {
    AppendChatMessage(Ids, LossMask, Message, Tok, Message.Role === "Assistant");
  }
  return { Ids, LossMask };
}

/** Render a conversation directly to token ids (the SAFE serving path). Uses the same per-message core
 *  as training, so serving and training tokenize identically. AddAssistantCue appends the assistant
 *  marker id to prompt the next reply. Bypasses the string round-trip that let content smuggle
 *  control tokens (via SpecialTokenizer.Encode re-scanning the whole rendered string). Assistant turns
 *  carrying Ids are spliced verbatim; assistant TEXT is base-encoded unless TrustAssistantText (only
 *  for authored, non-client content) — see AppendChatMessage. */
export function RenderChatToIds(
  Messages: ChatMessage[],
  Tok: SpecialTokenizer,
  AddAssistantCue = true,
  TrustAssistantText = false,
): number[] {
  const Ids: number[] = [];
  const LossMask: boolean[] = [];
  for (const Message of Messages) AppendChatMessage(Ids, LossMask, Message, Tok, false, TrustAssistantText);
  if (AddAssistantCue) Ids.push(Tok.Id(ChatTokens.Assistant));
  return Ids;
}

// Ids that END the assistant's turn when the model emits them: its own end-of-turn, or any role marker
// (a model that starts "speaking" as the user/system has left its turn). Truncating there keeps a
// stored assistant turn from ever carrying a forged boundary into the next prompt.
function TurnEndIds(Tok: SpecialTokenizer): Set<number> {
  const Out = new Set<number>();
  for (const Special of [ChatTokens.EndOfTurn, ChatTokens.System, ChatTokens.User, ChatTokens.Assistant]) {
    const Id = Tok.TryId(Special);
    if (Id !== null) Out.add(Id);
  }
  return Out;
}

/** The generated ids up to (excluding) the first end-of-turn / role-marker SPECIAL id. Detection is by
 *  id, so the text "<|endofturn|>" spelled with ordinary byte tokens does not end the turn. */
export function TruncateAtTurnEnd(Ids: readonly number[], Tok: SpecialTokenizer): number[] {
  const Ends = TurnEndIds(Tok);
  const Out: number[] = [];
  for (const Id of Ids) {
    if (Ends.has(Id)) break;
    Out.push(Id);
  }
  return Out;
}

export type ThinkingIdsSplit = { ThinkingIds: number[][]; AnswerIds: number[]; HadThinking: boolean };

/** Ids-level sibling of SplitThinking (Reasoning/ThinkingMode.ts): separates <|think|>…<|endthink|>
 *  spans — located by their SPECIAL ids only — from the visible answer ids. An unclosed think span hides
 *  everything after it (incomplete reasoning is never an answer), matching the string version. */
export function SplitThinkingIds(Ids: readonly number[], Tok: SpecialTokenizer): ThinkingIdsSplit {
  const ThinkId = Tok.TryId(ChatTokens.Think);
  const EndThinkId = Tok.TryId(ChatTokens.EndThink);
  const ThinkingIds: number[][] = [];
  const AnswerIds: number[] = [];
  let Current: number[] | null = null;
  for (const Id of Ids) {
    if (Current === null) {
      if (ThinkId !== null && Id === ThinkId) {
        Current = [];
        ThinkingIds.push(Current);
      } else {
        AnswerIds.push(Id);
      }
    } else if (EndThinkId !== null && Id === EndThinkId) {
      Current = null;
    } else {
      Current.push(Id);
    }
  }
  return { ThinkingIds, AnswerIds, HadThinking: ThinkingIds.length > 0 };
}
