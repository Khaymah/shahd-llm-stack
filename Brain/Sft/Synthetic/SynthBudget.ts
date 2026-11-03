// Token budgeting for synthetic conversations. A conversation is only useful if it fits the context the
// chat model is trained at, so every generator bounds its grounding snippet up front and the acceptance
// gate measures the WHOLE rendered conversation at the end. Measurement uses the real tokenizer when one
// is given (RenderForTraining — exactly what training will see) and otherwise a chars/3.6 estimate.

import type { SpecialTokenizer } from "../../Tokenizer/SpecialTokenizer.ts";
import { RenderChat, RenderForTraining } from "../ChatTemplate.ts";
import type { ChatDocument } from "../ChatDocument.ts";
import { RenderChatDocument } from "../ChatDocument.ts";

/** Average characters per token of the byte-level BPE on code (the fallback estimate's divisor). */
export const EstimatedCharsPerToken = 3.6;

/** The default conversation budget in tokens. */
export const DefaultSynthBudget = 512;

export type TextMeasure = (Text: string) => number;
export type DocumentMeasure = (Doc: ChatDocument) => number;

/** Estimated token count of a text. */
export function EstimateTokens(Text: string): number {
  return Math.ceil(Text.length / EstimatedCharsPerToken);
}

/** Text measure: the base tokenizer when given (content is base-encoded as user text), else the estimate. */
export function CreateTextMeasure(Tok: SpecialTokenizer | null): TextMeasure {
  return Tok === null ? EstimateTokens : (Text: string): number => Tok.EncodeBase(Text).length;
}

/** Whole-conversation measure: the exact training sequence length, or the estimate over the rendered chat. */
export function CreateDocumentMeasure(Tok: SpecialTokenizer | null): DocumentMeasure {
  if (Tok === null) return (Doc: ChatDocument): number => EstimateTokens(RenderChat(RenderChatDocument(Doc), false));
  return (Doc: ChatDocument): number => RenderForTraining(RenderChatDocument(Doc), Tok).Ids.length;
}

/**
 * A window of whole lines of Content starting at StartLine (leading blank lines skipped) whose measured
 * size stays within MaxTokens. Lines are measured one at a time (plus the newline), so the cost is linear;
 * a single line over the budget ends the window (minified / data lines make poor grounding anyway).
 */
export function BoundSnippet(Content: string, StartLine: number, MaxTokens: number, Measure: TextMeasure): string {
  const Lines = Content.split("\n");
  let First = Math.max(0, Math.min(StartLine, Lines.length - 1));
  while (First < Lines.length && Lines[First].trim() === "") First++;
  const Kept: string[] = [];
  let Used = 0;
  for (let I = First; I < Lines.length; I++) {
    const Cost = Measure(`${Lines[I]}\n`);
    if (Used + Cost > MaxTokens) break;
    Kept.push(Lines[I]);
    Used += Cost;
  }
  return Kept.join("\n").trimEnd();
}
