// Wraps a base tokenizer and adds atomic special tokens (chat roles, think, FIM) as single ids
// ABOVE the base vocab. Encoding splits on special-token strings (longest match) so e.g.
// "<|user|>" becomes one id, not a run of characters — required for chat-template SFT (Phase 4).

import type { Tokenizer } from "./TokenizerTypes.ts";

export class SpecialTokenizer implements Tokenizer {
  VocabSize: number;
  private Base: Tokenizer;
  private ById: string[]; // id of ById[i] = Base.VocabSize + i (stable id order)
  private MatchOrder: string[]; // longest-first, for greedy matching only
  private SpecialToId: Map<string, number>;

  constructor(Base: Tokenizer, Specials: string[]) {
    this.Base = Base;
    this.ById = [...Specials];
    // Longest-first so a longer special can't be shadowed by a prefix during matching.
    this.MatchOrder = [...Specials].sort((A, B) => B.length - A.length);
    this.VocabSize = Base.VocabSize + Specials.length;
    this.SpecialToId = new Map();
    for (let I = 0; I < this.ById.length; I++) this.SpecialToId.set(this.ById[I], Base.VocabSize + I);
  }

  Id(Special: string): number {
    const Found = this.SpecialToId.get(Special);
    if (Found === undefined) throw new Error(`SpecialTokenizer: unknown special ${Special}`);
    return Found;
  }

  /** The id of a special token, or null when this tokenizer was built without it. Lets ids-level
   *  parsers (tool calls, end-of-turn) degrade to "not present" instead of throwing on a tokenizer
   *  that never reserved that sentinel. */
  TryId(Special: string): number | null {
    return this.SpecialToId.get(Special) ?? null;
  }

  // Encode through the BASE tokenizer ONLY — never emits a special-token id, even if the text
  // literally contains a reserved string like "<|assistant|>". This is the control/data channel
  // separation: UNTRUSTED text (user messages, tool/MCP results, file contents) MUST be encoded with
  // this, never with Encode(), so it can't smuggle a real control token into the stream. Special
  // boundary tokens are then spliced in programmatically via Id() by the caller (RenderForTraining /
  // RenderChatToIds), which is the only sanctioned way to produce a control token.
  EncodeBase(Text: string): number[] {
    return this.Base.Encode(Text);
  }

  Encode(Text: string): number[] {
    const Out: number[] = [];
    let Buffer = "";
    let I = 0;
    while (I < Text.length) {
      const Matched = this.MatchSpecialAt(Text, I);
      if (Matched !== null) {
        if (Buffer.length > 0) {
          for (const Id of this.Base.Encode(Buffer)) Out.push(Id);
          Buffer = "";
        }
        Out.push(this.Id(Matched));
        I += Matched.length;
      } else {
        Buffer += Text[I];
        I += 1;
      }
    }
    if (Buffer.length > 0) for (const Id of this.Base.Encode(Buffer)) Out.push(Id);
    return Out;
  }

  Decode(Ids: number[]): string {
    let Out = "";
    let Run: number[] = [];
    const Flush = (): void => {
      if (Run.length > 0) {
        Out += this.Base.Decode(Run);
        Run = [];
      }
    };
    for (const Id of Ids) {
      if (Id >= this.Base.VocabSize) {
        Flush();
        Out += this.ById[Id - this.Base.VocabSize];
      } else {
        Run.push(Id);
      }
    }
    Flush();
    return Out;
  }

  private MatchSpecialAt(Text: string, Pos: number): string | null {
    for (const Special of this.MatchOrder) {
      if (Text.startsWith(Special, Pos)) return Special;
    }
    return null;
  }
}
