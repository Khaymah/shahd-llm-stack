// The acceptance gate every synthetic conversation passes before it is stored, in order:
//   1. ValidateChatDocument — role order, control-token literals, empty answers, closed thinking,
//      registered tools, and the tool REPLAY (arguments parse, recorded result = real result);
//   2. the token budget (the whole rendered conversation must fit);
//   3. exact dedup by ConversationHash (within the run; the merge job dedups across shards);
//   4. eval decontamination against the frozen suite (TrainingDecontaminationDocs by default) — a
//      conversation sharing a 13-word n-gram with any eval text is dropped.
// The verdict names the first failing reason, which is what the per-generator statistics count.

import { EvalNgramIndex } from "../../Data/Decontamination.ts";
import type { ChatDocument } from "../ChatDocument.ts";
import { ChatDocumentText, ConversationHash } from "../ChatDocument.ts";
import type { ToolReplay } from "../ChatDocumentValidation.ts";
import { ValidateChatDocument } from "../ChatDocumentValidation.ts";
import type { DocumentMeasure } from "./SynthBudget.ts";

export type AcceptanceVerdict = { Ok: true; Tokens: number; Hash: string } | { Ok: false; Reason: string; Detail: string };

export type AcceptanceOptions = {
  EvalDocs: readonly string[];
  Measure: DocumentMeasure;
  Budget: number;
  Replay: ToolReplay;
};

export class SynthAcceptance {
  private readonly Seen = new Set<string>();
  private readonly Index: EvalNgramIndex;
  private readonly Options: AcceptanceOptions;

  constructor(Options: AcceptanceOptions) {
    this.Options = Options;
    this.Index = new EvalNgramIndex(Options.EvalDocs);
  }

  /** Judge one conversation. Does NOT remember it — call Remember once it is actually stored. */
  async Judge(Doc: ChatDocument): Promise<AcceptanceVerdict> {
    const Problems = await ValidateChatDocument(Doc, this.Options.Replay);
    if (Problems.length > 0) return { Ok: false, Reason: Problems[0].Code, Detail: Problems.map((P) => P.Detail).join("; ") };
    const Tokens = this.Options.Measure(Doc);
    if (Tokens > this.Options.Budget) return { Ok: false, Reason: "over-budget", Detail: `${Tokens} > ${this.Options.Budget} tokens` };
    const Hash = ConversationHash(Doc);
    if (this.Seen.has(Hash)) return { Ok: false, Reason: "duplicate", Detail: Hash.slice(0, 16) };
    if (this.Index.Contaminates(ChatDocumentText(Doc))) return { Ok: false, Reason: "contaminated", Detail: "shares a 13-word n-gram with the eval suite" };
    return { Ok: true, Tokens, Hash };
  }

  /** Record a stored conversation's hash so a later identical one is a duplicate. */
  Remember(Hash: string): void {
    this.Seen.add(Hash);
  }
}
