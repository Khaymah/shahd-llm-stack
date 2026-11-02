// Eval-set decontamination (Phase 3). If training docs share a long exact n-gram with any held-out
// eval doc, the eval number measures memorization, not generalization — so drop those train docs
// BEFORE training (REVIEW.md: contaminated held-out perplexity makes the Phase-3 go/no-go bogus).

import { CodePretokenize } from "../Tokenizer/CodePretokenizer.ts";

// Code-aware WORD windows: the code pretokenizer finds the real token boundaries, then only
// word-like tokens (identifiers, keywords, numbers) form the n-grams. Punctuation is dropped on
// purpose: measured on a real corpus, 13-grams that kept punctuation removed 89 of 956 documents and
// 74 of those matched one Go idiom, `(w http.ResponseWriter, r *http.Request) {` - an idiom every
// handler shares, not leaked eval content. Thirteen consecutive WORDS shared with an eval doc is the
// classic (GPT-3 / Phi) contamination signal and does not fire on idioms.
const WordToken = /^[a-z0-9_$]{2,}$/;
function WordNgrams(Text: string, N: number): Set<string> {
  const Toks = CodePretokenize(Text.toLowerCase())
    .map((T) => T.trim())
    .filter((T) => WordToken.test(T));
  const Out = new Set<string>();
  for (let I = 0; I + N <= Toks.length; I++) Out.add(Toks.slice(I, I + N).join(" "));
  return Out;
}

export type DecontaminationResult = { Kept: number[]; Removed: number[] };

/** The eval set's N-word n-grams, built ONCE and queried per training document — for streaming callers
 *  (the SFT synthesizer judges one conversation at a time) that must not rebuild the index per doc. */
export class EvalNgramIndex {
  private readonly Ngrams = new Set<string>();
  private readonly NgramSize: number;

  constructor(EvalDocs: readonly string[], NgramSize = 13) {
    this.NgramSize = NgramSize;
    for (const Doc of EvalDocs) {
      for (const Gram of WordNgrams(Doc, NgramSize)) this.Ngrams.add(Gram);
    }
  }

  /** True when Doc shares any N-word n-gram with the eval set. */
  Contaminates(Doc: string): boolean {
    for (const Gram of WordNgrams(Doc, this.NgramSize)) {
      if (this.Ngrams.has(Gram)) return true;
    }
    return false;
  }
}

/** Remove train docs sharing any N-word n-gram with the eval set. */
export function Decontaminate(TrainDocs: string[], EvalDocs: string[], NgramSize = 13): DecontaminationResult {
  const Index = new EvalNgramIndex(EvalDocs, NgramSize);
  const Kept: number[] = [];
  const Removed: number[] = [];
  for (let I = 0; I < TrainDocs.length; I++) {
    if (Index.Contaminates(TrainDocs[I])) Removed.push(I);
    else Kept.push(I);
  }
  return { Kept, Removed };
}
