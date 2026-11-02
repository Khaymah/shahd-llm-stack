// Heuristic code-quality filter (Phase 3, phi-1-style "quality is the real lever"). Cheap,
// deterministic signals that keep clean, human-written code and drop minified/generated/binary-ish
// junk. Not perfect — a first-pass filter before the corpus reaches training.
//
// Code (not prose) is also judged on its comment-to-code ratio (Plan §2.2: code without comments
// teaches less): ~5–30% comment lines is the healthy band and costs nothing; no comments at all, a
// sliver, or a comment-heavy file cost a little; a file that is mostly comments (>60%: commented-out
// code, license walls) costs more. Files that carry tests or runnable examples earn a small bonus that
// can offset those penalties (the score never exceeds 1). Both show up in Reasons.

import { CommentLineRatio, HasTestsOrExamples, ScanSource } from "./EducationalFeatures.ts";

export type QualityResult = { Score: number; Passed: boolean; Reasons: string[] };

export type QualityOptions = {
  MaxAvgLineLength?: number; // very long avg lines => minified/generated
  MaxLongLineFraction?: number; // fraction of lines over 400 chars
  MinPrintableFraction?: number; // guards against binary/base64 blobs
  MinAlphaFraction?: number; // guards against near-empty / symbol soup
  MaxDupLineFraction?: number; // fraction of non-empty lines that are duplicates (boilerplate/degenerate)
  Threshold?: number; // pass if Score >= Threshold
  Prose?: boolean; // natural-language text (dialogue/articles): skip the code-only line-length checks
};

/** Prose (conversation / articles / books): natural language whose paragraphs are legitimately long,
 *  so the code-minification line-length heuristics do NOT apply. Keeps only the empty/binary/symbol
 *  guards. Use this for curated text sources; ScoreCodeQuality (line-length aware) is for source code. */
export function ScoreTextQuality(Text: string, Options: QualityOptions = {}): QualityResult {
  return ScoreCodeQuality(Text, { ...Options, Prose: true });
}

/** The healthy comment-line band (fraction of non-empty lines) and the "mostly comments" limit. */
export const CommentBand = { Low: 0.05, High: 0.3, Excess: 0.6 } as const;
const NoCommentPenalty = 0.1; // 0% comment lines
const OutOfBandPenalty = 0.05; // a sliver (<5%) or comment-heavy (30–60%)
const ExcessCommentPenalty = 0.2; // >60%: mostly comments
const TestsOrExamplesBonus = 0.1;
const MaxScore = 1;

/** Comment-ratio penalty (0 inside the healthy band) and a human-readable reason when non-zero. */
function CommentRatioPenalty(Ratio: number): { Penalty: number; Reason: string | null } {
  const Shown = Ratio.toFixed(2);
  if (Ratio === 0) return { Penalty: NoCommentPenalty, Reason: `comment ratio ${Shown}: no comments (-${NoCommentPenalty})` };
  if (Ratio < CommentBand.Low) return { Penalty: OutOfBandPenalty, Reason: `comment ratio ${Shown} < ${CommentBand.Low} (-${OutOfBandPenalty})` };
  if (Ratio > CommentBand.Excess) return { Penalty: ExcessCommentPenalty, Reason: `comment ratio ${Shown} > ${CommentBand.Excess}: mostly comments (-${ExcessCommentPenalty})` };
  if (Ratio > CommentBand.High) return { Penalty: OutOfBandPenalty, Reason: `comment ratio ${Shown} > ${CommentBand.High} (-${OutOfBandPenalty})` };
  return { Penalty: 0, Reason: null };
}

export function ScoreCodeQuality(Text: string, Options: QualityOptions = {}): QualityResult {
  const MaxAvgLineLength = Options.MaxAvgLineLength ?? 120;
  const MaxLongLineFraction = Options.MaxLongLineFraction ?? 0.1;
  const MinPrintableFraction = Options.MinPrintableFraction ?? 0.95;
  const MinAlphaFraction = Options.MinAlphaFraction ?? 0.25;
  const MaxDupLineFraction = Options.MaxDupLineFraction ?? 0.5;
  const Threshold = Options.Threshold ?? 0.6;

  const Reasons: string[] = [];
  let Score = 1;

  const Lines = Text.split("\n");
  const NonEmpty = Lines.filter((L) => L.trim().length > 0);
  if (NonEmpty.length === 0) {
    return { Score: 0, Passed: false, Reasons: ["empty"] };
  }

  // Repetition guard (Gopher/C4-style): heavily duplicated lines signal boilerplate or degenerate
  // (e.g. "a\na\na\n…") text that the length/printable checks miss because each line looks clean.
  if (NonEmpty.length >= 8) {
    const DupFraction = 1 - new Set(NonEmpty.map((L) => L.trim())).size / NonEmpty.length;
    if (DupFraction > MaxDupLineFraction) {
      Score -= 0.3;
      Reasons.push(`duplicate-line fraction ${DupFraction.toFixed(2)} > ${MaxDupLineFraction} (boilerplate/degenerate?)`);
    }
  }

  // Line-length signals detect MINIFIED/generated CODE — they do not apply to prose (a paragraph is
  // one long line by design), so Prose text skips them and is judged on content, not layout.
  if (!Options.Prose) {
    const AvgLineLength = Text.length / Lines.length;
    if (AvgLineLength > MaxAvgLineLength) {
      Score -= 0.4;
      Reasons.push(`avg line length ${AvgLineLength.toFixed(0)} > ${MaxAvgLineLength} (minified?)`);
    }

    const LongLines = Lines.filter((L) => L.length > 400).length;
    const LongFraction = LongLines / Lines.length;
    if (LongFraction > MaxLongLineFraction) {
      Score -= 0.3;
      Reasons.push(`long-line fraction ${LongFraction.toFixed(2)} > ${MaxLongLineFraction}`);
    }

    const Comments = CommentRatioPenalty(CommentLineRatio(ScanSource(Text)));
    if (Comments.Reason !== null) {
      Score -= Comments.Penalty;
      Reasons.push(Comments.Reason);
    }
    if (HasTestsOrExamples(Text)) {
      Score = Math.min(MaxScore, Score + TestsOrExamplesBonus); // offsets layout penalties, never lifts past 1
      Reasons.push(`has tests or examples (+${TestsOrExamplesBonus})`);
    }
  }

  let Printable = 0;
  let Alpha = 0;
  for (let I = 0; I < Text.length; I++) {
    const Code = Text.charCodeAt(I);
    // Non-ASCII (Code >= 128) is legitimate TEXT — Arabic, other scripts, emoji, unicode identifiers —
    // NOT binary. Counting it as printable+alpha keeps the filter from wrongly rejecting non-English
    // prose as "binary"; real binary is control chars (< 32, excluding tab/newline/CR), still caught.
    // U+FFFD (replacement char) is excluded from both: it marks a failed decode (mojibake/corrupted
    // bytes), so a run of them must NOT count as printable/alpha text or the binary guard is defeated.
    const IsReplacementChar = Code === 0xfffd;
    if (!IsReplacementChar && (Code === 9 || Code === 10 || Code === 13 || (Code >= 32 && Code < 127) || Code >= 128)) Printable++;
    if (!IsReplacementChar && ((Code >= 65 && Code <= 90) || (Code >= 97 && Code <= 122) || Code >= 128)) Alpha++;
  }
  const PrintableFraction = Printable / Text.length;
  const AlphaFraction = Alpha / Text.length;
  if (PrintableFraction < MinPrintableFraction) {
    Score -= 0.5;
    Reasons.push(`printable fraction ${PrintableFraction.toFixed(2)} < ${MinPrintableFraction} (binary/base64?)`);
  }
  if (AlphaFraction < MinAlphaFraction) {
    Score -= 0.2;
    Reasons.push(`alpha fraction ${AlphaFraction.toFixed(2)} < ${MinAlphaFraction}`);
  }

  Score = Math.min(MaxScore, Math.max(0, Score));
  return { Score, Passed: Score >= Threshold, Reasons };
}
