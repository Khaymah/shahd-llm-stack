// Tier classification (M3) — the one place a document's fate is decided, reusing the existing Data
// filters (rule #4: no reimplementation). Rules, in order:
//   general web    -> Raw (isolated, inspect-only) regardless of license (license is unverified).
//   non-permissive -> Rejected (with the license as the reason).
//   low quality    -> Rejected (with the structural quality reasons).
//   low educational value -> Rejected (code only, Plan §2.2 layer 3; the two features that pulled the
//                    score down the most are the reason).
//   otherwise      -> Filtered (permissive + clean, eligible for training).
// Synthetic conversations (Origin "synthetic") take their own path, ClassifySynthetic: the content must
// be a shahd-chat-v1 document, the License "<source SPDX>;teacher:<model>" with an accepted SOURCE
// (Foundry/SyntheticLicense.ts), and the rendered conversation is quality-scored as PROSE.
//
// The educational score applies to CODE: a document whose Lang is a source-code language (see
// IsCodeLang), whose origin is not curated prose, and that is at least the structural size window's
// minimum (DefaultFileFilter.MinBytes) — on a 30-byte snippet the per-KB / per-100-line rates are
// noise, and tiny files are the size gate's job (layer 1), not this one's. Callers that pass no Lang
// (or a prose tag such as "text-ar") keep the structural-only behaviour.
//
// QualityScore records the score of the quality gate that decided: the STRUCTURAL score when the
// structural gate fails (or the educational score does not apply), else the EDUCATIONAL score (the
// plan's "real quality_score instead of the heuristic"). It is computed even when the license gate
// rejects, so a later license backfill (ReclassifyBySource, MinQuality) judges the same number — and a
// structurally-broken file can never be promoted on a good educational score (its stored score is the
// failing structural one, below the 0.6 structural threshold).

import { IsPermissive } from "../Brain/Data/LicenseManifest.ts";
import { ScoreCodeQuality, ScoreTextQuality } from "../Brain/Data/QualityFilter.ts";
import { ActiveEduWeights, ScoreEducational, TopNegativeContributions } from "../Brain/Data/EducationalScore.ts";
import type { EduWeights } from "../Brain/Data/EducationalScore.ts";
import { DefaultFileFilter, IsCodeLang } from "./CodeFileFilter.ts";
import type { Tier, Origin } from "./DocumentRecord.ts";
import { ChatDocumentText, DecodeChatDocument } from "../Brain/Sft/ChatDocument.ts";
import { IsAcceptedSyntheticSource, ParseSyntheticLicense } from "./SyntheticLicense.ts";

export type TierDecision = { Tier: Tier; QualityScore: number; RejectReason: string | null };

export type ClassifyOptions = {
  /** The document's language tag. Enables the educational-value gate when it names a code language. */
  Lang?: string;
  /** Educational weights; default: Data/EduWeights.json, or the built-in prior when it is absent. */
  EduWeights?: EduWeights;
};

/** How many of the most negative features the low-educational-value reason names. */
const EduReasonFeatureCount = 2;

type QualityVerdict = { Score: number; RejectReason: string | null };

/** Structural gate, then (for code) the educational gate. RejectReason null = passed both. */
function JudgeQuality(Content: string, Origin: Origin, Options: ClassifyOptions): QualityVerdict {
  // curated sources (OASST / Wikipedia / books) are natural-language PROSE — score them as text so the
  // code-minification line-length heuristics don't wrongly reject detailed (long-line) dialogue/articles.
  const Structural = Origin === "curated" ? ScoreTextQuality(Content) : ScoreCodeQuality(Content);
  if (!Structural.Passed) return { Score: Structural.Score, RejectReason: `low quality: ${Structural.Reasons.join("; ")}` };
  const CodeLang = Origin === "curated" ? undefined : Options.Lang;
  const LargeEnough = Buffer.byteLength(Content, "utf8") >= DefaultFileFilter.MinBytes;
  if (CodeLang === undefined || !IsCodeLang(CodeLang) || !LargeEnough) return { Score: Structural.Score, RejectReason: null };

  const Weights = Options.EduWeights ?? ActiveEduWeights();
  const Edu = ScoreEducational(Content, CodeLang, Weights);
  if (Edu.Score >= Weights.Threshold) return { Score: Edu.Score, RejectReason: null };
  const Worst = TopNegativeContributions(Edu, EduReasonFeatureCount);
  const Named = Worst.length === 0 ? "no single dominant feature" : Worst.map((C) => `${C.Name}=${C.Value.toFixed(2)} (${C.Contribution.toFixed(2)})`).join(", ");
  return { Score: Edu.Score, RejectReason: `low educational value: ${Named} (score ${Edu.Score.toFixed(2)} < ${Weights.Threshold})` };
}

/** The tier of a synthetic conversation (see the header). Malformed content scores 0. */
function ClassifySynthetic(License: string, Content: string): TierDecision {
  const Doc = DecodeChatDocument(Content);
  if (Doc === null) return { Tier: "Rejected", QualityScore: 0, RejectReason: "malformed synthetic conversation (not a shahd-chat-v1 document)" };
  const Quality = ScoreTextQuality(ChatDocumentText(Doc));
  const Parsed = ParseSyntheticLicense(License);
  if (Parsed === null) return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `synthetic license must be "<source>;teacher:<model>", got ${License}` };
  if (!IsAcceptedSyntheticSource(Parsed.Source)) return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `non-permissive source license: ${Parsed.Source}` };
  if (!Quality.Passed) return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `low quality: ${Quality.Reasons.join("; ")}` };
  return { Tier: "Filtered", QualityScore: Quality.Score, RejectReason: null };
}

export function ClassifyDocument(License: string, Content: string, Origin: Origin, Options: ClassifyOptions = {}): TierDecision {
  if (Origin === "synthetic") return ClassifySynthetic(License, Content);
  const Quality = JudgeQuality(Content, Origin, Options);
  if (Origin === "web-general") {
    return { Tier: "Raw", QualityScore: Quality.Score, RejectReason: "general web: isolated for inspection, not training-eligible" };
  }
  // Our own code (owned) or a whole dataset we explicitly vetted + approved (curated: OASST /
  // Wikipedia / public-domain books): the source-level license is already accepted, so it is not
  // re-checked per document (License is still recorded on the record for provenance) — only quality.
  const LicenseAccepted = Origin === "owned" || Origin === "curated" || IsPermissive(License);
  if (!LicenseAccepted) {
    return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `non-permissive license: ${License}` };
  }
  if (Quality.RejectReason !== null) return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: Quality.RejectReason };
  return { Tier: "Filtered", QualityScore: Quality.Score, RejectReason: null };
}
