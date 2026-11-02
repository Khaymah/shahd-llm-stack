// Memorization probe: how much of what the model writes is copied VERBATIM from its training text?
// Prompts are the first MemorizationPromptTokens tokens of documents from BOTH sides — held-out
// documents and training documents — each continued greedily for MemorizationNewTokens tokens. Every
// generated token n-gram (MemorizationNgram, the decontamination size) is then looked up in the
// training text:
//   memorization.ngram                 — fraction of ALL generated n-grams found verbatim in training
//                                        (lower is better);
//   memorization.heldout_vs_train_gap  — train-prompt rate minus held-out-prompt rate: how much MORE the
//                                        model regurgitates when cued with text it trained on than with
//                                        unseen text (generic boilerplate both share cancels out).
// Token-level (model ids, through the base vocabulary), unlike Decontamination's code-word n-grams: it
// asks what the MODEL emits, at the model's own granularity.
//
// The training text is streamed once: only the few generated n-grams are kept in memory and a rolling
// polynomial hash (mod 2^32) prefilters every corpus position, so scanning tens of MB never builds a
// set of the corpus' n-grams. Without a training-text source the probe is skipped (and says why).

import type { RunnableModel } from "../../Checkpoint/LoadRunnableModel.ts";
import type { Tokenizer } from "../../Tokenizer/TokenizerTypes.ts";
import type { MetricResult } from "./SuiteTypes.ts";
import { SkippedMetric } from "./SuiteMetrics.ts";
import type { SuiteSettings } from "./SuiteSettings.ts";
import type { HeldOutSet, TrainCorpus } from "./SuiteCorpora.ts";
import { CaseRng, CompleteIds, EncodeUntrusted, ErrorReason, SamplingAt } from "./SuiteGeneration.ts";

const NgramMetric = "memorization.ngram";
const GapMetric = "memorization.heldout_vs_train_gap";
// A prompt's first tokens are taken from a bounded prefix of the document (no token spans more than
// this many characters, so the prefix always holds the full prompt for any real document).
const PrefixCharsPerToken = 32;
const HashBase = 0x01000193; // FNV prime — any odd multiplier works for the rolling hash
// Seed offset that keeps probe sampling streams apart from the case streams (all greedy today, but
// a later temperature change must not silently reuse a case's stream).
const ProbeSeedOffset = 7_000_000;

type Side = "heldout" | "train";
type Probe = { Side: Side; PromptIds: number[] };

/** Up to Count prompts of PromptTokens tokens each, from documents spread evenly across Docs. */
export function PickPrompts(Docs: readonly string[], Count: number, PromptTokens: number, Tok: Tokenizer): number[][] {
  const Out: number[][] = [];
  const Stride = Math.max(1, Math.floor(Docs.length / Count));
  for (let I = 0; I < Docs.length && Out.length < Count; I += Stride) {
    const Ids = EncodeUntrusted(Tok, Docs[I].slice(0, PromptTokens * PrefixCharsPerToken));
    if (Ids.length >= PromptTokens) Out.push(Ids.slice(0, PromptTokens));
  }
  return Out;
}

/** Hash of Ids[From .. From+N) — the same value the rolling scan reaches at that window. */
function WindowHash(Ids: readonly number[], From: number, N: number): number {
  let H = 0;
  for (let J = From; J < From + N; J++) H = (Math.imul(H, HashBase) + Ids[J]) >>> 0;
  return H;
}

function SameWindow(A: readonly number[], AFrom: number, B: readonly number[], BFrom: number, N: number): boolean {
  for (let J = 0; J < N; J++) if (A[AFrom + J] !== B[BFrom + J]) return false;
  return true;
}

type Gram = { Ids: readonly number[]; From: number; Found: boolean };

/** Mark every gram that occurs verbatim in the token stream Ids (rolling-hash scan, exact confirm). */
export function MarkFound(Ids: readonly number[], ByHash: ReadonlyMap<number, Gram[]>, N: number): void {
  if (Ids.length < N) return;
  let Power = 1; // HashBase^(N-1) mod 2^32 — removes the outgoing token
  for (let J = 1; J < N; J++) Power = Math.imul(Power, HashBase) >>> 0;
  let H = WindowHash(Ids, 0, N);
  for (let I = 0; ; I++) {
    const Bucket = ByHash.get(H);
    if (Bucket !== undefined) for (const G of Bucket) if (!G.Found && SameWindow(G.Ids, G.From, Ids, I, N)) G.Found = true;
    if (I + N >= Ids.length) break;
    H = (Math.imul((H - Math.imul(Ids[I], Power)) >>> 0, HashBase) + Ids[I + N]) >>> 0;
  }
}

function Rate(Grams: readonly Gram[]): number {
  return Grams.length === 0 ? 0 : Grams.filter((G) => G.Found).length / Grams.length;
}

/** memorization.ngram and memorization.heldout_vs_train_gap (skipped without a training source). */
export async function MemorizationMetrics(Runnable: RunnableModel, HeldOut: HeldOutSet, Train: TrainCorpus | null, Settings: SuiteSettings): Promise<MetricResult[]> {
  if (Train === null || Train.Docs.length === 0) {
    const Why = "no training-text source: pass --TrainCorpus=<file> or set DATABASE_URL (store Filtered tier)";
    return [SkippedMetric(NgramMetric, "worse", Why), SkippedMetric(GapMetric, "worse", Why)];
  }
  const Tok = Runnable.Tokenizer;
  const N = Settings.MemorizationNgram;
  const PerSide = Settings.MemorizationPromptsPerSide;
  const Probes: Probe[] = [
    ...PickPrompts(HeldOut.Docs.map((D) => D.Content), PerSide, Settings.MemorizationPromptTokens, Tok).map((PromptIds) => ({ Side: "heldout" as const, PromptIds })),
    ...PickPrompts(Train.Docs, PerSide, Settings.MemorizationPromptTokens, Tok).map((PromptIds) => ({ Side: "train" as const, PromptIds })),
  ];
  const Grams: Record<Side, Gram[]> = { heldout: [], train: [] };
  const ByHash = new Map<number, Gram[]>();
  const Errors: Record<string, number> = {};
  for (let I = 0; I < Probes.length; I++) {
    let Ids: number[];
    try {
      Ids = await CompleteIds(Runnable, Probes[I].PromptIds, { MaxTokens: Settings.MemorizationNewTokens, Sampling: SamplingAt(0), Rng: CaseRng(Settings.Seed + ProbeSeedOffset, I, 0) });
    } catch (Caught) {
      const Reason = ErrorReason(Caught);
      Errors[Reason] = (Errors[Reason] ?? 0) + 1;
      continue;
    }
    for (let From = 0; From + N <= Ids.length; From++) {
      const G: Gram = { Ids, From, Found: false };
      Grams[Probes[I].Side].push(G);
      const H = WindowHash(Ids, From, N);
      ByHash.set(H, [...(ByHash.get(H) ?? []), G]);
    }
  }
  for (const Doc of Train.Docs) MarkFound(EncodeUntrusted(Tok, Doc), ByHash, N);

  const All = [...Grams.heldout, ...Grams.train];
  const TrainRate = Rate(Grams.train);
  const HeldOutRate = Rate(Grams.heldout);
  const Details = {
    Ngram: N,
    Probes: { HeldOut: Probes.filter((P) => P.Side === "heldout").length, Train: Probes.filter((P) => P.Side === "train").length },
    Ngrams: { HeldOut: Grams.heldout.length, Train: Grams.train.length },
    TrainPromptRate: TrainRate,
    HeldOutPromptRate: HeldOutRate,
    TrainCorpus: { Origin: Train.Origin, Note: Train.Note, Docs: Train.Docs.length, Bytes: Train.Bytes },
    HeldOut: HeldOut.Note,
    ...(Object.keys(Errors).length > 0 ? { GenerationErrors: Errors } : {}),
  };
  const Ngram: MetricResult = All.length === 0
    ? SkippedMetric(NgramMetric, "worse", "no probe produced an n-gram (documents shorter than the prompt, or every generation failed)")
    : { Name: NgramMetric, Value: Rate(All), Higher: "worse", N: All.length, Details };
  const Gap: MetricResult = Grams.train.length === 0 || Grams.heldout.length === 0
    ? SkippedMetric(GapMetric, "worse", "needs n-grams from BOTH held-out and training prompts")
    : { Name: GapMetric, Value: TrainRate - HeldOutRate, Higher: "worse", N: All.length, Details: { TrainPromptRate: TrainRate, HeldOutPromptRate: HeldOutRate } };
  return [Ngram, Gap];
}
