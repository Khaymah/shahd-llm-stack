// The SAFE generation entry point for product use. Enforces, in one controllable place:
//   1. Safety on the prompt (block harmful requests before spending any compute).
//   2. Performance/resource Limits (cap new tokens + context length).
//   3. Safety on the generated output.
// Raw Generate/CachedGenerate stay the bare mechanism; product/serving code calls these instead.
//
// Three entry points share the guard steps (DRY):
//   GuardedGenerate          — synchronous, prompt string in, prompt + completion text out.
//   GuardedGenerateIdsStream — the ids-level core: prompt IDS in, completion ids out, streaming. Serving
//                              (InferenceServer) and the dashboard agent loop (ModelHolder) build their
//                              prompts as ids (chat template, untrusted text base-encoded), so they enter
//                              here and never round-trip through a string re-encode.
//   GuardedGenerateStream    — string-prompt convenience over the ids core, emitting decoded deltas.
// Every path generates through the KV-cached decoder (CachedForward.ts), which matches the uncached
// Generate token-for-token at a fraction of the per-token cost.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { SamplingOptions } from "../Sampling/Sampler.ts";
import { CachedGenerate, CachedGenerateAsync } from "../Sampling/CachedForward.ts";
import { SafetyPolicy } from "./SafetyPolicy.ts";

/** Tail window (chars) for the incremental per-token safety scan: generously larger than the longest
 *  span any HarmfulRules pattern can match (every pattern uses bounded {0,N} gaps, well under 200 chars
 *  total), so re-scanning only the newly grown text plus this much overlap catches every match a
 *  full-string scan would, without re-scanning the whole growing completion on every emitted token. */
export const SafetyScanWindowChars = 400;

/** Tokens re-decoded behind the emitted frontier, so a multibyte char split across tokens decodes right. */
const ReDecodeTokens = 4;
const IncompleteChar = String.fromCharCode(0xfffd); // U+FFFD: a not-yet-complete multibyte char

/** Streaming detokenizer: O(window) per token instead of re-decoding everything generated so far.
 *  Keeps the already-decoded text and decodes only the last ReDecodeTokens committed tokens plus the
 *  new tail; a trailing U+FFFD (an incomplete byte-level char) is held back until its bytes arrive. */
export class IncrementalDecoder {
  Text = "";
  private Ids: number[] = [];
  private Read = 0; // tokens whose bytes are fully reflected in Text
  private Tok: Tokenizer;

  constructor(Tok: Tokenizer) {
    this.Tok = Tok;
  }

  /** Append one token; returns the full stable text decoded so far. */
  Push(Id: number): string {
    this.Ids.push(Id);
    const Start = Math.max(0, this.Read - ReDecodeTokens);
    const Before = this.Tok.Decode(this.Ids.slice(Start, this.Read));
    const After = this.Tok.Decode(this.Ids.slice(Start));
    if (After.endsWith(IncompleteChar)) return this.Text; // hold back until the char completes
    // The overlap normally decodes identically; if a merge changed it, fall back to one full decode.
    this.Text = After.startsWith(Before) ? this.Text + After.slice(Before.length) : this.Tok.Decode(this.Ids);
    this.Read = this.Ids.length;
    return this.Text;
  }
}

/** The generation seam: prompt ids in, all ids out, OnToken per new id (a throw aborts), ShouldStop
 *  polled each step. CachedGenerateAsync by default; tests inject stubs with the same contract. */
export type TokenGenerator = typeof CachedGenerateAsync;

export type GuardedStreamOptions = {
  /** The text EnforceInput vets instead of the decoded prompt ids — e.g. the raw chat transcript a
   *  server rendered the ids from, or the prompt string itself. The input guard always runs. */
  InputScanText?: string;
  /** Ids that END the completion (e.g. <|endofturn|>): never decoded, vetted or delivered; generation
   *  stops right after one, and it is returned as the LAST completion id so callers can tell a natural
   *  stop from a length cap or an abort. */
  StopIds?: ReadonlySet<number>;
  /** Vetted decoded text deltas, in order; their concatenation is the decoded completion (stop id excluded). */
  OnDelta?: (Delta: string) => void;
  Generator?: TokenGenerator;
};

/** The context the model is fed: the trailing Config.Limits.MaxContextTokens prompt ids. */
export function CapPromptIds(PromptIds: readonly number[], Config: ResolvedConfig): number[] {
  return PromptIds.slice(Math.max(0, PromptIds.length - Config.Limits.MaxContextTokens));
}

function CapNewTokens(MaxNewTokens: number, Config: ResolvedConfig): number {
  return Math.min(MaxNewTokens, Config.Limits.MaxNewTokens);
}

export function GuardedGenerate(
  Model: Shahd,
  Tokenizer: Tokenizer,
  Prompt: string,
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Config: ResolvedConfig,
): string {
  const Policy = new SafetyPolicy(Config);

  // 1. Safety on the prompt.
  Policy.EnforceInput(Prompt);

  // 2. Resource limits: cap generation length and context.
  const PromptIds = CapPromptIds(Tokenizer.Encode(Prompt), Config);
  const GeneratedIds = CachedGenerate(Model, PromptIds, CapNewTokens(MaxNewTokens, Config), Options, Rng);
  const Text = Tokenizer.Decode(GeneratedIds);

  // 3. Safety on the output.
  Policy.EnforceOutput(Text);

  return Text;
}

/**
 * The ids-level guarded stream. Applies the three guards around CachedGenerateAsync and resolves to the
 * COMPLETION ids only (never the prompt; a terminating StopIds id is kept as the last element):
 *   1. EnforceInput on Guard.InputScanText, or on the decoded prompt ids — before any compute.
 *   2. Config.Limits caps: the trailing MaxContextTokens prompt ids, at most MaxNewTokens new tokens.
 *   3. EnforceOutput INCREMENTALLY: the text decoded so far (incrementally — no quadratic re-decode) is
 *      vetted, the new part plus a SafetyScanWindowChars overlap reaching back into the prompt, BEFORE
 *      Guard.OnDelta receives it and BEFORE OnToken fires for the id. Blocked content is therefore never
 *      delivered: the throw (SafetyBlockedError) aborts generation at the first offending token. A final
 *      unconditional vet covers the whole completion before the trailing held-back text is flushed.
 * OnToken fires once per vetted completion id (stop ids excluded). ShouldStop aborts early (e.g. the
 * client disconnected) without an error.
 */
export async function GuardedGenerateIdsStream(
  Model: Shahd,
  Tokenizer: Tokenizer,
  PromptIds: readonly number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Config: ResolvedConfig,
  OnToken: (Id: number) => void,
  ShouldStop?: () => boolean,
  Guard: GuardedStreamOptions = {},
): Promise<number[]> {
  const Policy = new SafetyPolicy(Config);
  Policy.EnforceInput(Guard.InputScanText ?? Tokenizer.Decode([...PromptIds])); // 1. before any compute

  const ContextIds = CapPromptIds(PromptIds, Config); // 2. resource limits
  // Only the prompt's tail can combine with the completion into a match the input scan didn't see.
  // Every token decodes to at least one char, so this many trailing ids always cover the scan window.
  const PromptTail = Tokenizer.Decode(ContextIds.slice(-SafetyScanWindowChars));
  const Decoder = new IncrementalDecoder(Tokenizer);
  const ContentIds: number[] = [];
  let StopId: number | null = null;
  let Emitted = 0;

  const OnGenerated = (Id: number): void => {
    if (StopId !== null) return; // a generator that ignores ShouldStop: nothing after the stop counts
    if (Guard.StopIds?.has(Id) === true) {
      StopId = Id;
      return;
    }
    ContentIds.push(Id);
    const Text = Decoder.Push(Id);
    if (Text.length > Emitted) {
      // 3. vet the new text plus a window of already-vetted context (prompt tail included) BEFORE delivery.
      const From = Emitted - SafetyScanWindowChars;
      Policy.EnforceOutput(From >= 0 ? Text.slice(From) : PromptTail.slice(From) + Text);
      Guard.OnDelta?.(Text.slice(Emitted));
      Emitted = Text.length;
    }
    OnToken(Id);
  };

  const Generator = Guard.Generator ?? CachedGenerateAsync;
  await Generator(Model, ContextIds, CapNewTokens(MaxNewTokens, Config), Options, Rng, OnGenerated, () => StopId !== null || ShouldStop?.() === true);

  const Completion = Tokenizer.Decode(ContentIds);
  Policy.EnforceOutput(PromptTail + Completion); // final, unconditional vet of the whole completion
  if (Completion.length > Emitted) Guard.OnDelta?.(Completion.slice(Emitted)); // flush a held-back tail
  return StopId === null ? ContentIds : [...ContentIds, StopId];
}

/**
 * Streaming sibling of GuardedGenerate for a prompt STRING: emits the completion as vetted decoded
 * deltas via OnDelta as tokens are produced, and returns the full completion (the NEW text only,
 * excluding the prompt). The guards are GuardedGenerateIdsStream's — the prompt string itself is what
 * the input guard vets, output is vetted before each delta is delivered, and a trailing U+FFFD from a
 * not-yet-complete multibyte token is held back until its bytes arrive. ShouldStop lets a caller abort
 * early (e.g. the client disconnected) so no compute is wasted generating for nobody.
 */
export async function GuardedGenerateStream(
  Model: Shahd,
  Tokenizer: Tokenizer,
  Prompt: string,
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Config: ResolvedConfig,
  OnDelta: (Delta: string) => void,
  ShouldStop?: () => boolean,
): Promise<string> {
  let Completion = "";
  const Deliver = (Delta: string): void => {
    Completion += Delta;
    OnDelta(Delta);
  };
  const NoOp = (): void => undefined;
  const PromptIds = Tokenizer.Encode(Prompt);
  await GuardedGenerateIdsStream(Model, Tokenizer, PromptIds, MaxNewTokens, Options, Rng, Config, NoOp, ShouldStop, {
    InputScanText: Prompt,
    OnDelta: Deliver,
  });
  return Completion; // the deltas concatenate to exactly the decoded completion (final flush included)
}
