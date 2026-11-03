// Every numeric knob of the evaluation suite, with its default, in ONE place (rule #5), validated by a
// Zod schema that carries no defaults (defaults live only in SuiteDefaults below). The suite is a
// MEASUREMENT: once a baseline is recorded, changing a default here changes what the numbers mean —
// bump the case-set Version (or record a new baseline) when you do.
//
// Not part of the model ResolvedConfig on purpose: adding a section there would change every config
// hash (and so every checkpoint's identity) for a knob that never affects training.

import { z } from "zod";

export const SuiteSettingsSchema = z.object({
  /** Cap on cases PER CATEGORY (code / format / tools / multi-turn); 0 = every case. */
  MaxCases: z.number().int().min(0),
  /** Wall-clock budget for ONE case (all its generations + test runs); late work counts as a fail. */
  CaseTimeoutMs: z.number().int().min(1),
  /** Base seed every sampled generation is derived from (case index + sample index mixed in). */
  Seed: z.number().int().min(0),
  /** New tokens for a code answer. */
  CodeMaxTokens: z.number().int().min(1),
  /** New tokens for a format / multi-turn reply. */
  ReplyMaxTokens: z.number().int().min(1),
  /** New tokens for a tool-call turn. */
  ToolMaxTokens: z.number().int().min(1),
  /** Samples drawn for pass@k (k = this). */
  PassAtKSamples: z.number().int().min(1),
  /** Temperature of the pass@k samples (pass@1 is always greedy, temperature 0). */
  SampleTemperature: z.number().positive(),
  /** Held-out documents read per data kind (store source). */
  HeldOutDocsPerKind: z.number().int().min(1),
  /** Upper bound on store ids scanned per kind while collecting held-out documents. */
  HeldOutMaxScan: z.number().int().min(1),
  /** Upper bound on non-overlapping BlockSize windows scored per kind. */
  BpbMaxWindows: z.number().int().min(1),
  /** Memorization: prompts drawn from EACH side (held-out and training). */
  MemorizationPromptsPerSide: z.number().int().min(1),
  /** Memorization: prompt length in tokens. */
  MemorizationPromptTokens: z.number().int().min(1),
  /** Memorization: greedy continuation length in tokens. */
  MemorizationNewTokens: z.number().int().min(1),
  /** Memorization: token n-gram size counted as "verbatim from training". */
  MemorizationNgram: z.number().int().min(2),
  /** Cap on the training text scanned for verbatim n-grams. */
  TrainCorpusMaxBytes: z.number().int().min(1),
}).strict();

export type SuiteSettings = z.infer<typeof SuiteSettingsSchema>;

export const SuiteDefaults: Readonly<SuiteSettings> = Object.freeze({
  MaxCases: 0, // all cases — the frozen set is small enough to run whole
  CaseTimeoutMs: 60_000, // one minute per case covers 1 greedy + k samples + the test runs on CPU
  Seed: 1, // matches DefaultConfig.Training.Seed so a fresh checkout reproduces recorded numbers
  CodeMaxTokens: 256, // a short function plus fences
  ReplyMaxTokens: 128, // a chat reply that does not stop by then is a format failure
  ToolMaxTokens: 96, // `<|tool_call|>{"name":…,"arguments":{…}}<|end_tool_call|>` fits easily
  PassAtKSamples: 5, // pass@5
  SampleTemperature: 0.8, // the usual pass@k sampling temperature (Chen et al. use 0.8 for k>1)
  HeldOutDocsPerKind: 50, // enough tokens for a stable mean at the current model sizes
  HeldOutMaxScan: 20_000, // ~10% are held out, so this finds 50 docs with a wide margin
  BpbMaxWindows: 64, // 64 x BlockSize tokens per kind (8k tokens at ctx 128)
  MemorizationPromptsPerSide: 8, // 16 probes total
  MemorizationPromptTokens: 32,
  MemorizationNewTokens: 64,
  MemorizationNgram: 13, // the decontamination n-gram size (Brain/Data/Decontamination.ts)
  TrainCorpusMaxBytes: 50_000_000, // 50 MB of training text scanned for verbatim n-grams
});

/** Defaults overlaid with the caller's overrides, validated (throws on a bad value). */
export function ResolveSuiteSettings(Overrides: Partial<SuiteSettings> = {}): SuiteSettings {
  return SuiteSettingsSchema.parse({ ...SuiteDefaults, ...Overrides });
}

/** At most MaxCases cases (all when MaxCases is 0), STRATIFIED across slices: slices take turns in
 *  order of first appearance, each contributing its cases in file order — so a small cap still samples
 *  every slice instead of only the first one, and the selection is deterministic. */
export function CapCases<T extends { Slice: string }>(Cases: readonly T[], MaxCases: number): readonly T[] {
  if (MaxCases <= 0 || MaxCases >= Cases.length) return Cases;
  const BySlice = new Map<string, T[]>();
  for (const Case of Cases) BySlice.set(Case.Slice, [...(BySlice.get(Case.Slice) ?? []), Case]);
  const Queues = [...BySlice.values()];
  const Picked = new Set<T>();
  for (let Round = 0; Picked.size < MaxCases; Round++) {
    for (const Queue of Queues) if (Round < Queue.length && Picked.size < MaxCases) Picked.add(Queue[Round]);
  }
  return Cases.filter((Case) => Picked.has(Case)); // file order; metrics seed by the index in the FULL set
}
