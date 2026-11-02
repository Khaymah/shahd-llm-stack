// What the two training entry scripts share (TrainOnFoundry = pretraining, TrainSftChat = chat SFT, and
// TrainBpeMerges for the tokenizer sample) — one home instead of two drifting copies:
//   • the reserved special-token layouts;
//   • spec-driven corpus assembly: the Filtered tier sampled PER SLICE of the Data Spec (each slice's
//     Langs -> documents, shares from the slice weights), deterministic order;
//   • the persisted-tokenizer choice (Data/Tokenizer.<Version>.json vs a checkpoint's own merges);
//   • the epoch guard (tokens seen / unique tokens, the memorisation check);
//   • the resume / overwrite-guard / stop-file contract.
// No top-level side effects: the scripts call in, and Tests/TrainingCorpus.Test.ts drives the same
// functions against the in-memory document store.

import { existsSync } from "node:fs";
import { DefaultDataSpecPath, LoadDataSpec } from "../Foundry/DataSpec.ts";
import type { DataSpec } from "../Foundry/DataSpec.ts";
import { MaxPageLimit } from "../Foundry/DocumentStore.ts";
import type { DocumentStore } from "../Foundry/DocumentStore.ts";
import type { DataKind } from "../Foundry/DataKinds.ts";
import type { FoundryStores } from "../Foundry/FoundryStores.ts";
import { IsValidationDocument } from "../Brain/Data/DocumentSplit.ts";
import { LoadBpeModel, SameMerges, TokenizerFilePath } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import { FimTokens } from "../Brain/Data/FimReformat.ts";
import { ChatTokenList } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokenList } from "../Brain/Serving/ToolProtocol.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";

// ---- special-token layouts -------------------------------------------------------------------------

/** Chat/tool control tokens — the SFT vocabulary's specials (and reserved by pretraining too). */
export const ChatSpecials: readonly string[] = [...ChatTokenList, ...ToolTokenList];

/** Pretraining reserves EOS + the FIM sentinels + every chat/tool token up front, so a chat run can
 *  warm-start from the base with an IDENTICAL vocab layout (see TrainOnFoundry). Order is load-bearing:
 *  it fixes the special ids above the BPE range. */
export const PretrainSpecials: readonly string[] = [SpecialTokens.Eos, ...Object.values(FimTokens), ...ChatSpecials];

// ---- Data Spec ---------------------------------------------------------------------------------------

/** The kind tables a pretraining / tokenizer sample reads (prose kinds feed the "docs" slice). */
export const PretrainKinds: readonly DataKind[] = ["code", "knowledge", "books"];

const BytesPerMb = 1e6;

/**
 * The Data Spec named by --Spec, or the default file when DefaultWhenExists and it exists; null when no
 * spec applies. "none" disables the default explicitly. A named file that does not exist is an error.
 */
export function ResolveSpec(Arg: string, DefaultWhenExists: boolean, Exists: (Path: string) => boolean = existsSync): { Spec: DataSpec; Path: string } | null {
  if (Arg === "none") return null;
  if (Arg === "") return DefaultWhenExists && Exists(DefaultDataSpecPath) ? { Spec: LoadDataSpec(DefaultDataSpecPath), Path: DefaultDataSpecPath } : null;
  if (!Exists(Arg)) throw new Error(`--Spec=${Arg}: file not found`);
  return { Spec: LoadDataSpec(Arg), Path: Arg };
}

export type SliceBudget = { Slice: string; Langs: readonly string[]; Share: number; TargetMb: number };

/** Each slice's share of TotalMb: Weight (default TargetMb) / sum of weights. Sorted by slice name. */
export function SliceBudgets(Spec: DataSpec, TotalMb: number): SliceBudget[] {
  const Entries = Object.entries(Spec.Slices)
    .filter((Entry): Entry is [string, NonNullable<(typeof Entry)[1]>] => Entry[1] !== undefined)
    .sort(([A], [B]) => (A < B ? -1 : A > B ? 1 : 0));
  const Weights = Entries.map(([, Budget]) => Budget.Weight ?? Budget.TargetMb);
  const Total = Weights.reduce((A, W) => A + W, 0);
  if (Entries.length === 0 || Total <= 0) throw new Error("SliceBudgets: the spec declares no weighted slices");
  return Entries.map(([Slice, Budget], I) => ({ Slice, Langs: Budget.Langs, Share: Weights[I] / Total, TargetMb: (TotalMb * Weights[I]) / Total }));
}

/** Everything the spec asks for: the sum of the slices' TargetMb. */
export function SpecTotalMb(Spec: DataSpec): number {
  return Object.values(Spec.Slices).reduce((A, Budget) => A + (Budget?.TargetMb ?? 0), 0);
}

/** A named document store to sample from (one per kind table). */
export type CorpusSource = { Name: string; Store: Pick<DocumentStore, "ListSummaries" | "ContentByIds"> };

export function PretrainSources(Stores: FoundryStores): CorpusSource[] {
  return PretrainKinds.map((Kind) => ({ Name: Kind, Store: Stores.Kind(Kind) }));
}

export type MixedDocument = { Id: string; Slice: string; Source: string; Content: string };
export type SliceTally = { Mb: number; Docs: number; TargetMb: number };
export type SpecMix = { Docs: MixedDocument[]; PerSlice: Record<string, SliceTally>; Mb: number };

type SummaryRow = { Id: string; Bytes: number };

/** Ascending-id keyset pager over one (store, lang) pair of the Filtered tier. */
class LangCursor {
  private Rows: readonly SummaryRow[] = [];
  private Pos = 0;
  private AfterId = "";
  private Done = false;

  constructor(readonly Source: CorpusSource, private readonly Lang: string) {}

  async Head(): Promise<SummaryRow | null> {
    if (this.Pos < this.Rows.length) return this.Rows[this.Pos];
    if (this.Done) return null;
    const Page = await this.Source.Store.ListSummaries({ Filter: { Tier: "Filtered", Lang: this.Lang }, Limit: MaxPageLimit, AfterId: this.AfterId });
    // Stores clamp pages to their own ceiling; only an EMPTY page means the end.
    if (Page.Rows.length === 0) {
      this.Done = true;
      return null;
    }
    this.Rows = Page.Rows;
    this.Pos = 0;
    this.AfterId = Page.Rows[Page.Rows.length - 1].Id;
    return this.Rows[0];
  }

  Advance(): void {
    this.Pos++;
  }
}

/** Pick ids for one slice: a k-way merge of its (source, lang) cursors in ascending id order, until the
 *  byte budget is reached. Ids are content hashes, so ascending-id order is an unbiased, deterministic
 *  sample in which each lang appears in proportion to its share of the slice. */
async function PickSliceIds(Cursors: LangCursor[], BudgetBytes: number, ExcludeValidation: boolean, Taken: Set<string>): Promise<{ Picks: { Id: string; Source: CorpusSource }[]; Bytes: number }> {
  const Picks: { Id: string; Source: CorpusSource }[] = [];
  let Bytes = 0;
  while (Bytes < BudgetBytes) {
    let Best: LangCursor | null = null;
    let BestRow: SummaryRow | null = null;
    for (const Cursor of Cursors) {
      const Row = await Cursor.Head();
      if (Row !== null && (BestRow === null || Row.Id < BestRow.Id)) {
        Best = Cursor;
        BestRow = Row;
      }
    }
    if (Best === null || BestRow === null) break; // the slice ran out of documents
    Best.Advance();
    if (Taken.has(BestRow.Id) || (ExcludeValidation && IsValidationDocument(BestRow.Id))) continue;
    Taken.add(BestRow.Id);
    Picks.push({ Id: BestRow.Id, Source: Best.Source });
    Bytes += BestRow.Bytes;
  }
  return { Picks, Bytes };
}

/**
 * Sample TotalMb from the Filtered tier, split across the spec's slices by weight. Each slice reads the
 * documents whose Lang is one of its Langs, from every source, in ascending id order (deterministic: the
 * same store contents always yield the same sample, so a resumed run sees the same corpus). A slice
 * short of documents simply contributes less — PerSlice reports actual vs target. ExcludeValidation
 * drops held-out documents (the tokenizer sample); pretraining keeps them and splits by id later.
 */
export async function AssembleSpecMix(Spec: DataSpec, Sources: readonly CorpusSource[], TotalMb: number, ExcludeValidation: boolean): Promise<SpecMix> {
  const Docs: MixedDocument[] = [];
  const PerSlice: Record<string, SliceTally> = {};
  const Taken = new Set<string>();
  let TotalBytes = 0;
  for (const Budget of SliceBudgets(Spec, TotalMb)) {
    const Cursors = Sources.flatMap((Source) => Budget.Langs.map((Lang) => new LangCursor(Source, Lang)));
    const { Picks } = await PickSliceIds(Cursors, Budget.TargetMb * BytesPerMb, ExcludeValidation, Taken);
    let Bytes = 0;
    let Count = 0;
    for (const Source of Sources) {
      const Ids = Picks.filter((P) => P.Source === Source).map((P) => P.Id);
      for (let From = 0; From < Ids.length; From += MaxPageLimit) {
        const Chunk = Ids.slice(From, From + MaxPageLimit);
        const Contents = await Source.Store.ContentByIds(Chunk);
        for (const Id of Chunk) {
          const Content = Contents.get(Id);
          if (Content === undefined || Content.length === 0) continue; // deleted since listing
          Docs.push({ Id, Slice: Budget.Slice, Source: Source.Name, Content });
          Bytes += Buffer.byteLength(Content, "utf8");
          Count++;
        }
      }
    }
    PerSlice[Budget.Slice] = { Mb: RoundMb(Bytes), Docs: Count, TargetMb: RoundMb(Budget.TargetMb * BytesPerMb) };
    TotalBytes += Bytes;
  }
  return { Docs, PerSlice, Mb: RoundMb(TotalBytes) };
}

const MbDecimals = 3;
function RoundMb(Bytes: number): number {
  return Number((Bytes / BytesPerMb).toFixed(MbDecimals));
}

/** One log line per slice: "ts 12.300/15.000MB (4210 docs)". */
export function DescribeMix(Mix: SpecMix): string[] {
  return Object.entries(Mix.PerSlice).map(([Slice, T]) => `${Slice} ${T.Mb.toFixed(MbDecimals)}/${T.TargetMb.toFixed(MbDecimals)}MB (${T.Docs} docs)${T.Mb < T.TargetMb ? " — short of target" : ""}`);
}

// ---- persisted tokenizer --------------------------------------------------------------------------

/** The tokenizer file the train scripts reuse by default (when it exists). */
export const DefaultTokenizerPath = TokenizerFilePath("v1");
/** --Tokenizer=fresh: ignore any tokenizer file and learn merges from this run's corpus. */
export const FreshTokenizerArg = "fresh";

export type TokenizerFileChoice = { Path: string; Explicit: boolean } | null;

/** --Tokenizer resolution: explicit path (must exist) > the default file when present > none. */
export function ChooseTokenizerFile(Arg: string, Exists: (Path: string) => boolean = existsSync): TokenizerFileChoice {
  if (Arg === FreshTokenizerArg) return null;
  if (Arg === "") return Exists(DefaultTokenizerPath) ? { Path: DefaultTokenizerPath, Explicit: false } : null;
  if (!Exists(Arg)) throw new Error(`--Tokenizer=${Arg}: file not found (train one with "bun run train:bpe", or pass --Tokenizer=${FreshTokenizerArg})`);
  return { Path: Arg, Explicit: true };
}

export type InheritedMerges = { Merges: [number, number][]; From: "checkpoint" | "base"; Name: string };
export type MergeChoice = { Merges: [number, number][]; Origin: string };

/**
 * The merge table a run uses. A resumed (or warm-started) run ALWAYS keeps the merges its weights were
 * trained with — token ids must land on the same embedding rows. An EXPLICIT --Tokenizer that disagrees
 * with them is refused (silently ignoring it would hide a mistake); the implicit default file is just
 * noted. Otherwise: the tokenizer file, else merges trained on this run's corpus.
 */
export function PickMerges(Inherited: InheritedMerges | null, File: TokenizerFileChoice, Train: () => [number, number][]): MergeChoice {
  if (Inherited !== null) {
    if (File === null) return { Merges: Inherited.Merges, Origin: `reused from ${Inherited.From} "${Inherited.Name}"` };
    const FileMerges = LoadBpeModel(File.Path).Merges;
    if (SameMerges(FileMerges, Inherited.Merges)) return { Merges: Inherited.Merges, Origin: `reused from ${Inherited.From} "${Inherited.Name}" (= ${File.Path})` };
    if (File.Explicit) {
      throw new Error(
        `--Tokenizer=${File.Path} (${FileMerges.length} merges) disagrees with the ${Inherited.From} "${Inherited.Name}" (${Inherited.Merges.length} merges): ` +
          `its weights were trained with ITS merges, so a ${Inherited.From === "checkpoint" ? "resumed" : "warm-started"} run must keep them — drop --Tokenizer, or ` +
          (Inherited.From === "checkpoint" ? "start a new run (new --Name, or --Fresh)" : "train from scratch without --From"),
      );
    }
    return { Merges: Inherited.Merges, Origin: `reused from ${Inherited.From} "${Inherited.Name}" (the default ${File.Path} differs and is ignored)` };
  }
  if (File !== null) {
    const Model = LoadBpeModel(File.Path);
    return { Merges: Model.Merges, Origin: `loaded ${File.Path} (version ${Model.Version}, trained on ${Model.TrainedOn.SampleMb}MB)` };
  }
  return { Merges: Train(), Origin: "trained on this run's corpus" };
}

// ---- epoch guard ------------------------------------------------------------------------------------

export type EpochCheck = { UniqueTokens: number; TokensSeen: number; Epochs: number; Limit: number | null; MaxSteps: number | null; Ok: boolean; Message: string };

/**
 * TokensSeen = Steps x Batch x Context; Epochs = TokensSeen / UniqueTokens. The limit is AllowEpochs
 * when given (an explicit override), else MaxEpochs (the spec's memorisation guard), else none (report
 * only). MaxSteps is the largest step count that stays within the limit.
 */
export function CheckEpochs(Opts: { UniqueTokens: number; Steps: number; Batch: number; Context: number; MaxEpochs: number | null; AllowEpochs: number | null }): EpochCheck {
  const TokensPerStep = Opts.Batch * Opts.Context;
  const TokensSeen = Opts.Steps * TokensPerStep;
  const Epochs = Opts.UniqueTokens > 0 ? TokensSeen / Opts.UniqueTokens : Number.POSITIVE_INFINITY;
  const Limit = Opts.AllowEpochs ?? Opts.MaxEpochs;
  const MaxSteps = Limit === null ? null : Math.floor((Limit * Opts.UniqueTokens) / TokensPerStep);
  const Summary = `epochs = TokensSeen/UniqueTokens = ${TokensSeen.toLocaleString("en-US")}/${Opts.UniqueTokens.toLocaleString("en-US")} = ${Epochs.toFixed(2)} (${Opts.Steps} steps x batch ${Opts.Batch} x context ${Opts.Context})`;
  if (Limit === null) return { UniqueTokens: Opts.UniqueTokens, TokensSeen, Epochs, Limit, MaxSteps, Ok: true, Message: `${Summary}; no epoch limit (pass --Spec to enforce the spec's MaxEpochs)` };
  const Source = Opts.AllowEpochs !== null ? `--AllowEpochs=${Opts.AllowEpochs}` : `Spec.Training.MaxEpochs=${Opts.MaxEpochs}`;
  if (Epochs <= Limit) return { UniqueTokens: Opts.UniqueTokens, TokensSeen, Epochs, Limit, MaxSteps, Ok: true, Message: `${Summary} <= ${Source}` };
  return {
    UniqueTokens: Opts.UniqueTokens,
    TokensSeen,
    Epochs,
    Limit,
    MaxSteps,
    Ok: false,
    Message: `${Summary} exceeds ${Source} — the model would memorise the corpus. At most --Steps=${MaxSteps} fits; or add data, or pass --AllowEpochs=<n> to override deliberately.`,
  };
}

// ---- resume / overwrite guard / stop file -------------------------------------------------------

export type BpeTokenizerState = { Merges: [number, number][]; Specials: string[] | null };

/** The checkpoint's BPE tokenizer state, or null when it has none (or a non-BPE one). */
export function ReadBpeState(Ckpt: Checkpoint | null): BpeTokenizerState | null {
  const State = (Ckpt?.TokenizerState ?? null) as { Kind?: unknown; Merges?: unknown; Specials?: unknown } | null;
  if (State === null || State.Kind !== "Bpe" || !Array.isArray(State.Merges)) return null;
  return { Merges: State.Merges as [number, number][], Specials: Array.isArray(State.Specials) ? (State.Specials as string[]) : null };
}

/** The step a checkpoint was saved at (Meta.Step; 0 when absent). */
export function CheckpointStep(Ckpt: Checkpoint): number {
  return Number(Ckpt.Meta["Step"] ?? 0);
}

export type ResumePlan = { Ckpt: Checkpoint; Step: number; State: BpeTokenizerState };

/**
 * Resume detection + the overwrite guard (one contract for both scripts). A same-name checkpoint with a
 * matching architecture and tokenizer resumes when it is the same run (same MaxSteps — crash recovery)
 * or --Resume extends it to more steps. A same-name checkpoint that is NOT resumed is refused: a later
 * periodic save would silently replace a finished model — overwriting must be explicit (--Fresh).
 */
export function DecideResume(Opts: {
  Existing: Checkpoint | null;
  Name: string;
  Steps: number;
  ResumeFlag: boolean;
  RequireSpecials: boolean;
  ArchMatches: (Ckpt: Checkpoint, State: BpeTokenizerState) => boolean;
  What: string;
}): { Resume: ResumePlan | null; Refusal: string | null } {
  const Existing = Opts.Existing;
  if (Existing === null) return { Resume: null, Refusal: null };
  const State = ReadBpeState(Existing);
  const Done = CheckpointStep(Existing);
  if (State !== null && (!Opts.RequireSpecials || State.Specials !== null)) {
    const Match = Opts.ArchMatches(Existing, State) && (Opts.ResumeFlag || Existing.Config.Schedule.MaxSteps === Opts.Steps);
    if (Match && Done < Opts.Steps) return { Resume: { Ckpt: Existing, Step: Done, State }, Refusal: null };
  }
  const Refusal = Opts.ResumeFlag
    ? `cannot resume "${Opts.Name}": the saved ${Opts.What} (step ${Done}) has a different architecture/tokenizer or is already at/past the requested ${Opts.Steps} steps — pick a new name or raise Steps`
    : `model "${Opts.Name}" already exists (step ${Done}) — pick a NEW name, enable Resume to extend it, or pass --Fresh to overwrite it intentionally`;
  return { Resume: null, Refusal };
}

/** Graceful pause: a predicate polled once per step that turns true when the stop file appears. */
export function StopFilePredicate(StopFile: string): (() => boolean) | undefined {
  return StopFile === "" ? undefined : (): boolean => existsSync(StopFile);
}
