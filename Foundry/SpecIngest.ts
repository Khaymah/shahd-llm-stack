// Spec-driven collection (Plan: Brain-Strengthening, phase 2). IngestSpec walks Data/Spec.json's repo
// list through the GitHubRepoListProvider and stores each repo as soon as it is downloaded, while
// tracking every slice's progress against its TargetMb budget:
//   - progress = Filtered-tier bytes per slice, measured from the store (RepoSummaries filtered by
//     tier + lang — Stats().ByLang only carries document COUNTS, not bytes);
//   - a slice is "full" once its Filtered bytes reach TargetMb; it is re-measured after every repo, so
//     no further repo is downloaded for it and files of a full slice are no longer kept from mixed
//     repos (a repo already in progress is finished — a slice can overshoot by at most one repo);
//   - an entry is "done" once the store holds rows of that repo in the entry's slice languages; done
//     entries are skipped unless Force.
// MeasureSpec / PlanSpec / RenderSpecStatus / RenderSpecPlan are pure reads for --Status / --DryRun.

import type { DataSpec, DataSpecRepo } from "./DataSpec.ts";
import { SliceForLang } from "./DataSpec.ts";
import type { DocumentStore } from "./DocumentStore.ts";
import { IngestDocuments, IngestFailureBudget } from "./Ingest.ts";
import type { IngestStats, SourceInput } from "./Ingest.ts";
import { CreateGitHubRepoListProvider } from "./GitHubRepoListProvider.ts";
import type { RepoListEntry, RepoListInfo, RepoListOptions } from "./GitHubRepoListProvider.ts";

/** TargetMb is in MiB: 1 MB here = 2^20 bytes of stored (license-stripped, sanitized) content. */
export const BytesPerMb = 1024 * 1024;
const DefaultEmbeddingDim = 256;
const PercentScale = 100;

export type SliceProgress = { Slice: string; TargetBytes: number; CollectedBytes: number; Files: number; EntriesDone: number; EntriesTotal: number };
export type SpecProgress = { Slices: SliceProgress[]; DoneEntries: ReadonlySet<string> };

export type SpecFilter = { Slice?: string; Repo?: string; Force?: boolean };
export type PlanAction = "fetch" | "done" | "slice-full" | "filtered";
export type PlanRow = { Entry: DataSpecRepo; Action: PlanAction };

export type SpecIngestOptions = SpecFilter & {
  IngestedAt: string; // ISO timestamp (supplied by the caller — keeps ingestion pure/testable)
  EmbeddingDim?: number;
  Provider?: Omit<RepoListOptions, "Bounds" | "SliceOfLang" | "ShouldFetch" | "OnRepoReady">; // http/token/log injection
  Budget?: IngestFailureBudget; // one run-wide failure budget across every repo
  Log?: (Message: string) => void;
  Signal?: AbortSignal;
};

export type SpecIngestResult = { Stats: IngestStats; Repos: RepoListInfo[]; Before: SpecProgress; After: SpecProgress; Plan: PlanRow[] };

/** Ledger key of one spec entry: a repo may feed several slices through separate entries. */
export function EntryKey(Entry: { Repo: string; Slice?: string }): string {
  return `${Entry.Repo}#${Entry.Slice ?? "*"}`;
}

function SliceNames(Spec: DataSpec): string[] {
  return Object.keys(Spec.Slices);
}

/** The language tags an entry can collect: its slice's Langs, or every spec language when unsliced. */
function EntryLangs(Spec: DataSpec, Slice: string | undefined): string[] {
  const Budgets = Object.entries(Spec.Slices).filter(([Name]) => Slice === undefined || Name === Slice);
  return Budgets.flatMap(([, Budget]) => Budget.Langs);
}

function TargetBytes(Spec: DataSpec, Slice: string): number {
  const Budget = Object.entries(Spec.Slices).find(([Name]) => Name === Slice)?.[1];
  return Budget === undefined ? 0 : Math.round(Budget.TargetMb * BytesPerMb);
}

/** Filtered-tier bytes + files per slice, summed over the slice's languages. */
export async function MeasureSliceBytes(Spec: DataSpec, Store: DocumentStore): Promise<Map<string, { Bytes: number; Files: number }>> {
  const Out = new Map<string, { Bytes: number; Files: number }>();
  for (const [Slice, Budget] of Object.entries(Spec.Slices)) {
    const Total = { Bytes: 0, Files: 0 };
    for (const Lang of Budget.Langs) {
      for (const Repo of await Store.RepoSummaries({ Tier: "Filtered", Lang })) {
        Total.Bytes += Repo.Bytes;
        Total.Files += Repo.Files;
      }
    }
    Out.set(Slice, Total);
  }
  return Out;
}

/** Per-slice progress against TargetMb and the set of entries (EntryKey) already collected. */
export async function MeasureSpec(Spec: DataSpec, Store: DocumentStore): Promise<SpecProgress> {
  const Bytes = await MeasureSliceBytes(Spec, Store);
  // Any tier counts as "collected": a repo whose files were all Rejected was still fetched and judged.
  const SourcesByLang = new Map<string, Set<string>>();
  for (const Lang of EntryLangs(Spec, undefined)) {
    SourcesByLang.set(Lang, new Set((await Store.RepoSummaries({ Lang })).map((Repo) => Repo.Source)));
  }
  const DoneEntries = new Set<string>();
  for (const Entry of Spec.Repos) {
    if (EntryLangs(Spec, Entry.Slice).some((Lang) => SourcesByLang.get(Lang)?.has(Entry.Repo) === true)) DoneEntries.add(EntryKey(Entry));
  }
  const Slices = SliceNames(Spec).map((Slice): SliceProgress => {
    const Entries = Spec.Repos.filter((Entry) => Entry.Slice === Slice);
    return {
      Slice,
      TargetBytes: TargetBytes(Spec, Slice),
      CollectedBytes: Bytes.get(Slice)?.Bytes ?? 0,
      Files: Bytes.get(Slice)?.Files ?? 0,
      EntriesDone: Entries.filter((Entry) => DoneEntries.has(EntryKey(Entry))).length,
      EntriesTotal: Entries.length,
    };
  });
  return { Slices, DoneEntries };
}

/** Throws when --Slice / --Repo name something the spec does not contain (a typo must not no-op). */
export function ValidateSpecFilter(Spec: DataSpec, Filter: SpecFilter): void {
  if (Filter.Slice !== undefined && !SliceNames(Spec).includes(Filter.Slice)) {
    throw new Error(`--Slice=${Filter.Slice} is not a slice of the spec (have: ${SliceNames(Spec).join(", ")})`);
  }
  if (Filter.Repo !== undefined && !Spec.Repos.some((Entry) => Entry.Repo.toLowerCase() === Filter.Repo?.toLowerCase())) {
    throw new Error(`--Repo=${Filter.Repo} is not listed in the spec`);
  }
}

function IsFull(Progress: ReadonlyMap<string, number>, Spec: DataSpec, Slice: string): boolean {
  return (Progress.get(Slice) ?? 0) >= TargetBytes(Spec, Slice);
}

/** Slices an entry may still feed under the filter, given the current byte counts. */
function OpenSlices(Spec: DataSpec, Entry: { Slice?: string }, Filter: SpecFilter, Bytes: ReadonlyMap<string, number>): string[] {
  const Candidates = Entry.Slice !== undefined ? [Entry.Slice] : SliceNames(Spec);
  return Candidates.filter((Slice) => (Filter.Slice === undefined || Slice === Filter.Slice) && !IsFull(Bytes, Spec, Slice));
}

function BytesBySlice(Progress: SpecProgress): Map<string, number> {
  return new Map(Progress.Slices.map((Slice) => [Slice.Slice, Slice.CollectedBytes]));
}

/** What a run would do with each entry, in spec order (the --DryRun answer). */
export function PlanSpec(Spec: DataSpec, Progress: SpecProgress, Filter: SpecFilter): PlanRow[] {
  const Bytes = BytesBySlice(Progress);
  return Spec.Repos.map((Entry): PlanRow => {
    const InRepo = Filter.Repo === undefined || Entry.Repo.toLowerCase() === Filter.Repo.toLowerCase();
    const InSlice = Filter.Slice === undefined || Entry.Slice === undefined || Entry.Slice === Filter.Slice;
    if (!InRepo || !InSlice) return { Entry, Action: "filtered" };
    if (Filter.Force !== true && Progress.DoneEntries.has(EntryKey(Entry))) return { Entry, Action: "done" };
    if (OpenSlices(Spec, Entry, Filter, Bytes).length === 0) return { Entry, Action: "slice-full" };
    return { Entry, Action: "fetch" };
  });
}

function Mb(Bytes: number): string {
  return (Bytes / BytesPerMb).toFixed(1);
}

function Table(Header: string[], Rows: string[][]): string {
  const Widths = Header.map((Cell, Index) => Math.max(Cell.length, ...Rows.map((Row) => (Row[Index] ?? "").length)));
  const Line = (Row: string[]): string => Row.map((Cell, Index) => Cell.padEnd(Widths[Index] ?? 0)).join("  ").trimEnd();
  return [Line(Header), Line(Widths.map((Width) => "-".repeat(Width))), ...Rows.map(Line)].join("\n");
}

/** The --Status table: slice -> target / collected / % / entries done. */
export function RenderSpecStatus(Progress: SpecProgress): string {
  const Rows = Progress.Slices.map((Slice) => [
    Slice.Slice,
    Mb(Slice.TargetBytes),
    Mb(Slice.CollectedBytes),
    `${Slice.TargetBytes === 0 ? 0 : Math.min(PercentScale, Math.floor((Slice.CollectedBytes / Slice.TargetBytes) * PercentScale))}%`,
    String(Slice.Files),
    `${Slice.EntriesDone}/${Slice.EntriesTotal}`,
  ]);
  const Total = Progress.Slices.reduce((Sum, Slice) => ({ Target: Sum.Target + Slice.TargetBytes, Got: Sum.Got + Slice.CollectedBytes }), { Target: 0, Got: 0 });
  Rows.push(["total", Mb(Total.Target), Mb(Total.Got), `${Total.Target === 0 ? 0 : Math.floor((Total.Got / Total.Target) * PercentScale)}%`, "", ""]);
  return Table(["slice", "target MB", "collected MB", "%", "files", "repos done"], Rows);
}

/** The --DryRun listing: one line per entry with its planned action. */
export function RenderSpecPlan(Plan: readonly PlanRow[]): string {
  const Shown = Plan.filter((Row) => Row.Action !== "filtered");
  const Rows = Shown.map((Row) => [Row.Action, Row.Entry.Repo, Row.Entry.Slice ?? "*", Row.Entry.Ref ?? "(default)", (Row.Entry.Include ?? ["(all code paths)"]).join(" ")]);
  const Fetch = Shown.filter((Row) => Row.Action === "fetch").length;
  return `${Table(["action", "repo", "slice", "ref", "include"], Rows)}\n\n${Fetch} entr${Fetch === 1 ? "y" : "ies"} to fetch`;
}

function AddStats(Total: IngestStats, Part: IngestStats): void {
  Total.Ingested += Part.Ingested;
  Total.New += Part.New;
  Total.Duplicate += Part.Duplicate;
  Total.Failed += Part.Failed;
  Total.ByTier.Filtered += Part.ByTier.Filtered;
  Total.ByTier.Raw += Part.ByTier.Raw;
  Total.ByTier.Rejected += Part.ByTier.Rejected;
}

/** Collect every pending spec entry (see the header comment for the budget/skip rules). */
export async function IngestSpec(Spec: DataSpec, Store: DocumentStore, Options: SpecIngestOptions): Promise<SpecIngestResult> {
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));
  ValidateSpecFilter(Spec, Options);
  const Budget = Options.Budget ?? new IngestFailureBudget();
  const Before = await MeasureSpec(Spec, Store);
  const Plan = PlanSpec(Spec, Before, Options);
  const Entries: RepoListEntry[] = Plan.filter((Row) => Row.Action === "fetch").map((Row) => Row.Entry);
  if ((Spec.LocalFolders?.length ?? 0) > 0) Log(`[spec] note: ${Spec.LocalFolders?.length} LocalFolders entries are not collected by ingest-spec`);
  const Stats: IngestStats = { Ingested: 0, New: 0, Duplicate: 0, ByTier: { Filtered: 0, Raw: 0, Rejected: 0 }, Failed: 0 };
  const Repos: RepoListInfo[] = [];
  let Bytes = BytesBySlice(Before);

  const Provider = CreateGitHubRepoListProvider(Entries, {
    ...Options.Provider,
    Bounds: Spec.Exclusions,
    // Re-evaluated per file: a slice that fills up mid-run stops accepting files from later repos.
    SliceOfLang: (Lang: string): string | null => {
      const Slice = SliceForLang(Spec, Lang);
      return Slice !== null && OpenSlices(Spec, { Slice }, Options, Bytes).length > 0 ? Slice : null;
    },
    ShouldFetch: (Entry: RepoListEntry): boolean => {
      const Open = OpenSlices(Spec, Entry, Options, Bytes);
      if (Open.length === 0) Log(`[spec] skip ${EntryKey(Entry)}: slice budget reached`);
      return Open.length > 0;
    },
    OnRepo: (Info: RepoListInfo): void => {
      Repos.push(Info);
      Options.Provider?.OnRepo?.(Info);
    },
    OnRepoReady: async (Source: string, Files: SourceInput[]): Promise<void> => {
      AddStats(Stats, await IngestDocuments(Files, Store, Options.IngestedAt, Options.EmbeddingDim ?? DefaultEmbeddingDim, undefined, Budget));
      Bytes = new Map([...(await MeasureSliceBytes(Spec, Store))].map(([Slice, Total]) => [Slice, Total.Bytes]));
      const Line = [...Bytes].map(([Slice, Got]) => `${Slice}=${Mb(Got)}/${Mb(TargetBytes(Spec, Slice))}MB`).join(" ");
      Log(`[spec] stored ${Source}: ${Files.length} files | ${Line}`);
    },
  });
  await Provider.Fetch(Options.Repo ?? "", Entries.length, Options.Signal);
  const After = await MeasureSpec(Spec, Store);
  return { Stats, Repos, Before, After, Plan };
}
