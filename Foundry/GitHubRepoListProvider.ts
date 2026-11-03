// Spec-driven GitHub provider (Plan: Brain-Strengthening, phase 2 — "curate by spec, not by query").
// Instead of searching, it walks an EXPLICIT list of repositories (Data/Spec.json), downloads each
// tarball once through the shared GitHub http helpers, and keeps only the paths the spec asks for:
//   1. the spec's global PathGlobs (fixtures, snapshots, i18n, minified, …) drop a path outright,
//   2. the entry's Exclude globs drop it, its Include globs (when given) must match it,
//   3. the file's language must map to a slice the spec wants (and to the entry's Slice, when set),
//   4. raw size within [MinFileBytes, MaxFileBytes],
//   5. the built-in CodeFileFilter path gate (vendored/dist/dotfiles/config/lockfiles) — with the
//      curation opt-ins below — then PrepareRepoFile (per-file license header + content gate).
// Curation opt-ins: the built-in gate drops test, docs, example and migration directories because a
// QUERY-driven crawl cannot judge them. A spec can: test directories / `.test.` files are always
// opted back in (an entry drops them with Exclude — the spec wants ~10% tests), and doc/site/example/
// migration directories are opted back in only when the Include glob that matched names them
// literally (e.g. "docs/**"). Markdown (.md/.mdx) is collected as Lang "markdown" for the docs slice.

import { parseTarGzip } from "nanotar";
import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { HttpJson, FetchBytes, RepoLicense } from "./GitHubHttp.ts";
import { DefaultGitHubJson, DefaultGitHubBytes, FetchRepoLicense } from "./GitHubHttp.ts";
import { DetectSpdx } from "./SpdxDetector.ts";
import { PrepareRepoFile, DefaultRepoLimits, StripRoot } from "./RepoArchive.ts";
import { IsSubstantiveCodePath, LangForPath } from "./CodeFileFilter.ts";
import { ResolveFileLicense } from "./FileLicenseHeader.ts";
import { IsPermissive } from "../Brain/Data/LicenseManifest.ts";

/** One spec entry as the provider needs it (structurally compatible with DataSpecRepo). */
export type RepoListEntry = { Repo: string; Ref?: string; Include?: string[]; Exclude?: string[]; Slice?: string };

/** The spec's global file rules (structurally compatible with DataSpec["Exclusions"]). */
export type FileBounds = { PathGlobs: readonly string[]; MinFileBytes: number; MaxFileBytes: number };

/** What happened to one repository (one tarball) — the per-repo audit line. */
export type RepoListInfo = {
  Repo: string;
  Ref: string;
  License: string;
  Ingested: boolean;
  Files: number;
  Bytes: number;
  BySlice: Record<string, { Files: number; Bytes: number }>;
  Reason?: string;
};

export type RepoListOptions = {
  Token?: string;
  Http?: HttpJson; // repo metadata (default branch + license)
  FetchBytes?: FetchBytes; // repo tarball
  FetchLicense?: (FullName: string) => Promise<RepoLicense | null>; // NOASSERTION resolver
  Bounds?: FileBounds; // default: no extra globs, no minimum, the archive's per-file cap
  // Slice a language tag belongs to, or null when unwanted. Evaluated per file, so a caller can make a
  // slice "full" mid-run. Default: every known language is its own slice.
  SliceOfLang?: (Lang: string) => string | null;
  ShouldFetch?: (Entry: RepoListEntry) => boolean | Promise<boolean>; // budget/ledger gate, per entry
  OnRepoStart?: (Repo: string) => void;
  OnRepo?: (Info: RepoListInfo) => void;
  OnRepoReady?: RepoSink; // stream each repo to storage right after download (errors propagate)
  Log?: (Message: string) => void;
};

/** Lang tag stored for Markdown/MDX prose (LangForPath has no prose tags). */
export const MarkdownLang = "markdown";
const MarkdownExtension = /\.(md|mdx|markdown)$/i;

// Directory names the built-in gate drops that a spec may opt back in (see the header comment).
const TestSegments: ReadonlySet<string> = new Set(["test", "tests", "__tests__", "spec", "specs"]);
const NamedOptInSegments: ReadonlySet<string> = new Set(["doc", "docs", "site", "website", "example", "examples", "sample", "samples", "migration", "migrations"]);
const TestFileInfix = /\.(test|spec)(\.[a-z0-9]+)$/i;
// Placeholders that make an opted-in path look like ordinary source to the built-in gate, so every
// OTHER rule it applies (vendor, dist, dotfiles, config, lockfiles, generated) still runs unchanged.
const NeutralSegment = "_";
const NeutralCodeExtension = ".ts";
const SafeRef = /^[A-Za-z0-9._/-]+$/;
const SafeRepo = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const GitHubApi = "https://api.github.com";

const GlobCache = new Map<string, Bun.Glob>();
function CompiledGlob(Pattern: string): Bun.Glob {
  let Glob = GlobCache.get(Pattern);
  if (Glob === undefined) {
    Glob = new Bun.Glob(Pattern);
    GlobCache.set(Pattern, Glob);
  }
  return Glob;
}

/** The first glob (Bun.Glob syntax: `**`, `*`, `?`, `{a,b}`) matching a repo-relative path, or null. */
export function FirstMatchingGlob(Path: string, Globs: readonly string[]): string | null {
  for (const Pattern of Globs) if (CompiledGlob(Pattern).match(Path)) return Pattern;
  return null;
}

/** Lang tag for a repo path: "markdown" for .md/.mdx, else CodeFileFilter's tag ("unknown" if none). */
export function ListedLangForPath(Path: string): string {
  return MarkdownExtension.test(Path) ? MarkdownLang : LangForPath(Path);
}

/** Rewrite a path so CodeFileFilter judges only what the spec cannot opt back in (header comment). */
function GatePath(Path: string, MatchedInclude: string | null): string {
  const Named = new Set((MatchedInclude ?? "").toLowerCase().split(/[^a-z0-9_]+/));
  const Segments = Path.split("/");
  const Gated = Segments.map((Segment, Index) => {
    if (Index === Segments.length - 1) return Segment; // the file name itself is never rewritten
    const Lower = Segment.toLowerCase();
    return TestSegments.has(Lower) || (NamedOptInSegments.has(Lower) && Named.has(Lower)) ? NeutralSegment : Segment;
  });
  return Gated.join("/").replace(TestFileInfix, "$2").replace(MarkdownExtension, NeutralCodeExtension);
}

/**
 * Path-level selection for one spec entry: the slice + lang the file would be stored under, or null
 * when the spec does not want it. Size and content gates run later (they need the bytes).
 */
export function SelectListedPath(
  Path: string,
  Entry: RepoListEntry,
  PathGlobs: readonly string[],
  SliceOfLang: (Lang: string) => string | null,
): { Lang: string; Slice: string } | null {
  if (FirstMatchingGlob(Path, PathGlobs) !== null) return null;
  if (Entry.Exclude !== undefined && FirstMatchingGlob(Path, Entry.Exclude) !== null) return null;
  const MatchedInclude = Entry.Include === undefined ? null : FirstMatchingGlob(Path, Entry.Include);
  if (Entry.Include !== undefined && MatchedInclude === null) return null;
  const Lang = ListedLangForPath(Path);
  if (Lang === "unknown") return null;
  const Slice = SliceOfLang(Lang);
  if (Slice === null || (Entry.Slice !== undefined && Entry.Slice !== Slice)) return null;
  if (!IsSubstantiveCodePath(GatePath(Path, MatchedInclude))) return null;
  return { Lang, Slice };
}

// GitHub reports spdx_id "NOASSERTION" (or nothing) for a LICENSE its matcher can't classify. Read the
// real LICENSE text and, ONLY when it is provably one clean permissive template, use the detected id;
// anything ambiguous keeps its label (and is therefore not collected). Failures are non-fatal.
export async function ResolvePermissiveLicense(
  FullName: string,
  Current: string,
  Fetch: (Name: string) => Promise<RepoLicense | null>,
  Log: (Message: string) => void,
): Promise<string> {
  try {
    const License = await Fetch(FullName);
    if (License === null) return Current;
    const Detected = DetectSpdx(License.Text);
    if (Detected.Permissive && Detected.Spdx !== null) {
      Log(`[spec] ${FullName}: license ${Current} -> ${Detected.Spdx} (verified permissive from LICENSE)`);
      return Detected.Spdx;
    }
    Log(`[spec] ${FullName}: license stays ${Current} (${Detected.Note})`);
    return Current;
  } catch (Caught) {
    Log(`[spec] ${FullName}: license resolve failed: ${(Caught as Error).message}`);
    return Current;
  }
}

type RepoMeta = { default_branch?: string; license?: { spdx_id?: string } | null };
type RepoGroup = { Repo: string; Ref: string | undefined; Entries: RepoListEntry[] };

/** Entries of the same repo+ref share ONE tarball download; order follows the spec. */
function GroupEntries(Entries: readonly RepoListEntry[]): RepoGroup[] {
  const Groups = new Map<string, RepoGroup>();
  for (const Entry of Entries) {
    const Key = `${Entry.Repo}@${Entry.Ref ?? ""}`;
    const Group = Groups.get(Key) ?? { Repo: Entry.Repo, Ref: Entry.Ref, Entries: [] };
    Group.Entries.push(Entry);
    Groups.set(Key, Group);
  }
  return [...Groups.values()];
}

/** "" / "*" = every repo; otherwise a ","/";"-separated owner/name list narrowing the spec list. */
function RepoFilter(Query: string): (Repo: string) => boolean {
  const Wanted = Query.split(/[;,\n]+/).map((Q) => Q.trim().toLowerCase()).filter((Q) => Q.length > 0 && Q !== "*");
  return Wanted.length === 0 ? (): boolean => true : (Repo: string): boolean => Wanted.includes(Repo.toLowerCase());
}

type ExtractContext = {
  Repo: string;
  Ref: string;
  License: string;
  Entries: readonly RepoListEntry[];
  Bounds: FileBounds;
  SliceOfLang: (Lang: string) => string | null;
};

/** Walk one tarball and turn every file the spec wants into a SourceInput (first matching entry wins). */
async function ExtractListedFiles(Gz: Uint8Array, Context: ExtractContext): Promise<{ Inputs: SourceInput[]; Info: RepoListInfo; Conflicts: number }> {
  // Path + size selection runs in nanotar's filter, so unwanted entries (most of a big monorepo) are
  // never materialised; the decision is remembered per entry name for the content pass below.
  const Selection = new Map<string, { Lang: string; Slice: string }>();
  const Items = await parseTarGzip(Gz, {
    filter: (Meta) => {
      if (Meta.type !== "file" || Meta.size < Context.Bounds.MinFileBytes || Meta.size > Context.Bounds.MaxFileBytes) return false;
      const Path = StripRoot(Meta.name);
      for (const Entry of Context.Entries) {
        const Selected = SelectListedPath(Path, Entry, Context.Bounds.PathGlobs, Context.SliceOfLang);
        if (Selected !== null) {
          Selection.set(Meta.name, Selected);
          return true;
        }
      }
      return false;
    },
  });
  const Inputs: SourceInput[] = [];
  const Info: RepoListInfo = { Repo: Context.Repo, Ref: Context.Ref, License: Context.License, Ingested: true, Files: 0, Bytes: 0, BySlice: {} };
  let Conflicts = 0;
  for (const Item of Items) {
    const Selected = Selection.get(Item.name);
    if (Selected === undefined) continue;
    const Path = StripRoot(Item.name);
    const File = PrepareRepoFile(Path, Item.text, { MinBytes: Context.Bounds.MinFileBytes, MaxBytes: Context.Bounds.MaxFileBytes, Lang: Selected.Lang });
    if (File === null) continue;
    const Decision = ResolveFileLicense(Context.License, File.Header);
    if (Decision.Conflict) Conflicts++;
    Inputs.push({
      Source: Context.Repo,
      License: Decision.License,
      Lang: Selected.Lang,
      Content: File.Content,
      Provenance: `${Context.Repo}@${Context.Ref}:${Path}`,
      Origin: "web-permissive",
    });
    const Bytes = Buffer.byteLength(File.Content, "utf8");
    const Slot = Info.BySlice[Selected.Slice] ?? { Files: 0, Bytes: 0 };
    Slot.Files++;
    Slot.Bytes += Bytes;
    Info.BySlice[Selected.Slice] = Slot;
    Info.Files++;
    Info.Bytes += Bytes;
  }
  return { Inputs, Info, Conflicts };
}

export function CreateGitHubRepoListProvider(List: readonly RepoListEntry[], Options: RepoListOptions = {}): WebProvider {
  const Http = Options.Http ?? DefaultGitHubJson(Options.Token);
  const Fetch = Options.FetchBytes ?? DefaultGitHubBytes(Options.Token);
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));
  const FetchLicense = Options.FetchLicense ?? ((Name: string): Promise<RepoLicense | null> => FetchRepoLicense(Name, Options.Token));
  const Bounds: FileBounds = Options.Bounds ?? { PathGlobs: [], MinFileBytes: 0, MaxFileBytes: DefaultRepoLimits.MaxContentBytes };
  const SliceOfLang = Options.SliceOfLang ?? ((Lang: string): string | null => Lang);
  const Skip = (Repo: string, Ref: string, License: string, Reason: string): void => {
    Log(`[spec] skip ${Repo}: ${Reason}`);
    Options.OnRepo?.({ Repo, Ref, License, Ingested: false, Files: 0, Bytes: 0, BySlice: {}, Reason });
  };

  return {
    Name: "github-repo-list",
    Semantics: "bounded", // a fixed list: once every entry is collected, re-running adds nothing
    // Query narrows the list (see RepoFilter); Limit caps the number of repositories looked at.
    Fetch: async (Query: string, Limit: number, Signal?: AbortSignal): Promise<SourceInput[]> => {
      const Wanted = RepoFilter(Query);
      const Out: SourceInput[] = [];
      const Totals = { Looked: 0, Stored: 0, Files: 0, Skipped: 0, Errored: 0 };
      for (const Group of GroupEntries(List.filter((Entry) => Wanted(Entry.Repo)))) {
        if (Totals.Looked >= Limit || Signal?.aborted === true) break;
        Totals.Looked++;
        const Pending: RepoListEntry[] = [];
        for (const Entry of Group.Entries) if ((await Options.ShouldFetch?.(Entry)) ?? true) Pending.push(Entry);
        if (Pending.length === 0) {
          Totals.Skipped++;
          continue; // nothing left to collect from this repo — no download
        }
        // Both end up in an api.github.com URL path: refuse "."/".." segments and odd ref characters.
        const DotSegment = (Value: string): boolean => Value.split("/").some((Part) => Part === "." || Part === "..");
        const UnsafeRepo = !SafeRepo.test(Group.Repo) || DotSegment(Group.Repo);
        if (UnsafeRepo || (Group.Ref !== undefined && (!SafeRef.test(Group.Ref) || DotSegment(Group.Ref)))) {
          Totals.Errored++;
          Skip(Group.Repo, Group.Ref ?? "", "unknown", `unsafe repo/ref "${Group.Repo}@${Group.Ref ?? ""}"`);
          continue;
        }
        let Result: { Inputs: SourceInput[]; Info: RepoListInfo; Conflicts: number };
        // One bad repo (rate limit, oversized tarball, renamed repo) must not end the run: download and
        // extraction errors are logged per repo. Storage (OnRepoReady) runs OUTSIDE this guard so a
        // broken store / exceeded failure budget still aborts the whole run loudly.
        try {
          const Meta = (await Http(`${GitHubApi}/repos/${Group.Repo}`, Signal)) as RepoMeta;
          const Ref = Group.Ref ?? Meta.default_branch ?? "HEAD";
          const Reported = Meta.license?.spdx_id ?? "unknown";
          const License = Reported === "NOASSERTION" || Reported === "unknown" ? await ResolvePermissiveLicense(Group.Repo, Reported, FetchLicense, Log) : Reported;
          if (!IsPermissive(License)) {
            Totals.Skipped++;
            Skip(Group.Repo, Ref, License, `repo license ${License} is not permissive — fix the spec`);
            continue;
          }
          Options.OnRepoStart?.(Group.Repo);
          Log(`[spec] downloading ${Group.Repo}@${Ref} (license ${License}, entries ${Pending.map((E) => E.Slice ?? "*").join(",")})…`);
          const Gz = await Fetch(`${GitHubApi}/repos/${Group.Repo}/tarball/${Ref}`, Signal);
          Result = await ExtractListedFiles(Gz, { Repo: Group.Repo, Ref, License, Entries: Pending, Bounds, SliceOfLang });
        } catch (Caught) {
          Totals.Errored++;
          Skip(Group.Repo, Group.Ref ?? "", "unknown", `error: ${(Caught as Error).message}`);
          continue;
        }
        const Slices = Object.entries(Result.Info.BySlice).map(([Slice, S]) => `${Slice}=${S.Files}f/${(S.Bytes / 1024).toFixed(0)}KB`);
        Log(`[spec] ${Group.Repo}: ${Result.Info.Files} files (${Slices.join(" ") || "none matched — check Include"})`);
        if (Result.Conflicts > 0) Log(`[spec] ${Group.Repo}: ${Result.Conflicts} file(s) with a conflicting license header -> rejected (file_license_conflict)`);
        Options.OnRepo?.(Result.Info);
        if (Options.OnRepoReady !== undefined) await Options.OnRepoReady(Group.Repo, Result.Inputs);
        else Out.push(...Result.Inputs);
        Totals.Stored++;
        Totals.Files += Result.Inputs.length;
      }
      Log(`[spec] run summary: looked=${Totals.Looked} stored=${Totals.Stored} (${Totals.Files} files) skipped=${Totals.Skipped} errored=${Totals.Errored}`);
      return Out;
    },
  };
}
