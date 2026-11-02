// Learn-source routing (data engine): ONE home for "which kind table does a collect source land in",
// "what unit does its per-run cap count", "how is its persisted cursor parsed", and "which providers
// serve it". Pure functions plus a default provider factory — the Learn runner (LearnJob) takes the
// factory as an injected dependency, so tests swap in fake providers without touching the network.

import type { LearnSettings } from "./DashboardTypes.ts";
import { KindForProvider, type DataKind } from "./DataKinds.ts";
import type { CapUnit } from "./CollectionState.ts";
import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { RepoIngestInfo } from "./RepoQuality.ts";
import type { SourceInput } from "./Ingest.ts";
import { CreateHfParquetProvider, WikiDumpSource, StackExchangeSource } from "./HfParquetProvider.ts";
import { CreateOasstProvider, Oasst2Url } from "./OasstProvider.ts";
import { CreateWikipediaProvider } from "./WikipediaProvider.ts";
import { CreateGsmProvider } from "./GsmProvider.ts";
import { CreateLocalFolderProvider } from "./LocalFolderProvider.ts";
import { CreateGitHubRepoProvider } from "./GitHubRepoProvider.ts";
import { CreateLocalRepoProvider } from "./LocalRepoProvider.ts";

/** Route a collect run to the kind table for its source. The truth table lives in
 *  DataKinds.KindForProvider (tested there); this is the LearnSettings-shaped entry point. */
export function SourceKindFor(Settings: Pick<LearnSettings, "Source" | "Kind">): DataKind {
  return KindForProvider(Settings.Source, Settings.Kind);
}

/** The unit Settings.MaxRepos counts for a source, or null when its providers ignore the cap entirely
 *  (on-disk folders/repos are walked in full). GitHub caps REPOSITORIES; the dataset sources cap
 *  DOCUMENTS. Exhaustion (ComputeExhausted) must compare the run's output in this same unit. */
export function CapUnitFor(Source: LearnSettings["Source"]): CapUnit | null {
  switch (Source) {
    case "github":
    case "both":
      return "repos";
    case "local":
    case "folder":
      return null;
    case "oasst":
    case "oasst2":
    case "gsm8k":
    case "wikipedia":
    case "wikidump":
    case "stackexchange":
      return "documents";
  }
}

/** Parse a persisted parquet cursor ('{"Shard":N,"Offset":M}') into shard/offset, defaulting to the
 *  start on missing/garbage input so a corrupt ledger row can never crash a collect run. */
export function ParseShardCursor(Raw: string | undefined): { Shard: number; Offset: number } {
  try {
    const P = JSON.parse(Raw ?? "{}") as { Shard?: unknown; Offset?: unknown };
    const Shard = typeof P.Shard === "number" && P.Shard >= 0 ? Math.floor(P.Shard) : 0;
    const Offset = typeof P.Offset === "number" && P.Offset >= 0 ? Math.floor(P.Offset) : 0;
    return { Shard, Offset };
  } catch {
    return { Shard: 0, Offset: 0 };
  }
}

/** The callbacks a Learn run hands its providers. OnRepoReady resolves true only when the whole
 *  batch was stored (the parquet provider holds its cursor otherwise); OnCursor persists the cursor. */
export type ProviderHooks = {
  Kind: DataKind;
  PrevCursor: string | undefined; // the source's persisted cursor (ledger), if any
  GitHubToken: string | undefined;
  Skip: (Repo: string) => boolean;
  OnRepoStart: (Name: string) => void; // "working" signal + Stop boundary (throws on abort)
  OnRepo: (Info: RepoIngestInfo) => void;
  OnRepoReady: (Source: string, Files: SourceInput[]) => Promise<boolean>;
  OnCursor: (Shard: number, Offset: number) => Promise<void>;
};

export type ProviderFactory = (Settings: LearnSettings, Hooks: ProviderHooks) => WebProvider[];

/** The production provider set for a Learn run's source (the network/disk-backed implementations). */
export const CreateLearnProviders: ProviderFactory = (Settings, Hooks) => {
  const { OnRepoStart, OnRepo, Skip } = Hooks;
  // Providers other than parquet only need "stored or threw" — the void sink adapter drops the flag.
  const OnRepoReady: RepoSink = async (Source, Files) => {
    await Hooks.OnRepoReady(Source, Files);
  };
  switch (Settings.Source) {
    // General/text sources: Query is the language filter, MaxRepos is the max items to pull.
    case "oasst":
      return [CreateOasstProvider({ OnRepoStart, OnRepoReady })];
    case "oasst2":
      return [CreateOasstProvider({ Url: Oasst2Url, OnRepoStart, OnRepoReady })];
    case "wikipedia":
      return [CreateWikipediaProvider({ OnRepoStart, OnRepoReady })];
    case "gsm8k":
      return [CreateGsmProvider({ OnRepoStart, OnRepoReady })];
    case "folder":
      // Ingest every text file under the given folder(s) as the chosen kind — the on-disk path for a
      // downloaded corpus (e.g. all of Gutenberg). License defaults to public-domain (the books case).
      // NB: no MaxContentBytes override (the code cap is 512 KB — most books are bigger; the provider's
      // book-sized 20 MB default applies), and NO SkipRoot — per-folder skip would short-circuit the
      // whole folder on a second run (Source is the folder name), blocking the "add more books,
      // re-collect" workflow; content-hash dedup already skips books already stored, cheaply.
      return [
        CreateLocalFolderProvider({
          Roots: Settings.Repos,
          License: Settings.License,
          Lang: Hooks.Kind === "books" ? "text" : `text-${Hooks.Kind}`,
          OnRepoStart,
          OnRepoReady,
        }),
      ];
    case "wikidump":
    case "stackexchange": {
      // The two HF-parquet sources share the generic provider: resume from the persisted {Shard,
      // Offset} cursor; the provider reports the committed cursor after every stored batch.
      const Cursor = ParseShardCursor(Hooks.PrevCursor);
      return [
        CreateHfParquetProvider(Settings.Source === "wikidump" ? WikiDumpSource : StackExchangeSource, {
          StartShard: Cursor.Shard,
          StartOffset: Cursor.Offset,
          MaxPerRun: Settings.MaxRepos,
          OnRepoStart,
          OnRepoReady: Hooks.OnRepoReady,
          OnCursor: Hooks.OnCursor,
        }),
      ];
    }
    case "github":
    case "local":
    case "both": {
      const Providers: WebProvider[] = [];
      if (Settings.Source !== "local") {
        Providers.push(CreateGitHubRepoProvider({ Token: Hooks.GitHubToken, MinLevel: Settings.MinLevel, MaxFilesPerRepo: Settings.MaxFilesPerRepo, MaxBytesPerRepo: Settings.MaxBytesPerRepo, MaxContentBytesPerRepo: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
      }
      if (Settings.Source !== "github") {
        Providers.push(CreateLocalRepoProvider({ Roots: Settings.Repos, MinLevel: Settings.MinLevel, MaxFiles: Settings.MaxFilesPerRepo, MaxBytes: Settings.MaxBytesPerRepo, MaxContentBytes: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
      }
      return Providers;
    }
  }
};
