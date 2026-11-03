// Whole-repo GitHub provider (M7): search repositories, then for each download the ENTIRE repo
// (tarball), assess its level, and — if it meets the minimum level — ingest EVERY substantive source
// file so the model learns the whole project, not a scattered sample. Repos below the minimum level
// are skipped (understand the level, learn from the good ones). Both fetchers are injected (mock in
// tests; real GitHub by default) — a token is recommended for the tarball rate limit.

import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { HttpJson, FetchBytes, RepoLicense } from "./GitHubHttp.ts";
import { DefaultGitHubJson, DefaultGitHubBytes, FetchRepoLicense } from "./GitHubHttp.ts";
import { FetchRepoFiles } from "./RepoArchive.ts";
import { AssessRepo, LevelRank, EmptyAssessment } from "./RepoQuality.ts";
import type { RepoLevel, RepoIngestInfo } from "./RepoQuality.ts";
import { LangForPath } from "./CodeFileFilter.ts";
import { ResolveFileLicense } from "./FileLicenseHeader.ts";
import { ResolvePermissiveLicense } from "./GitHubRepoListProvider.ts";

type RepoItem = { full_name: string; default_branch: string; license: { spdx_id: string } | null };
type SearchResult = { items?: RepoItem[] };

export type GitHubRepoOptions = {
  Token?: string;
  Http?: HttpJson; // repo search (JSON)
  FetchBytes?: FetchBytes; // repo tarball (bytes)
  MaxFilesPerRepo?: number; // cap so a monorepo can't dominate (generous by default)
  MaxBytesPerRepo?: number; // byte budget per repo
  MaxContentBytesPerRepo?: number; // per-file size cap
  MinLevel?: RepoLevel; // skip repos below this level
  SkipRepo?: (Repo: string) => boolean; // skip already-learned repos (before download)
  OnRepoStart?: (Repo: string) => void; // fired BEFORE a repo's tarball download (a "working" signal)
  OnRepo?: (Info: RepoIngestInfo) => void; // progress/reporting hook
  OnRepoReady?: RepoSink; // when set, each repo is streamed here (stored) right after download
  Log?: (Message: string) => void; // server-console trail (default console.log): every repo's fate
  FetchLicense?: (FullName: string) => Promise<RepoLicense | null>; // NOASSERTION resolver (injected in tests)
};

// GitHub search returns at most 100 results per page and 1000 total per query.
const SearchPerPage = 100;
const SearchMaxResults = 1000;


export function CreateGitHubRepoProvider(Options: GitHubRepoOptions = {}): WebProvider {
  const Http = Options.Http ?? DefaultGitHubJson(Options.Token);
  const Fetch = Options.FetchBytes ?? DefaultGitHubBytes(Options.Token);
  const Limits = { MaxFiles: Options.MaxFilesPerRepo, MaxBytes: Options.MaxBytesPerRepo, MaxContentBytes: Options.MaxContentBytesPerRepo };
  const MinLevel = Options.MinLevel ?? "medium";
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));
  const FetchLicense = Options.FetchLicense ?? ((Name: string): Promise<RepoLicense | null> => FetchRepoLicense(Name, Options.Token));

  return {
    Name: "github-repo",
    Semantics: "streaming", // different search queries keep surfacing fresh repos — the corpus can grow
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      // GROWTH LEVER: one GitHub query is capped at 1000 results, so re-running it (with SkipRepo) soon
      // yields only already-learned repos. Accept MULTIPLE queries separated by ";" (or newlines) and
      // run each — different queries surface different repos, so the code corpus grows well past 1000 in
      // a single run. A plain single query is unchanged (the list is just [Query]).
      const Queries = Query.split(/[;\n]+/).map((Q) => Q.trim()).filter((Q) => Q.length > 0);
      const EffectiveQueries = Queries.length > 0 ? Queries : [Query];

      // Run-wide counters (across all queries) so the run ends with one auditable summary. Remaining is
      // the shared MaxRepos budget: repos LOOKED AT this run, drawn from the queries in order.
      let SkippedLearned = 0;
      let SkippedLevel = 0;
      let Errored = 0;
      let Stored = 0;
      let StoredFiles = 0;
      let Remaining = Limit;
      const Out: SourceInput[] = [];

      for (const SubQuery of EffectiveQueries) {
        if (Remaining <= 0) break;
        // Paginate the search so MaxRepos beyond 100 is honored (up to GitHub's 1000-result cap). A
        // search failure (bad token / rate limit) is logged and stops pagination — we process whatever
        // pages already came back instead of throwing away the whole run.
        const Want = Math.min(Remaining, SearchMaxResults);
        const Repos: RepoItem[] = [];
        for (let Page = 1; Repos.length < Want && Page <= Math.ceil(SearchMaxResults / SearchPerPage); Page++) {
          const PerPage = Math.min(SearchPerPage, Want - Repos.length);
          try {
            const Search = (await Http(`https://api.github.com/search/repositories?q=${encodeURIComponent(SubQuery)}&per_page=${PerPage}&page=${Page}`)) as SearchResult;
            const Items = Search.items ?? [];
            Log(`[gh] search "${SubQuery}" page ${Page}: ${Items.length} repos`);
            Repos.push(...Items);
            if (Items.length < PerPage) break; // no more results
          } catch (Caught) {
            Log(`[gh] SEARCH ERROR "${SubQuery}" page ${Page}: ${(Caught as Error).message}`);
            break;
          }
        }
        Log(`[gh] search "${SubQuery}": ${Repos.length} repos found (want ${Want})`);

        for (const Repo of Repos) {
          if (Remaining <= 0) break; // MaxRepos budget spent — stop this and any further query
          Remaining--; // count every repo we look at against the shared budget
          const License = Repo.license?.spdx_id ?? "unknown";
          if (Options.SkipRepo?.(Repo.full_name) === true) {
            SkippedLearned++;
            Log(`[gh] skip (already learned): ${Repo.full_name}`);
            Options.OnRepo?.({ Repo: Repo.full_name, License, Assessment: EmptyAssessment, Ingested: false, Reason: "already learned" });
            continue; // don't re-download a repo we already ingested
          }
          Options.OnRepoStart?.(Repo.full_name); // signal work before download; THROWS on Stop (propagates)
          // One bad repo — a 403 secondary rate-limit, a giant tarball, a network blip — must NOT end the
          // whole run. Wrap the download/assess/ingest so a failure is logged and we move to the next repo.
          try {
            Log(`[gh] downloading ${Repo.full_name} (license ${License})…`);
            const Files = await FetchRepoFiles(`https://api.github.com/repos/${Repo.full_name}/tarball/${Repo.default_branch}`, Fetch, Limits);
            const Assessment = AssessRepo(Files);
            const Ingested = LevelRank[Assessment.Level] >= LevelRank[MinLevel];
            if (!Ingested) {
              Log(`[gh] ${Repo.full_name}: level=${Assessment.Level} files=${Assessment.FileCount} bytes=${Assessment.TotalBytes} -> skip (below ${MinLevel})`);
              Options.OnRepo?.({ Repo: Repo.full_name, License, Assessment, Ingested: false, Reason: `level ${Assessment.Level} below minimum ${MinLevel}` });
              SkippedLevel++;
              continue; // skip low-level repos entirely
            }
            // Only resolve an unclassifiable license for repos we will actually store (after the level
            // gate) — one extra /license call per NOASSERTION repo, not per searched repo.
            const StoreLicense = License === "NOASSERTION" || License === "unknown" ? await ResolvePermissiveLicense(Repo.full_name, License, FetchLicense, Log) : License;
            Log(`[gh] ${Repo.full_name}: level=${Assessment.Level} files=${Assessment.FileCount} bytes=${Assessment.TotalBytes} -> ingest (license ${StoreLicense})`);
            Options.OnRepo?.({ Repo: Repo.full_name, License: StoreLicense, Assessment, Ingested: true });
            // Per-file license: a copyleft/non-permissive header overrides the repo license (-> Rejected
            // with file_license_conflict); a differing permissive SPDX header becomes the file's license.
            let Conflicts = 0;
            const Inputs: SourceInput[] = Files.map((File) => {
              const Decision = ResolveFileLicense(StoreLicense, File.Header);
              if (Decision.Conflict) Conflicts++;
              return {
                Source: Repo.full_name,
                License: Decision.License,
                Lang: LangForPath(File.Path),
                Content: File.Content,
                Provenance: `https://github.com/${Repo.full_name}/blob/${Repo.default_branch}/${File.Path}`,
                Origin: "web-permissive",
              };
            });
            if (Conflicts > 0) Log(`[gh] ${Repo.full_name}: ${Conflicts} file(s) with a conflicting license header -> rejected (file_license_conflict)`);
            // Incremental: store this repo NOW (before downloading the next). Batch: collect for the caller.
            if (Options.OnRepoReady !== undefined) await Options.OnRepoReady(Repo.full_name, Inputs);
            else Out.push(...Inputs);
            Stored++;
            StoredFiles += Inputs.length;
            Log(`[gh] stored ${Repo.full_name}: ${Inputs.length} files`);
          } catch (Caught) {
            Errored++;
            const Message = (Caught as Error).message;
            Log(`[gh] ERROR ${Repo.full_name}: ${Message}`);
            Options.OnRepo?.({ Repo: Repo.full_name, License, Assessment: EmptyAssessment, Ingested: false, Reason: `error: ${Message}` });
          }
        }
      }
      Log(`[gh] run summary: queries=${EffectiveQueries.length} alreadyLearned=${SkippedLearned} belowLevel=${SkippedLevel} errored=${Errored} stored=${Stored} repos (${StoredFiles} files)`);
      return Out;
    },
  };
}
