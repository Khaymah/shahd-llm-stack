// One-time license backfill (Phase 3b). GitHub's search API tags many repos "NOASSERTION" (its own
// matcher could not classify the LICENSE file), so the Foundry rejected every file from them on
// license grounds — even the ones that are actually MIT/BSD/Apache. This re-fetches each such repo's
// real LICENSE text, runs the strict local SpdxDetector, and — ONLY when the text is provably a
// single clean permissive license — promotes that repo's quality-passing docs from Rejected to
// Filtered. Copyleft / commercial / mixed / unresolved repos stay rejected (the legal guardrail).
// Scans EVERY per-kind table (documents_<kind>); each repo's LICENSE is fetched once even if its docs
// span several kinds. Dry-run by default; pass --Apply to write. Needs DATABASE_URL + GITHUB_TOKEN.
//
//   bun run Scripts/BackfillLicenses.ts            # dry run: report what WOULD change
//   bun run Scripts/BackfillLicenses.ts --Apply    # write the reclassification

import { DataKinds } from "../Foundry/DataKinds.ts";
import type { DataKind } from "../Foundry/DataKinds.ts";
import type { RepoSummary } from "../Foundry/DocumentStore.ts";
import { DetectSpdx } from "../Foundry/SpdxDetector.ts";
import { FetchRepoLicense } from "../Foundry/GitHubHttp.ts";
import { GitHubToken, ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

const Apply = ReadFlag("--Apply");
const MinQuality = 0.6; // matches ScoreCodeQuality's pass Threshold — promote only quality-passing docs
const Unresolved = "NOASSERTION";
const Token = GitHubToken();
const Stores = ResolveFoundryStores();

// Per kind: the NOASSERTION repos in that kind's table (aggregate only — no document content read).
const ByKind: { Kind: DataKind; Repos: RepoSummary[] }[] = [];
for (const Kind of DataKinds) {
  try {
    const Repos = await Stores.Kind(Kind).RepoSummaries({ License: Unresolved });
    if (Repos.length > 0) ByKind.push({ Kind, Repos });
  } catch (Caught) {
    console.log(`[${Kind}] skipped (${(Caught as Error).message})`); // table missing → run foundry:migrate
  }
}
const AllRepos = new Map<string, number>();
for (const { Repos } of ByKind) for (const R of Repos) AllRepos.set(R.Source, (AllRepos.get(R.Source) ?? 0) + R.Files);
console.log(`${Unresolved}: ${AllRepos.size} repos, ${[...AllRepos.values()].reduce((A, N) => A + N, 0)} docs across ${ByKind.length} kind tables. Mode: ${Apply ? "APPLY" : "DRY-RUN"}\n`);

// Resolve each repo's license ONCE: the verified permissive SPDX id, or null (keep rejected).
const Verdicts = new Map<string, string | null>();
for (const [Source, N] of AllRepos) {
  let Spdx: string | null = null;
  try {
    const Lic = await FetchRepoLicense(Source, Token);
    if (Lic === null) {
      console.log(`KEEP  ${String(N).padStart(5)}  ${Source}  (no LICENSE file)`);
    } else {
      const Det = DetectSpdx(Lic.Text);
      if (Det.Permissive && Det.Spdx !== null) Spdx = Det.Spdx;
      else console.log(`KEEP  ${String(N).padStart(5)}  ${Source}  (${Det.Note})`);
    }
  } catch (Caught) {
    console.log(`KEEP  ${String(N).padStart(5)}  ${Source}  (fetch error: ${(Caught as Error).message})`);
  }
  Verdicts.set(Source, Spdx);
}

let PromoteDocs = 0;
for (const { Kind, Repos } of ByKind) {
  for (const { Source, Files } of Repos) {
    const Spdx = Verdicts.get(Source) ?? null;
    if (Spdx === null) continue;
    if (Apply) {
      const { Promoted, KeptLowQuality } = await Stores.Kind(Kind).ReclassifyBySource(Source, Spdx, MinQuality);
      console.log(`PROMOTE ${String(Promoted).padStart(5)}  [${Kind}] ${Source}  -> ${Spdx}  (kept ${KeptLowQuality} low-quality rejected)`);
      PromoteDocs += Promoted;
    } else {
      console.log(`PROMOTE ${String(Files).padStart(5)}  [${Kind}] ${Source}  -> ${Spdx}  (dry-run; before quality gate)`);
      PromoteDocs += Files;
    }
  }
}
const PromoteRepos = [...Verdicts.values()].filter((V) => V !== null).length;
console.log(`\n${Apply ? "APPLIED" : "DRY-RUN"}: promoted ${PromoteRepos} repos (${PromoteDocs} docs${Apply ? "" : " before quality gate"}); kept ${Verdicts.size - PromoteRepos} repos rejected.`);
if (!Apply) console.log("Re-run with --Apply to write the changes.");
await Stores.Close();
