// Ingest from the web into the Foundry. Reads everything from the environment (Bun loads .env):
// DATABASE_URL (=> Postgres by default), GITHUB_TOKEN, BRAVE_API_KEY, FOUNDRY_QUERY, FOUNDRY_STORE.
// GitHub repos are ingested WHOLE (permissive -> training-eligible); general web search (if
// BRAVE_API_KEY is set) is isolated to the Raw tier. Gated by Config.Data.WebEnabled.
//   bun run foundry:ingest-web                 # uses .env
//   bun run foundry:ingest-web --Query="language:go stars:>2000" --Store=postgres

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { IngestFromWeb, CreateGitHubRepoProvider, CreateWebSearchProvider, BuildReport, RenderReportText } from "../Foundry/FoundryBarrel.ts";
import type { WebProvider, SearchBackend } from "../Foundry/FoundryBarrel.ts";
import { ResolveStore, Query, GitHubToken, BraveKey } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

// Brave Search backend (used only if BRAVE_API_KEY is set); swap for any search API by shape.
function BraveSearch(Key: string): SearchBackend {
  return async (Q, Limit) => {
    const Response = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(Q)}&count=${Limit}`, {
      headers: { "X-Subscription-Token": Key, Accept: "application/json" },
    });
    if (!Response.ok) throw new Error(`Brave ${Response.status}`);
    const Data = (await Response.json()) as { web?: { results?: { url: string; title: string }[] } };
    return (Data.web?.results ?? []).map((R) => ({ Url: R.url, Title: R.title }));
  };
}

const Config = LoadConfig({ UseCli: true, UseEnv: false });
if (!Config.Data.WebEnabled) {
  console.log("web ingestion disabled (Config.Data.WebEnabled=false) — nothing fetched.");
  process.exit(0);
}

const { Store, Kind } = ResolveStore();
const Providers: WebProvider[] = [
  CreateGitHubRepoProvider({
    Token: GitHubToken(),
    MinLevel: "medium",
    OnRepo: (Info) =>
      console.log(`  ${Info.Repo}: level=${Info.Assessment.Level} files=${Info.Assessment.FileCount} avgQ=${Info.Assessment.AvgQuality.toFixed(2)} -> ${Info.Ingested ? "INGESTED WHOLE" : "skipped"}`),
  }),
];
const Brave = BraveKey();
if (Brave !== undefined) Providers.push(CreateWebSearchProvider({ Search: BraveSearch(Brave) }));

const Queries = [Query("language:typescript license:mit stars:>500")];
console.log(`store=${Kind} | token=${GitHubToken() ? "yes" : "no (rate-limited)"} | providers=${Providers.map((P) => P.Name).join(",")} | query=${JSON.stringify(Queries[0])}\n`);

const Stats = await IngestFromWeb(Providers, Queries, Store, new Date().toISOString(), Number(ReadArg("--PerQuery=", "5")), Config.Data.EmbeddingDim);
console.log(`\ningested ${Stats.Ingested} files -> Filtered=${Stats.ByTier.Filtered} Raw=${Stats.ByTier.Raw} Rejected=${Stats.ByTier.Rejected}\n`);
console.log(RenderReportText(BuildReport(await Store.All())));
await Store.Close?.(); // release the Postgres pool, otherwise the open sockets keep the process alive forever
