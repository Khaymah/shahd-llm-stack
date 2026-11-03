// Ingest OUR OWN repositories (local directories) into the Foundry as 'owned' code — trained on
// regardless of license. Reads from the environment (Bun loads .env): DATABASE_URL (=> Postgres by
// default), FOUNDRY_REPOS (comma-separated roots), FOUNDRY_STORE.
//   bun run foundry:ingest-repos                          # uses .env (FOUNDRY_REPOS)
//   bun run foundry:ingest-repos --Repos=.,../client --Store=postgres

import { CreateLocalRepoProvider, IngestFromWeb, BuildReport, RenderReportText } from "../Foundry/FoundryBarrel.ts";
import { ResolveStore, RepoRoots } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

const Roots = RepoRoots();
const { Store, Kind } = ResolveStore();

const Provider = CreateLocalRepoProvider({
  Roots,
  License: ReadArg("--License=", "OWNED"),
  MinLevel: "medium",
  OnRepo: (Info) =>
    console.log(`  ${Info.Repo}: level=${Info.Assessment.Level} files=${Info.Assessment.FileCount} avgQ=${Info.Assessment.AvgQuality.toFixed(2)} bytes=${Info.Assessment.TotalBytes} -> ${Info.Ingested ? "INGESTED WHOLE" : "skipped"}`),
});

console.log(`store=${Kind} | own repos: ${Roots.join(", ")}`);
const Stats = await IngestFromWeb([Provider], [""], Store, new Date().toISOString());
console.log(`\ningested ${Stats.Ingested} files -> Filtered=${Stats.ByTier.Filtered} Rejected=${Stats.ByTier.Rejected}\n`);
console.log(RenderReportText(BuildReport(await Store.All())));
await Store.Close?.(); // release the Postgres pool, otherwise the open sockets keep the process alive forever
