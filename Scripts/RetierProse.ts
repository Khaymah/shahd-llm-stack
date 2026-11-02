// Re-tier curated PROSE that the code-quality filter wrongly Rejected (Phase 9 fix). Conversation /
// knowledge / books are natural language; the old tiering scored them with ScoreCodeQuality, whose
// line-length heuristics reject detailed (long-line) dialogue and articles as "minified?". This pass
// re-scores every Rejected curated doc with ScoreTextQuality (prose-aware) and PROMOTES the ones that
// now pass to Filtered (clearing the reject reason) via the store's explicit Retier path. It only ever
// promotes — never demotes — and only touches Origin="curated" rows, so it is safe to re-run.
//
// Streams: pages of summaries (no content) by id keyset, then the content of just that page — memory
// stays bounded by one page no matter how many Rejected rows a kind holds, and promoting rows mid-scan
// cannot make the keyset skip or repeat anything.
//
//   bun run Scripts/RetierProse.ts            # dry-run: report how many would be promoted
//   bun run Scripts/RetierProse.ts --Apply    # write the promotions to Postgres

import { ScoreTextQuality } from "../Brain/Data/QualityFilter.ts";
import { MaxPageLimit } from "../Foundry/DocumentStore.ts";
import type { DocumentStore } from "../Foundry/DocumentStore.ts";
import type { DataKind } from "../Foundry/DataKinds.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

const Apply = ReadFlag("--Apply");
const ProseKinds: DataKind[] = ["conversation", "knowledge", "books"];
const Stores = ResolveFoundryStores();

type KindTally = { Scanned: number; Promoted: number; Kept: number; Missing: number };

async function RetierKind(Store: DocumentStore): Promise<KindTally> {
  const Tally: KindTally = { Scanned: 0, Promoted: 0, Kept: 0, Missing: 0 };
  let Cursor = "";
  for (;;) {
    // Only curated rows are re-judged as text; anything else is left exactly as it is.
    const Page = await Store.ListSummaries({ Filter: { Tier: "Rejected", Origin: "curated" }, AfterId: Cursor, Limit: MaxPageLimit });
    if (Page.Rows.length === 0) break;
    Cursor = Page.Rows[Page.Rows.length - 1]!.Id;
    const Contents = await Store.ContentByIds(Page.Rows.map((R) => R.Id));
    for (const Row of Page.Rows) {
      Tally.Scanned++;
      const Content = Contents.get(Row.Id);
      if (Content === undefined) {
        Tally.Missing++; // deleted between the summary page and the content read
        continue;
      }
      const Q = ScoreTextQuality(Content);
      if (!Q.Passed) {
        Tally.Kept++;
        continue;
      }
      if (Apply && !(await Store.Retier(Row.Id, "Filtered", null, Q.Score))) {
        Tally.Missing++;
        continue;
      }
      Tally.Promoted++;
    }
  }
  return Tally;
}

let TotalPromoted = 0;
let TotalKept = 0;
try {
  for (const Kind of ProseKinds) {
    let Tally: KindTally;
    try {
      Tally = await RetierKind(Stores.Kind(Kind));
    } catch (Caught) {
      // A missing table (run foundry:migrate) or a mid-scan DB error; promotions already written stay.
      console.log(`[${Kind}] stopped: ${(Caught as Error).message}`);
      continue;
    }
    console.log(`[${Kind}] curated Rejected=${Tally.Scanned} -> promote ${Tally.Promoted}, keep ${Tally.Kept}${Tally.Missing > 0 ? `, vanished ${Tally.Missing}` : ""}`);
    TotalPromoted += Tally.Promoted;
    TotalKept += Tally.Kept;
  }
} finally {
  await Stores.Close();
}
console.log(`${Apply ? "APPLIED" : "DRY-RUN"}: promoted ${TotalPromoted}, kept ${TotalKept}.${Apply ? "" : " Re-run with --Apply to write."}`);
process.exit(0);
