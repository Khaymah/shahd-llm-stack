// Dump a random sample of Filtered documents for HAND LABELLING (the educational-value classifier,
// Plan: Brain-Strengthening §2.2 layer 3). Each slice (ts/js/html/css/go/sql/other, by the document's
// Lang) gets its own seeded reservoir sample, so rare slices are not drowned out by TypeScript.
// Output is JSONL with "label": null — set each to 1 ("teaches something") or 0, then run
// `bun run foundry:train-edu --Labels=<file>`.
//
// Streams: keyset pages of summaries (no content), reservoir-sampled per slice; only the chosen ids'
// content is read afterwards. Memory stays bounded by one page + the sample, whatever the table size.
// Refuses to overwrite an existing file (it may already hold your labels) unless --Force.
//
//   bun run Scripts/LabelSamples.ts [--PerSlice=50] [--Kind=code] [--Seed=7] [--Out=Runs/Labels/sample.jsonl] [--Force]

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { EduSliceForLang, EduSlices } from "../Brain/Data/EducationalFeatures.ts";
import type { EduSlice } from "../Brain/Data/EducationalFeatures.ts";
import { MaxPageLimit } from "../Foundry/DocumentStore.ts";
import type { DocumentStore, DocumentSummary } from "../Foundry/DocumentStore.ts";
import { IsDataKind } from "../Foundry/DataKinds.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag, ReadIntArg } from "./ScriptArgs.ts";

const DefaultOut = "Runs/Labels/sample.jsonl";
const DefaultPerSlice = 50;
const DefaultSeed = 7;

type Reservoir = { Seen: number; Picked: DocumentSummary[] };

/** Seeded per-slice reservoir sampling (Algorithm R) over every Filtered row, page by page. */
async function SampleFiltered(Store: DocumentStore, PerSlice: number, Rng: SeededRng): Promise<Map<EduSlice, Reservoir>> {
  const Reservoirs = new Map<EduSlice, Reservoir>(EduSlices.map((Slice) => [Slice, { Seen: 0, Picked: [] }]));
  let Cursor = "";
  for (;;) {
    const Page = await Store.ListSummaries({ Filter: { Tier: "Filtered" }, AfterId: Cursor, Limit: MaxPageLimit });
    if (Page.Rows.length === 0) break;
    Cursor = Page.Rows[Page.Rows.length - 1]!.Id;
    for (const Row of Page.Rows) {
      const Pool = Reservoirs.get(EduSliceForLang(Row.Lang))!;
      if (Pool.Picked.length < PerSlice) Pool.Picked.push(Row);
      else {
        const Slot = Math.floor(Rng.NextFloat() * (Pool.Seen + 1));
        if (Slot < PerSlice) Pool.Picked[Slot] = Row;
      }
      Pool.Seen++;
    }
  }
  return Reservoirs;
}

async function Main(): Promise<void> {
  const Out = ReadArg("--Out=", DefaultOut);
  const PerSlice = ReadIntArg("--PerSlice", DefaultPerSlice, { Min: 1 });
  const Seed = ReadIntArg("--Seed", DefaultSeed);
  const Kind = ReadArg("--Kind=", "code");
  if (!IsDataKind(Kind)) throw new Error(`--Kind must be a data kind, got '${Kind}'`);
  if ((process.env["DATABASE_URL"] ?? "") === "") throw new Error("needs DATABASE_URL: samples are drawn from the Postgres Filtered tier");
  if (existsSync(Out) && !ReadFlag("--Force")) throw new Error(`${Out} exists (it may hold labels) — pass --Force to overwrite or --Out=<new file>`);

  const Stores = ResolveFoundryStores();
  try {
    const Store = Stores.Kind(Kind);
    const Reservoirs = await SampleFiltered(Store, PerSlice, new SeededRng(Seed));
    const Picked = [...Reservoirs.values()].flatMap((Pool) => Pool.Picked);
    const Lines: string[] = [];
    for (let Start = 0; Start < Picked.length; Start += MaxPageLimit) {
      const Chunk = Picked.slice(Start, Start + MaxPageLimit);
      const Contents = await Store.ContentByIds(Chunk.map((Row) => Row.Id));
      for (const Row of Chunk) {
        const Text = Contents.get(Row.Id);
        if (Text === undefined) continue; // deleted since the summary scan
        Lines.push(JSON.stringify({ id: Row.Id, path: Row.Provenance, source: Row.Source, lang: Row.Lang, slice: EduSliceForLang(Row.Lang), text: Text, label: null }));
      }
    }
    mkdirSync(dirname(Out), { recursive: true });
    writeFileSync(Out, Lines.length === 0 ? "" : `${Lines.join("\n")}\n`);
    for (const [Slice, Pool] of Reservoirs) console.log(`  ${Slice.padEnd(6)} sampled ${Pool.Picked.length} of ${Pool.Seen}`);
    console.log(`[label] wrote ${Lines.length} rows to ${Out} — set each "label" to 1 or 0, then: bun run foundry:train-edu --Labels=${Out}`);
  } finally {
    await Stores.Close();
  }
}

try {
  await Main();
} catch (Caught) {
  console.error(`[label] ${(Caught as Error).message}`);
  process.exit(1);
}
