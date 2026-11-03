// The text the suite measures against, and where it comes from:
//   • the FROZEN held-out set (BPB + the held-out side of the memorization probe), in priority order:
//       (a) a document store: Filtered documents on the validation side of the shared split rule
//           (Brain/Data/DocumentSplit.ts — the same rule Scripts/TrainOnFoundry.ts trains with), capped
//           per kind, deterministic order by id;
//       (b) a text file (--HeldOut=path);
//       (c) the built-in seed corpus under Corpus/Seed/ — a LAST RESORT, and only held-out for a model
//           that was not trained on it (the Note says so);
//   • the TRAINING text the memorization probe searches for verbatim n-grams: a text file
//     (--TrainCorpus=path) or the store's Filtered tier on the TRAINING side of the split, capped in bytes.
//
// Brain never imports Foundry: a store is reached through the structural interface below, which a
// Foundry DocumentStore satisfies as-is (the script passes one per data kind).

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { IsValidationDocument } from "../../Data/DocumentSplit.ts";

/** The subset of a document store the suite reads (keyset id paging + content by id + tier reads). */
export interface SuiteDocumentSource {
  ListSummaries(Opts: { Filter?: { Tier?: "Filtered" }; Limit?: number; AfterId?: string }): Promise<{ Rows: readonly { Id: string }[] }>;
  ContentByIds(Ids: readonly string[]): Promise<Map<string, string>>;
  ByTier(Tier: "Filtered", Limit?: number): Promise<readonly { Id: string; Content: string }[]>;
}

export type HeldOutDoc = { Kind: string; Id: string; Content: string };
export type CorpusOrigin = "store" | "file" | "seed";
export type HeldOutSet = { Origin: CorpusOrigin; Note: string; Docs: HeldOutDoc[] };
export type TrainCorpus = { Origin: Exclude<CorpusOrigin, "seed">; Note: string; Docs: string[]; Bytes: number };

// One store page. Stores clamp larger requests to their own ceiling (Foundry: MaxPageLimit = 500), and
// the paging loops below stop only on an EMPTY page, so a clamped page is never mistaken for the end.
const StorePageSize = 500;
// Pretraining reads ~400 documents per MB (Scripts/TrainOnFoundry.ts over-reads the same way): the
// tier read is sized from the byte budget, then cut at the budget itself.
const DocsPerMegabyte = 400;
const BytesPerMegabyte = 1_000_000;
const MinTierRead = 200;
// File text is split into "documents" at blank lines (paragraph / top-level block boundaries).
const ParagraphBreak = /\n\s*\n/;

const SeedCorpusDir = fileURLToPath(new URL("../../../Corpus/Seed", import.meta.url));

function Utf8Bytes(Text: string): number {
  return Buffer.byteLength(Text, "utf8");
}

/** (a) Held-out documents from per-kind stores: validation-side ids, ascending, up to PerKind each. */
export async function LoadHeldOutFromStore(
  Sources: ReadonlyMap<string, SuiteDocumentSource>,
  PerKind: number,
  MaxScan: number,
): Promise<HeldOutSet> {
  const Docs: HeldOutDoc[] = [];
  const Counts: string[] = [];
  for (const [Kind, Store] of Sources) {
    const Ids: string[] = [];
    let AfterId = "";
    let Scanned = 0;
    while (Ids.length < PerKind && Scanned < MaxScan) {
      const Page = await Store.ListSummaries({ Filter: { Tier: "Filtered" }, Limit: StorePageSize, AfterId });
      if (Page.Rows.length === 0) break;
      for (const Row of Page.Rows) {
        if (IsValidationDocument(Row.Id) && Ids.length < PerKind) Ids.push(Row.Id);
      }
      Scanned += Page.Rows.length;
      AfterId = Page.Rows[Page.Rows.length - 1].Id;
    }
    for (let From = 0; From < Ids.length; From += StorePageSize) {
      const Chunk = Ids.slice(From, From + StorePageSize);
      const Contents = await Store.ContentByIds(Chunk);
      for (const Id of Chunk) {
        const Content = Contents.get(Id);
        if (Content !== undefined && Content.length > 0) Docs.push({ Kind, Id, Content }); // deleted since listing -> skipped
      }
    }
    Counts.push(`${Kind}=${Docs.filter((D) => D.Kind === Kind).length}`);
  }
  return { Origin: "store", Note: `store Filtered tier, validation side of the document split (sha256(id) mod 100 < 10), ascending id: ${Counts.join(", ")}`, Docs };
}

/** (b) Held-out text from a file, one kind ("file"), split into documents at blank lines. */
export function LoadHeldOutFromFile(Path: string): HeldOutSet {
  const Parts = readFileSync(Path, "utf8").split(ParagraphBreak).filter((P) => P.trim().length > 0);
  return { Origin: "file", Note: `file ${Path} (${Parts.length} blocks)`, Docs: Parts.map((Content, I) => ({ Kind: "file", Id: `${Path}#${I}`, Content })) };
}

/** (c) The built-in seed corpus (every file under Corpus/Seed/, sorted by name), kind "seed". */
export function LoadHeldOutFromSeed(Dir: string = SeedCorpusDir): HeldOutSet {
  const Names = readdirSync(Dir).filter((Name) => statSync(join(Dir, Name)).isFile()).sort();
  return {
    Origin: "seed",
    Note: `LAST RESORT: the built-in seed corpus (${Names.length} files under Corpus/Seed/) — no store and no --HeldOut file; held-out only for a model NOT trained on the seed corpus (TrainOnCorpus trains on it)`,
    Docs: Names.map((Name) => ({ Kind: "seed", Id: Name, Content: readFileSync(join(Dir, Name), "utf8") })),
  };
}

/** Training text from a file, split at blank lines, cut at MaxBytes. */
export function LoadTrainCorpusFromFile(Path: string, MaxBytes: number): TrainCorpus {
  const Docs: string[] = [];
  let Bytes = 0;
  for (const Part of readFileSync(Path, "utf8").split(ParagraphBreak)) {
    if (Part.trim().length === 0) continue;
    if (Bytes >= MaxBytes) break;
    Docs.push(Part);
    Bytes += Utf8Bytes(Part);
  }
  return { Origin: "file", Note: `file ${Path}`, Docs, Bytes };
}

/** Training text from per-kind stores: Filtered tier, TRAINING side of the split, ascending id (the
 *  order pretraining reads it in), until MaxBytes in total (kinds in the map's order). */
export async function LoadTrainCorpusFromStore(Sources: ReadonlyMap<string, SuiteDocumentSource>, MaxBytes: number): Promise<TrainCorpus> {
  const Docs: string[] = [];
  let Bytes = 0;
  const Counts: string[] = [];
  const Read = Math.max(MinTierRead, Math.ceil((MaxBytes / BytesPerMegabyte) * DocsPerMegabyte));
  for (const [Kind, Store] of Sources) {
    if (Bytes >= MaxBytes) break;
    let Count = 0;
    for (const Doc of await Store.ByTier("Filtered", Read)) {
      if (IsValidationDocument(Doc.Id)) continue;
      if (Bytes >= MaxBytes) break;
      Docs.push(Doc.Content);
      Bytes += Utf8Bytes(Doc.Content);
      Count++;
    }
    Counts.push(`${Kind}=${Count}`);
  }
  return { Origin: "store", Note: `store Filtered tier, training side of the document split: ${Counts.join(", ")}`, Docs, Bytes };
}
