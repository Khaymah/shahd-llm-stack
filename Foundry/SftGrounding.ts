// Grounding documents for the SFT synthesizer, from either source:
//   • the Foundry store (documents_code, Filtered tier) — the permissive code the spec collected;
//   • a local directory (--Corpus=<dir>) when no database is configured — walked with the SAME
//     substantive-code filter as own-repo ingestion (LocalRepoProvider, every quality level accepted).
// Either way only the TRAINING side of the document split is returned (Brain/Data/DocumentSplit.ts): a
// conversation grounded on a validation document would leak it into SFT and inflate held-out numbers.
// Each document is mapped to its data-spec slice by language; languages outside the spec's code slices
// are dropped.

import { PartitionByDocumentSplit } from "../Brain/Data/DocumentSplit.ts";
import type { CodeSlice, GroundingDoc } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { CodeSlices } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import type { DataSpec } from "./DataSpec.ts";
import { SliceForLang } from "./DataSpec.ts";
import type { DocumentStore } from "./DocumentStore.ts";
import { CreateLocalRepoProvider } from "./LocalRepoProvider.ts";
import { ContentHashOf } from "./Ingest.ts";

/** Default cap on the documents read from the store (ascending id, so the selection is deterministic). */
export const DefaultGroundingLimit = 20_000;

/** The License a --Corpus file gets when its header names none and the caller passes none. */
export const UnknownCorpusLicense = "NOASSERTION";

function CodeSliceFor(Spec: DataSpec, Lang: string): CodeSlice | null {
  const Slice = SliceForLang(Spec, Lang);
  return Slice !== null && (CodeSlices as readonly string[]).includes(Slice) ? (Slice as CodeSlice) : null;
}

/** Training-side Filtered code documents from the store, mapped to spec slices. */
export async function LoadStoreGrounding(Store: Pick<DocumentStore, "ByTier">, Spec: DataSpec, Limit = DefaultGroundingLimit): Promise<GroundingDoc[]> {
  const Train = PartitionByDocumentSplit(await Store.ByTier("Filtered", Limit)).Train;
  const Out: GroundingDoc[] = [];
  for (const Doc of Train) {
    const Slice = CodeSliceFor(Spec, Doc.Lang);
    if (Slice === null) continue;
    Out.push({ Id: Doc.Id, Slice, Lang: Doc.Lang, License: Doc.License, Source: Doc.Source, Path: Doc.Provenance, Content: Doc.Content });
  }
  return Out;
}

/** Training-side code files under a local directory, mapped to spec slices. The id is the content hash
 *  (ContentHashOf), so the split and the shard of a file never depend on where it was checked out. */
export async function LoadCorpusGrounding(Dir: string, Spec: DataSpec, License = UnknownCorpusLicense): Promise<GroundingDoc[]> {
  const Provider = CreateLocalRepoProvider({ Roots: [Dir], License, MinLevel: "low" });
  const Inputs = await Provider.Fetch("", 0);
  const Docs: GroundingDoc[] = [];
  for (const Input of Inputs) {
    const Slice = CodeSliceFor(Spec, Input.Lang);
    if (Slice === null) continue;
    Docs.push({ Id: ContentHashOf(Input.Content), Slice, Lang: Input.Lang, License: Input.License, Source: Input.Source, Path: Input.Provenance, Content: Input.Content });
  }
  return PartitionByDocumentSplit(Docs).Train;
}
