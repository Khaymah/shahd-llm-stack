// Import synthetic SFT records (the SyntheticSft JSONL) into documents_conversation through the NORMAL
// ingest path (IngestDocuments: sanitize, content hash, tier, embed, upsert). Idempotent by
// construction: the row id hashes origin + license + content and the content hash dedups across
// licenses, so importing the same file twice stores nothing new the second time.
//
// Every record is re-validated first — the artifact crossed machines and is external JSON: the Zod
// record schema (on read), the chat-document structure, and the tool REPLAY against the committed
// fixture workspace (a stored tool result must be what the tool really returns). Invalid records are
// counted by reason and skipped, never stored.

import type { ToolReplay } from "../Brain/Sft/ChatDocumentValidation.ts";
import { ValidateChatDocument } from "../Brain/Sft/ChatDocumentValidation.ts";
import { SerializeChatDocument } from "../Brain/Sft/ChatDocument.ts";
import type { SynthRecord } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import type { DocumentStore } from "./DocumentStore.ts";
import type { IngestStats, SourceInput } from "./Ingest.ts";
import { IngestDocuments, IngestFailureBudget } from "./Ingest.ts";
import { SyntheticLicense } from "./SyntheticLicense.ts";

/** The ingest input of one record. */
export function SyntheticSourceInput(Record: SynthRecord): SourceInput {
  return {
    Source: `synthetic-${Record.Generator}`,
    License: SyntheticLicense(Record.SourceLicense, Record.Teacher),
    Lang: `chat-${Record.Slice}`,
    Content: SerializeChatDocument(Record.Document),
    Provenance: Record.Provenance,
    Origin: "synthetic",
  };
}

/** Records ingested per IngestDocuments call (one batched dedup lookup each; bounded memory). */
const ImportBatchSize = 500;

export type SyntheticImportResult = { Stats: IngestStats; Invalid: Record<string, number> };

/** Validate (structure + tool replay) and ingest the records into Store. */
export async function ImportSyntheticRecords(Records: readonly SynthRecord[], Store: DocumentStore, IngestedAt: string, Replay: ToolReplay): Promise<SyntheticImportResult> {
  const Invalid: Record<string, number> = {};
  const Inputs: SourceInput[] = [];
  for (const Record of Records) {
    const Problems = await ValidateChatDocument(Record.Document, Replay);
    if (Problems.length > 0) {
      Invalid[Problems[0].Code] = (Invalid[Problems[0].Code] ?? 0) + 1;
      continue;
    }
    Inputs.push(SyntheticSourceInput(Record));
  }
  const Stats: IngestStats = { Ingested: 0, New: 0, Duplicate: 0, ByTier: { Filtered: 0, Raw: 0, Rejected: 0 }, Failed: 0 };
  const Budget = new IngestFailureBudget(); // one failure budget for the whole import
  for (let Start = 0; Start < Inputs.length; Start += ImportBatchSize) {
    const Batch = await IngestDocuments(Inputs.slice(Start, Start + ImportBatchSize), Store, IngestedAt, undefined, undefined, Budget);
    Stats.Ingested += Batch.Ingested;
    Stats.New += Batch.New;
    Stats.Duplicate += Batch.Duplicate;
    Stats.Failed += Batch.Failed;
    for (const Tier of ["Filtered", "Raw", "Rejected"] as const) Stats.ByTier[Tier] += Batch.ByTier[Tier];
  }
  return { Stats, Invalid };
}
