// The on-disk interchange of the synthesizer: one JSONL line per accepted conversation (the shard files,
// the merged SyntheticSft artifact, the import input) and one stats JSON per shard. Both are EXTERNAL
// JSON when read back (artifacts cross machines), so both are Zod-validated on read.

import { z } from "zod";
import { ChatDocumentSchema, ConversationHash, SerializeChatDocument } from "../ChatDocument.ts";
import type { ChatDocument } from "../ChatDocument.ts";
import { SynthGenerators } from "./SynthTypes.ts";
import type { SynthGenerator } from "./SynthTypes.ts";

export const SynthRecordSchema = z.object({
  Generator: z.enum(SynthGenerators),
  Teacher: z.string().min(1),
  Slice: z.string().min(1),
  Source: z.string().min(1), // grounding repo/corpus name, or the generator for ungrounded items
  SourceId: z.string().min(1),
  SourceLicense: z.string().min(1), // SPDX of the grounding doc, or UngroundedSourceLicense
  Provenance: z.string().min(1), // synthetic:<generator>:<teacher>:<source-id>
  Tokens: z.number().int().nonnegative(), // measured size of the rendered conversation
  Hash: z.string().min(1), // ConversationHash (exact-dedup key)
  Document: ChatDocumentSchema,
});

export type SynthRecord = z.infer<typeof SynthRecordSchema>;

/** The provenance string of a synthetic conversation. */
export function SyntheticProvenance(Generator: SynthGenerator, Teacher: string, SourceId: string): string {
  return `synthetic:${Generator}:${Teacher}:${SourceId}`;
}

/** One JSONL line (the document in canonical key order). */
export function SerializeSynthRecord(Record: SynthRecord): string {
  const Document = JSON.parse(SerializeChatDocument(Record.Document)) as ChatDocument;
  return JSON.stringify({ ...Record, Document });
}

export type ParsedRecords = { Records: SynthRecord[]; Invalid: { Line: number; Reason: string }[] };

/** Parse JSONL text; malformed lines are reported (never silently dropped) and skipped. */
export function ParseSynthRecords(Text: string): ParsedRecords {
  const Records: SynthRecord[] = [];
  const Invalid: ParsedRecords["Invalid"] = [];
  Text.split("\n").forEach((Raw, Index) => {
    const Line = Raw.trim();
    if (Line === "") return;
    let Json: unknown;
    try {
      Json = JSON.parse(Line);
    } catch {
      Invalid.push({ Line: Index + 1, Reason: "not JSON" });
      return;
    }
    const Parsed = SynthRecordSchema.safeParse(Json);
    if (Parsed.success) Records.push(Parsed.data);
    else Invalid.push({ Line: Index + 1, Reason: Parsed.error.issues.map((I) => `${I.path.join(".")}: ${I.message}`).join("; ") });
  });
  return { Records, Invalid };
}

export const GeneratorStatsSchema = z.object({
  Attempted: z.number().int().nonnegative(),
  Accepted: z.number().int().nonnegative(),
  Tokens: z.number().int().nonnegative(), // summed over accepted conversations
  Rejected: z.record(z.string(), z.number().int().nonnegative()), // reason -> count
});

export const SynthStatsSchema = z.object({
  Teacher: z.string(),
  Shard: z.string(), // "k/N"
  StopReason: z.string(),
  ElapsedMs: z.number().nonnegative(),
  Generators: z.record(z.string(), GeneratorStatsSchema),
});

export type GeneratorStats = z.infer<typeof GeneratorStatsSchema>;
export type SynthStats = z.infer<typeof SynthStatsSchema>;

export function EmptyGeneratorStats(): GeneratorStats {
  return { Attempted: 0, Accepted: 0, Tokens: 0, Rejected: {} };
}

/** Sum per-generator statistics over several shards. */
export function SumGeneratorStats(All: readonly SynthStats[]): Record<string, GeneratorStats> {
  const Out: Record<string, GeneratorStats> = {};
  for (const Stats of All) {
    for (const [Name, G] of Object.entries(Stats.Generators)) {
      const Into = (Out[Name] ??= EmptyGeneratorStats());
      Into.Attempted += G.Attempted;
      Into.Accepted += G.Accepted;
      Into.Tokens += G.Tokens;
      for (const [Reason, Count] of Object.entries(G.Rejected)) Into.Rejected[Reason] = (Into.Rejected[Reason] ?? 0) + Count;
    }
  }
  return Out;
}

const PercentScale = 100;

/** One human line per generator: accepted / attempted, rate, mean tokens, rejections by reason. */
export function DescribeGeneratorStats(Generators: Record<string, GeneratorStats>): string[] {
  return Object.entries(Generators).map(([Name, G]) => {
    const Rate = G.Attempted === 0 ? 0 : (G.Accepted / G.Attempted) * PercentScale;
    const Mean = G.Accepted === 0 ? 0 : G.Tokens / G.Accepted;
    const Reasons = Object.entries(G.Rejected).sort((A, B) => B[1] - A[1]).map(([Reason, Count]) => `${Reason} ${Count}`).join(", ");
    return `${Name}: ${G.Accepted}/${G.Attempted} accepted (${Rate.toFixed(1)}%), mean ${Mean.toFixed(0)} tokens${Reasons === "" ? "" : ` — rejected: ${Reasons}`}`;
  });
}

/** Exact dedup across shards by the RECOMPUTED conversation hash (the stored Hash field is not trusted);
 *  the first occurrence wins, input order is kept. */
export function DedupSynthRecords(Records: readonly SynthRecord[]): { Records: SynthRecord[]; Duplicates: number } {
  const Seen = new Set<string>();
  const Out: SynthRecord[] = [];
  for (const Record of Records) {
    const Hash = ConversationHash(Record.Document);
    if (Seen.has(Hash)) continue;
    Seen.add(Hash);
    Out.push({ ...Record, Hash });
  }
  return { Records: Out, Duplicates: Records.length - Out.length };
}
