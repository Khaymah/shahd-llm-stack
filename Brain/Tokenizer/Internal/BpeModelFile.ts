// The persisted tokenizer file (Data/Tokenizer.<Version>.json): a trained merge table plus what it was
// trained on, so every training run can REUSE one tokenizer instead of re-learning a small one per run.
// Shape: { Kind: "Bpe", Version, Merges, TrainedOn: { SampleMb, PerSlice, Source? }, CreatedAt }.
// The file is a superset of BpeModel ({ Merges }), so it feeds BytePairEncoder directly.

import { readFileSync } from "node:fs";
import { z } from "zod";
import { ByteVocab } from "./IncrementalPairStats.ts";
import { AssertPairKeyCapacity } from "./PairKey.ts";

const TokenId = z.number().int().nonnegative();

const SliceSampleSchema = z.object({
  Mb: z.number().finite().nonnegative(), // bytes actually sampled for the slice, in MB (1e6 bytes)
  Docs: z.number().int().nonnegative(),
  TargetMb: z.number().finite().nonnegative().optional(), // what the weights asked for
});

export const BpeModelFileSchema = z.object({
  Kind: z.literal("Bpe"),
  Version: z.string().min(1),
  Merges: z.array(z.tuple([TokenId, TokenId])),
  TrainedOn: z.object({
    SampleMb: z.number().finite().nonnegative(),
    PerSlice: z.record(z.string(), SliceSampleSchema),
    Source: z.string().optional(), // "spec:<path>" or "corpus:<path>"
  }),
  CreatedAt: z.string().min(1),
});

export type BpeModelFile = z.infer<typeof BpeModelFileSchema>;
export type SliceSample = z.infer<typeof SliceSampleSchema>;

/** Throws unless every merge references only ids that exist before it (bytes or earlier merges). */
export function AssertValidMerges(Merges: readonly (readonly [number, number])[], Where: string): void {
  AssertPairKeyCapacity(ByteVocab + Merges.length, Where);
  for (let Rank = 0; Rank < Merges.length; Rank++) {
    const Limit = ByteVocab + Rank;
    const [A, B] = Merges[Rank];
    if (!Number.isInteger(A) || !Number.isInteger(B) || A < 0 || B < 0 || A >= Limit || B >= Limit) {
      throw new Error(`${Where}: merge #${Rank} [${A}, ${B}] references an id that does not exist yet (must be < ${Limit})`);
    }
  }
}

/** Validate a parsed tokenizer file (schema + merge-table integrity). */
export function ParseBpeModelFile(Json: unknown, Where = "BpeModelFile"): BpeModelFile {
  const Parsed = BpeModelFileSchema.safeParse(Json);
  if (!Parsed.success) throw new Error(`${Where}: not a Bpe tokenizer file — ${Parsed.error.issues.map((I) => `${I.path.join(".")}: ${I.message}`).join("; ")}`);
  AssertValidMerges(Parsed.data.Merges, Where);
  return Parsed.data;
}

/** Load and validate Data/Tokenizer.<Version>.json (or any path in the same format). */
export function LoadBpeModel(Path: string): BpeModelFile {
  let Json: unknown;
  try {
    Json = JSON.parse(readFileSync(Path, "utf8"));
  } catch (Error_) {
    throw new Error(`LoadBpeModel: cannot read ${Path} — ${(Error_ as Error).message}`);
  }
  return ParseBpeModelFile(Json, `LoadBpeModel(${Path})`);
}

/** The conventional path of a tokenizer version. */
export function TokenizerFilePath(Version: string, Dir = "Data"): string {
  return `${Dir}/Tokenizer.${Version}.json`;
}

/** True when two merge tables are identical (same pairs, same order). */
export function SameMerges(A: readonly (readonly [number, number])[], B: readonly (readonly [number, number])[]): boolean {
  if (A.length !== B.length) return false;
  for (let I = 0; I < A.length; I++) if (A[I][0] !== B[I][0] || A[I][1] !== B[I][1]) return false;
  return true;
}
