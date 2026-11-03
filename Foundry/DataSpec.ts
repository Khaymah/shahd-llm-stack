// The Data Spec (Plan: Brain-Strengthening, phase 2): the corpus is DESIGNED before it is collected.
// Instead of a GitHub search query, a spec names the repositories, the slice each one feeds, what to
// include/exclude, the byte budget per slice and the epoch cap. Ingestion (Scripts/IngestSpec.ts)
// collects exactly what the spec says; training (--Spec) samples the mix the spec declares. The file
// is Data/Spec.json; Data/Spec.md explains how to curate it.

import { readFileSync } from "node:fs";
import { z } from "zod";
import { EvalSlices, type EvalSlice } from "../Brain/Eval/Suite/SuiteTypes.ts";

export type DataSlice = EvalSlice | "docs"; // docs = prose about the domain (READMEs, guides, knowledge)
export const DataSlices: readonly DataSlice[] = [...EvalSlices.filter((S) => S !== "general"), "docs"];

const SliceSchema = z.enum(["ts", "js", "html", "css", "go", "sql", "docs"]);

const RepoSchema = z.object({
  Repo: z.string().regex(/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/, "owner/name"),
  Slice: SliceSchema.optional(), // default: derived per file from LangForPath
  Ref: z.string().min(1).optional(), // branch or tag; default = the repo's default branch
  Include: z.array(z.string().min(1)).optional(), // path globs (relative to repo root); default: all code paths
  Exclude: z.array(z.string().min(1)).optional(),
  Note: z.string().optional(), // why this repo is in the spec (human-facing)
});

const SliceBudgetSchema = z.object({
  TargetMb: z.number().positive(), // unique bytes wanted in the Filtered tier for this slice
  Langs: z.array(z.string().min(1)).min(1), // LangForPath tags that belong to the slice (e.g. ["typescript"], ["css","scss"])
  Weight: z.number().positive().optional(), // relative share in the training mix; default = TargetMb
});

export const DataSpecSchema = z.object({
  Version: z.string().min(1),
  Domain: z.string().min(1), // one line: what the brain is for
  Slices: z.record(SliceSchema, SliceBudgetSchema),
  Repos: z.array(RepoSchema).min(1),
  LocalFolders: z.array(z.object({ Path: z.string().min(1), Slice: SliceSchema, Note: z.string().optional() })).optional(),
  Exclusions: z.object({
    PathGlobs: z.array(z.string().min(1)), // on top of CodeFileFilter's built-in list
    MinFileBytes: z.number().int().positive(),
    MaxFileBytes: z.number().int().positive(),
  }),
  Training: z.object({
    MaxEpochs: z.number().positive(), // the memorisation guard: tokens seen / unique tokens must stay below this
    ContextTokens: z.number().int().positive(),
    VocabSize: z.number().int().positive(), // tokenizer merges + 256 bytes + specials
  }),
});

export type DataSpec = z.infer<typeof DataSpecSchema>;
export type DataSpecRepo = z.infer<typeof RepoSchema>;

export const DefaultDataSpecPath = "Data/Spec.json";

export function ParseDataSpec(Json: unknown): DataSpec {
  const Spec = DataSpecSchema.parse(Json);
  for (const [Slice, Budget] of Object.entries(Spec.Slices)) {
    if (Budget.Langs.length === 0) throw new Error(`DataSpec: slice ${Slice} lists no Langs`);
  }
  if (Spec.Exclusions.MinFileBytes >= Spec.Exclusions.MaxFileBytes) throw new Error("DataSpec: MinFileBytes must be below MaxFileBytes");
  return Spec;
}

export function LoadDataSpec(Path = DefaultDataSpecPath): DataSpec {
  return ParseDataSpec(JSON.parse(readFileSync(Path, "utf8")));
}

/** The slice a language tag belongs to under this spec, or null when the spec does not want it. */
export function SliceForLang(Spec: DataSpec, Lang: string): DataSlice | null {
  for (const [Slice, Budget] of Object.entries(Spec.Slices)) if (Budget.Langs.includes(Lang)) return Slice as DataSlice;
  return null;
}
