// Train the educational-value classifier (Plan: Brain-Strengthening §2.2, filter layer 3) from hand
// labels, and write Data/EduWeights.json — the file tiering loads (see Brain/Data/EducationalScore.ts).
//
// Labels are JSONL, one object per line: {"id"?, "path"?, "text"?, "lang"?, "label": 0 | 1 | null}.
//   text  — used as-is when present (Scripts/LabelSamples.ts writes it);
//   id    — otherwise read from the store (needs DATABASE_URL; --Kind picks the documents_<kind> table);
//   path  — otherwise read from disk.
//   label — 1 = "this file teaches something", 0 = not; null rows (not labelled yet) are skipped.
// The fit is deterministic: a seeded 80/20 hold-out split, zero init, fixed full-batch GD steps with L2.
// The weights written are the ones fitted on the TRAIN split, so the reported hold-out numbers are
// an honest estimate of what tiering will do.
//
//   bun run foundry:train-edu --Labels=Runs/Labels/labels.jsonl          # fit, report, write
//   bun run foundry:train-edu --Labels=… --DryRun                        # fit + report only
//   bun run foundry:train-edu --WriteDefault                             # (re)write the hand-set prior

import { existsSync, readFileSync } from "node:fs";
import { z } from "zod";
import {
  DefaultEduWeights,
  DefaultEduWeightsPath,
  EvaluateEdu,
  FitEduLogistic,
  SaveEduWeights,
  SplitHoldOut,
} from "../Brain/Data/EducationalScore.ts";
import type { EduMetrics, EduSample, EduWeights } from "../Brain/Data/EducationalScore.ts";
import { EduFeatureNames, EduFeatureVector, ExtractEduFeatures } from "../Brain/Data/EducationalFeatures.ts";
import { LangForPath } from "../Foundry/CodeFileFilter.ts";
import { IsDataKind } from "../Foundry/DataKinds.ts";
import type { DocumentStore } from "../Foundry/DocumentStore.ts";
import type { FoundryStores } from "../Foundry/FoundryStores.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag, ReadFloatArg, ReadIntArg } from "./ScriptArgs.ts";

const DefaultLabelsPath = "Runs/Labels/labels.jsonl";
const DefaultHoldOut = 0.2;
const DefaultSeed = 7;
const DefaultSteps = 2000;
const DefaultLr = 0.5;
const DefaultL2 = 1e-3;
const DefaultThreshold = 0.5;
/** Below this many labelled rows the hold-out numbers are meaningless — refuse instead of pretending. */
const MinLabelledRows = 10;
const ShownWeights = 8;

const LabelRowSchema = z.object({
  id: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  text: z.string().optional(),
  lang: z.string().min(1).optional(),
  label: z.union([z.literal(0), z.literal(1), z.null()]),
});
type LabelRow = z.infer<typeof LabelRowSchema>;

function ParseLabels(Path: string): LabelRow[] {
  if (!existsSync(Path)) throw new Error(`labels file not found: ${Path} (create one with Scripts/LabelSamples.ts and label it)`);
  const Rows: LabelRow[] = [];
  const Lines = readFileSync(Path, "utf8").split("\n");
  for (let I = 0; I < Lines.length; I++) {
    const Line = Lines[I]!.trim();
    if (Line === "") continue;
    let Json: unknown;
    try {
      Json = JSON.parse(Line);
    } catch (Caught) {
      throw new Error(`${Path}:${I + 1}: invalid JSON (${(Caught as Error).message})`);
    }
    const Parsed = LabelRowSchema.safeParse(Json);
    if (!Parsed.success) throw new Error(`${Path}:${I + 1}: ${Parsed.error.issues.map((Issue) => `${Issue.path.join(".")}: ${Issue.message}`).join("; ")}`);
    Rows.push(Parsed.data);
  }
  return Rows;
}

/** Lazily opened store (only when a row needs text by id) — closed by the caller. */
class LazyStore {
  private Stores: FoundryStores | null = null;
  constructor(private readonly Kind: string) {}

  Get(): DocumentStore {
    if ((process.env["DATABASE_URL"] ?? "") === "") throw new Error("a label row has only an id: set DATABASE_URL to read its text from the store, or add \"text\"/\"path\"");
    if (!IsDataKind(this.Kind)) throw new Error(`--Kind must be a data kind, got '${this.Kind}'`);
    if (this.Stores === null) this.Stores = ResolveFoundryStores();
    return this.Stores.Kind(this.Kind);
  }

  async Close(): Promise<void> {
    await this.Stores?.Close();
  }
}

async function ResolveSample(Row: LabelRow, Store: LazyStore, Where: string): Promise<EduSample> {
  let Text = Row.text;
  let Lang = Row.lang;
  if (Text === undefined && Row.id !== undefined) {
    const Doc = await Store.Get().DocumentById(Row.id);
    if (Doc === null) throw new Error(`${Where}: id ${Row.id} is not in the store`);
    Text = Doc.Content;
    Lang ??= Doc.Lang;
  }
  if (Text === undefined && Row.path !== undefined) Text = readFileSync(Row.path, "utf8");
  if (Text === undefined) throw new Error(`${Where}: needs "text", "id" or "path"`);
  Lang ??= Row.path !== undefined ? LangForPath(Row.path) : "other";
  return { Vector: EduFeatureVector(ExtractEduFeatures(Text, Lang)), Label: Row.label === 1 ? 1 : 0 };
}

function FormatMetrics(Name: string, M: EduMetrics): string {
  const Pct = (Value: number): string => `${(Value * 100).toFixed(1)}%`;
  return `${Name.padEnd(9)} n=${String(M.N).padStart(5)}  accuracy=${Pct(M.Accuracy)}  precision=${Pct(M.Precision)}  recall=${Pct(M.Recall)}  f1=${Pct(M.F1)}`;
}

function StrongestWeights(Weights: EduWeights): string {
  return EduFeatureNames.map((Name, I) => ({ Name, Weight: Weights.Weights[I]! }))
    .sort((A, B) => Math.abs(B.Weight) - Math.abs(A.Weight))
    .slice(0, ShownWeights)
    .map((W) => `${W.Name}=${W.Weight >= 0 ? "+" : ""}${W.Weight.toFixed(3)}`)
    .join("  ");
}

async function Main(): Promise<void> {
  const Out = ReadArg("--Out=", DefaultEduWeightsPath);
  if (ReadFlag("--WriteDefault")) {
    SaveEduWeights(DefaultEduWeights, Out);
    console.log(`[edu] wrote the hand-set prior (${DefaultEduWeights.Version}) to ${Out}`);
    return;
  }
  const LabelsPath = ReadArg("--Labels=", DefaultLabelsPath);
  const HoldOut = ReadFloatArg("--HoldOut", DefaultHoldOut, { Min: 0.05, Max: 0.5 });
  const Seed = ReadIntArg("--Seed", DefaultSeed);
  const Steps = ReadIntArg("--Steps", DefaultSteps, { Min: 1 });
  const LearningRate = ReadFloatArg("--Lr", DefaultLr, { Min: 0 });
  const L2 = ReadFloatArg("--L2", DefaultL2, { Min: 0 });
  const Threshold = ReadFloatArg("--Threshold", DefaultThreshold, { Min: 0, Max: 1 });
  const DryRun = ReadFlag("--DryRun");

  const Rows = ParseLabels(LabelsPath);
  const Labelled = Rows.filter((Row) => Row.label !== null);
  console.log(`[edu] ${LabelsPath}: ${Rows.length} rows, ${Labelled.length} labelled, ${Rows.length - Labelled.length} skipped (label null)`);
  if (Labelled.length < MinLabelledRows) throw new Error(`need at least ${MinLabelledRows} labelled rows, got ${Labelled.length}`);

  const Store = new LazyStore(ReadArg("--Kind=", "code"));
  const Samples: EduSample[] = [];
  try {
    for (let I = 0; I < Rows.length; I++) {
      const Row = Rows[I]!;
      if (Row.label !== null) Samples.push(await ResolveSample(Row, Store, `${LabelsPath} row ${I + 1}`));
    }
  } finally {
    await Store.Close();
  }
  const Positives = Samples.filter((S) => S.Label === 1).length;
  if (Positives === 0 || Positives === Samples.length) throw new Error(`labels must contain both classes (got ${Positives} positive of ${Samples.length})`);

  const { Train, Test } = SplitHoldOut(Samples, HoldOut, Seed);
  const Version = `trained-${Samples.length}-${new Date().toISOString().slice(0, 10)} (seed ${Seed}, hold-out ${HoldOut})`;
  const Weights = FitEduLogistic(Train, { Steps, LearningRate, L2, Threshold, Version });
  console.log(`[edu] ${Positives} positive / ${Samples.length - Positives} negative; fitted ${Steps} steps (lr ${LearningRate}, l2 ${L2})`);
  console.log(FormatMetrics("train", EvaluateEdu(Train, Weights)));
  console.log(FormatMetrics("held-out", EvaluateEdu(Test, Weights)));
  console.log(`[edu] strongest weights: ${StrongestWeights(Weights)}`);
  if (DryRun) {
    console.log("[edu] --DryRun: weights not written");
    return;
  }
  SaveEduWeights(Weights, Out);
  console.log(`[edu] wrote ${Out} (threshold ${Threshold}); tiering picks it up on the next process start`);
}

try {
  await Main();
} catch (Caught) {
  console.error(`[edu] ${(Caught as Error).message}`);
  process.exit(1);
}
