// Educational-value score — filter layer 3 (Plan: Brain-Strengthening §2.2, FineWeb-Edu / phi-1
// idea). "Does this file teach something?" as a logistic regression over the hand-crafted features of
// EducationalFeatures: Score = sigmoid(Bias + Σ Weight_i · (Feature_i − Mean_i) / Std_i).
//
// The weights live in Data/EduWeights.json ({Version, FeatureNames, Mean, Std, Weights, Bias,
// Threshold}). Until that file is TRAINED on hand-labelled files (Scripts/LabelSamples.ts →
// Scripts/TrainEduClassifier.ts), it holds DefaultEduWeights below: a hand-set PRIOR that encodes
// the plan's intuitions, not a fitted model. DefaultEduWeights is the single source of the prior —
// the shipped JSON is generated from it (`bun run foundry:train-edu --WriteDefault`), and a missing
// file falls back to it so tests and the in-memory path need no file.
//
// Training (FitEduLogistic) is plain full-batch gradient descent with L2, in pure TypeScript and
// fully deterministic: zero init, a seeded hold-out split, a fixed step count.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { z } from "zod";
import { SeededRng } from "../Random/SeededRng.ts";
import { EduFeatureNames, EduFeatureVector, ExtractEduFeatures } from "./EducationalFeatures.ts";
import type { EduFeatureName, EduFeatures } from "./EducationalFeatures.ts";

export type EduWeights = {
  Version: string;
  FeatureNames: string[];
  Mean: number[];
  Std: number[];
  Weights: number[];
  Bias: number;
  Threshold: number; // tiering rejects code whose Score is below this
};

/** One feature's signed push on the logit (Weight · standardised value). Negative = "less educational". */
export type EduContribution = { Name: EduFeatureName; Value: number; Contribution: number };
export type EduScoreResult = { Score: number; Features: EduFeatures; Contributions: EduContribution[] };

/** Where the active weights live (relative to the repo root, like Data/Spec.json). */
export const DefaultEduWeightsPath = "Data/EduWeights.json";

// The hand-set prior, one row per feature: [Mean, Std, Weight]. Mean/Std are rough typical values of
// clean, human-written source, so a standardised value says "how unusual is this file"; Weight is the
// direction and strength of the plan's intuition (in logit units per standard deviation):
//   + comments, doc comments, descriptive identifiers, real functions, types, error handling, tests
//   − extreme nesting, long lines, string/number-literal mass (data tables, i18n), TODO debt,
//     console.log debugging. Blank lines (+) mark structure.
//   language one-hot: an offset, not a judgement. Markup, style sheets and SQL cannot carry type
//     annotations, error handling or doc comments, and HTML is attribute-string heavy by nature, so
//     without an offset the prior would reject those whole slices. ts/js/go/other start neutral (0);
//     real per-language base rates are something only labelled data can tell us.
const PriorRows: Record<EduFeatureName, readonly [number, number, number]> = {
  CommentRatio: [0.12, 0.1, 0.7],
  DocCommentCoverage: [0.3, 0.35, 0.5],
  AvgIdentifierLength: [6, 2, 0.6],
  IdentifierTokenRatio: [0.4, 0.1, 0.3],
  FunctionsPerKb: [2, 1.5, 0.4],
  NestingExcess: [0.5, 2, -0.5],
  AvgLineLength: [35, 20, -0.4],
  LogMaxLineLength: [4.8, 0.7, -0.6],
  BlankLineRatio: [0.12, 0.08, 0.2],
  StringLiteralRatio: [0.15, 0.15, -0.7],
  NumberLiteralRatio: [0.03, 0.05, -0.5],
  TodoDensity: [0.3, 2, -0.3],
  HasTypeAnnotations: [0.5, 0.5, 0.3],
  HasErrorHandling: [0.4, 0.5, 0.3],
  HasTestSignals: [0.15, 0.36, 0.2],
  ConsoleLogDensity: [0.3, 2, -0.4],
  Lang_ts: [0, 1, 0],
  Lang_js: [0, 1, 0],
  Lang_html: [0, 1, 2],
  Lang_css: [0, 1, 0.8],
  Lang_go: [0, 1, 0],
  Lang_sql: [0, 1, 0.5],
  Lang_other: [0, 1, 0],
};
/** Logit of a perfectly typical file (every standardised feature 0) ≈ sigmoid(1) ≈ 0.73. */
const PriorBias = 1;
/** Prior cut: reject only what the prior is fairly sure about. Re-derived when trained. */
const PriorThreshold = 0.3;

export const DefaultEduWeights: EduWeights = {
  Version: "prior-v1 (hand-set, untrained; run Scripts/TrainEduClassifier.ts on labelled files)",
  FeatureNames: [...EduFeatureNames],
  Mean: EduFeatureNames.map((Name) => PriorRows[Name][0]),
  Std: EduFeatureNames.map((Name) => PriorRows[Name][1]),
  Weights: EduFeatureNames.map((Name) => PriorRows[Name][2]),
  Bias: PriorBias,
  Threshold: PriorThreshold,
};

const EduWeightsSchema = z.object({
  Version: z.string().min(1),
  FeatureNames: z.array(z.string()),
  Mean: z.array(z.number().finite()),
  Std: z.array(z.number().finite().positive()),
  Weights: z.array(z.number().finite()),
  Bias: z.number().finite(),
  Threshold: z.number().min(0).max(1),
});

/** Validate parsed JSON as weights for THIS feature set (names and order must match exactly). */
export function ParseEduWeights(Json: unknown): EduWeights {
  const Weights = EduWeightsSchema.parse(Json);
  const Expected = EduFeatureNames.join(",");
  if (Weights.FeatureNames.join(",") !== Expected) {
    throw new Error(`EduWeights: FeatureNames do not match this build's features (expected ${Expected}); retrain or regenerate the file`);
  }
  for (const Key of ["Mean", "Std", "Weights"] as const) {
    if (Weights[Key].length !== EduFeatureNames.length) throw new Error(`EduWeights: ${Key} has ${Weights[Key].length} entries, expected ${EduFeatureNames.length}`);
  }
  return Weights;
}

/** Load weights from a file; a MISSING file yields the built-in prior, an invalid one throws. */
export function LoadEduWeights(Path: string = DefaultEduWeightsPath): EduWeights {
  if (!existsSync(Path)) return DefaultEduWeights;
  return ParseEduWeights(JSON.parse(readFileSync(Path, "utf8")));
}

export function SaveEduWeights(Weights: EduWeights, Path: string = DefaultEduWeightsPath): void {
  ParseEduWeights(Weights); // never write a file the loader would refuse
  mkdirSync(dirname(Path), { recursive: true });
  writeFileSync(Path, `${JSON.stringify(Weights, null, 2)}\n`);
}

// Tiering scores every ingested code document; the weights file is read once per process and path.
const WeightsCache = new Map<string, EduWeights>();

/** The weights tiering uses: Data/EduWeights.json (cached), or the prior when the file is absent. */
export function ActiveEduWeights(Path: string = DefaultEduWeightsPath): EduWeights {
  let Weights = WeightsCache.get(Path);
  if (Weights === undefined) {
    Weights = LoadEduWeights(Path);
    WeightsCache.set(Path, Weights);
  }
  return Weights;
}

function Sigmoid(X: number): number {
  return X >= 0 ? 1 / (1 + Math.exp(-X)) : Math.exp(X) / (1 + Math.exp(X));
}

function Standardise(Vector: readonly number[], Weights: EduWeights): number[] {
  return Vector.map((Value, I) => (Value - Weights.Mean[I]!) / Weights.Std[I]!);
}

function Logit(Standardised: readonly number[], Weights: EduWeights): number {
  let Sum = Weights.Bias;
  for (let I = 0; I < Standardised.length; I++) Sum += Weights.Weights[I]! * Standardised[I]!;
  return Sum;
}

/** Score an already-extracted feature vector (EduFeatureNames order) → probability in 0..1. */
export function ScoreEduVector(Vector: readonly number[], Weights: EduWeights = DefaultEduWeights): number {
  return Sigmoid(Logit(Standardise(Vector, Weights), Weights));
}

/** The educational value of a document: Score in 0..1, its features, and each feature's push. */
export function ScoreEducational(Text: string, Lang: string, Weights: EduWeights = DefaultEduWeights): EduScoreResult {
  const Features = ExtractEduFeatures(Text, Lang);
  const Standardised = Standardise(EduFeatureVector(Features), Weights);
  const Contributions = EduFeatureNames.map((Name, I) => ({ Name, Value: Features[Name], Contribution: Weights.Weights[I]! * Standardised[I]! }));
  return { Score: Sigmoid(Logit(Standardised, Weights)), Features, Contributions };
}

/** The N features that pulled the score down the most (most negative first; only negative ones). */
export function TopNegativeContributions(Result: EduScoreResult, N: number): EduContribution[] {
  return Result.Contributions.filter((C) => C.Contribution < 0)
    .sort((A, B) => A.Contribution - B.Contribution)
    .slice(0, N);
}

// ---------------------------------------------------------------------------------------------
// Training: logistic regression by full-batch gradient descent with L2 (bias unregularised).

export type EduSample = { Vector: number[]; Label: 0 | 1 };
export type EduFitOptions = { Steps?: number; LearningRate?: number; L2?: number; Threshold?: number; Version?: string };
export type EduMetrics = { N: number; Accuracy: number; Precision: number; Recall: number; F1: number };

const DefaultFitSteps = 2000;
const DefaultLearningRate = 0.5;
const DefaultL2 = 1e-3;
const DefaultDecisionThreshold = 0.5;
/** A feature that never varies in training gets Std 1 (its standardised value is then constant 0). */
const MinStd = 1e-6;

/** Deterministic hold-out split: a seeded Fisher–Yates shuffle, then the last HoldOutFraction. */
export function SplitHoldOut<T>(Items: readonly T[], HoldOutFraction: number, Seed: number): { Train: T[]; Test: T[] } {
  const Shuffled = [...Items];
  const Rng = new SeededRng(Seed);
  for (let I = Shuffled.length - 1; I > 0; I--) {
    const J = Math.floor(Rng.NextFloat() * (I + 1));
    [Shuffled[I], Shuffled[J]] = [Shuffled[J]!, Shuffled[I]!];
  }
  const TestCount = Math.min(Shuffled.length - 1, Math.max(1, Math.round(Shuffled.length * HoldOutFraction)));
  return { Train: Shuffled.slice(0, Shuffled.length - TestCount), Test: Shuffled.slice(Shuffled.length - TestCount) };
}

/** Fit weights on labelled vectors (EduFeatureNames order). Mean/Std come from the training set. */
export function FitEduLogistic(Samples: readonly EduSample[], Options: EduFitOptions = {}): EduWeights {
  if (Samples.length === 0) throw new Error("FitEduLogistic: no labelled samples");
  const Dim = EduFeatureNames.length;
  for (const Sample of Samples) {
    if (Sample.Vector.length !== Dim) throw new Error(`FitEduLogistic: vector has ${Sample.Vector.length} features, expected ${Dim}`);
  }
  const Steps = Options.Steps ?? DefaultFitSteps;
  const Lr = Options.LearningRate ?? DefaultLearningRate;
  const L2 = Options.L2 ?? DefaultL2;
  const Mean = new Array<number>(Dim).fill(0);
  const Std = new Array<number>(Dim).fill(0);
  for (const Sample of Samples) for (let I = 0; I < Dim; I++) Mean[I] += Sample.Vector[I]! / Samples.length;
  for (const Sample of Samples) for (let I = 0; I < Dim; I++) Std[I] += (Sample.Vector[I]! - Mean[I]!) ** 2 / Samples.length;
  for (let I = 0; I < Dim; I++) Std[I] = Math.sqrt(Std[I]!) > MinStd ? Math.sqrt(Std[I]!) : 1;

  const Fitted: EduWeights = {
    Version: Options.Version ?? `trained-${Samples.length}`,
    FeatureNames: [...EduFeatureNames],
    Mean,
    Std,
    Weights: new Array<number>(Dim).fill(0),
    Bias: 0,
    Threshold: Options.Threshold ?? DefaultDecisionThreshold,
  };
  const X = Samples.map((Sample) => Standardise(Sample.Vector, Fitted));
  for (let Step = 0; Step < Steps; Step++) {
    const Grad = new Array<number>(Dim).fill(0);
    let GradBias = 0;
    for (let N = 0; N < X.length; N++) {
      const Residual = Sigmoid(Logit(X[N]!, Fitted)) - Samples[N]!.Label;
      for (let I = 0; I < Dim; I++) Grad[I] += (Residual * X[N]![I]!) / X.length;
      GradBias += Residual / X.length;
    }
    for (let I = 0; I < Dim; I++) Fitted.Weights[I] -= Lr * (Grad[I]! + L2 * Fitted.Weights[I]!);
    Fitted.Bias -= Lr * GradBias;
  }
  return Fitted;
}

/** Accuracy / precision / recall / F1 of the weights at their Threshold on labelled vectors. */
export function EvaluateEdu(Samples: readonly EduSample[], Weights: EduWeights): EduMetrics {
  let TruePos = 0;
  let FalsePos = 0;
  let FalseNeg = 0;
  let Correct = 0;
  for (const Sample of Samples) {
    const Predicted = ScoreEduVector(Sample.Vector, Weights) >= Weights.Threshold ? 1 : 0;
    if (Predicted === Sample.Label) Correct++;
    if (Predicted === 1 && Sample.Label === 1) TruePos++;
    if (Predicted === 1 && Sample.Label === 0) FalsePos++;
    if (Predicted === 0 && Sample.Label === 1) FalseNeg++;
  }
  const Precision = TruePos + FalsePos === 0 ? 0 : TruePos / (TruePos + FalsePos);
  const Recall = TruePos + FalseNeg === 0 ? 0 : TruePos / (TruePos + FalseNeg);
  const F1 = Precision + Recall === 0 ? 0 : (2 * Precision * Recall) / (Precision + Recall);
  return { N: Samples.length, Accuracy: Samples.length === 0 ? 0 : Correct / Samples.length, Precision, Recall, F1 };
}
