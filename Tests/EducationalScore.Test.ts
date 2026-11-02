// Filter layer 3: the educational-value score (features, the hand-set prior, the weights file, and
// the deterministic logistic-regression trainer).

import { test, expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DefaultEduWeights,
  DefaultEduWeightsPath,
  EvaluateEdu,
  FitEduLogistic,
  LoadEduWeights,
  ParseEduWeights,
  SaveEduWeights,
  ScoreEducational,
  ScoreEduVector,
  SplitHoldOut,
  TopNegativeContributions,
} from "../Brain/Data/EducationalScore.ts";
import type { EduSample } from "../Brain/Data/EducationalScore.ts";
import { EduFeatureNames, EduFeatureVector, EduSliceForLang, ExtractEduFeatures } from "../Brain/Data/EducationalFeatures.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

const Documented = `import { readFile } from "node:fs/promises";

/**
 * A small LRU cache keyed by string. Evicts the least-recently-used entry once Capacity is exceeded.
 * @example
 *   const Cache = new LruCache<number>(2);
 *   Cache.Set("a", 1);
 */
export class LruCache<Value> {
  private readonly Entries = new Map<string, Value>();

  constructor(private readonly Capacity: number) {
    if (Capacity <= 0) throw new RangeError("Capacity must be positive");
  }

  /** Returns the cached value and marks it as recently used. */
  Get(Key: string): Value | undefined {
    const Found = this.Entries.get(Key);
    if (Found === undefined) return undefined;
    // Re-insert to move the key to the most-recent position.
    this.Entries.delete(Key);
    this.Entries.set(Key, Found);
    return Found;
  }

  /** Stores a value, evicting the oldest entry when full. */
  Set(Key: string, Value: Value): void {
    this.Entries.delete(Key);
    this.Entries.set(Key, Value);
    if (this.Entries.size > this.Capacity) {
      const Oldest = this.Entries.keys().next().value;
      if (Oldest !== undefined) this.Entries.delete(Oldest);
    }
  }
}

/** Load a JSON config file, failing with the path in the message. */
export async function LoadConfig(Path: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(Path, "utf8"));
  } catch (Caught) {
    throw new Error(\`cannot load \${Path}: \${(Caught as Error).message}\`);
  }
}
`;

const Minified = "var a=function(b,c){return b+c},d=function(e){for(var f=0,g=0;g<e.length;g++)f+=e[g];return f};".repeat(40);
const StringTable = `export const Messages = {\n${Array.from({ length: 60 }, (_V, I) => `  "key_${I}": "This is translated message number ${I} for the user interface",`).join("\n")}\n};\n`;
const Boilerplate = `${Array.from({ length: 30 }, (_V, I) => `console.log("step ${I}", ${I}, ${I * 2}); // TODO remove`).join("\n")}\n`;

test("a documented, idiomatic TS module scores clearly above minified / string-table / debug boilerplate", () => {
  const Good = ScoreEducational(Documented, "typescript").Score;
  const Bad = [Minified, StringTable, Boilerplate].map((Text) => ScoreEducational(Text, "ts").Score);
  expect(Good).toBeGreaterThan(0.8);
  for (const Score of Bad) {
    expect(Score).toBeLessThan(DefaultEduWeights.Threshold);
    expect(Good - Score).toBeGreaterThan(0.6);
  }
  const StringReasons = TopNegativeContributions(ScoreEducational(StringTable, "ts"), 2);
  expect(StringReasons.length).toBe(2);
  expect(StringReasons[0]!.Name).toBe("StringLiteralRatio");
  expect(StringReasons[0]!.Contribution).toBeLessThanOrEqual(StringReasons[1]!.Contribution);
  expect(TopNegativeContributions(ScoreEducational(Minified, "js"), 3).map((C) => C.Name)).toContain("LogMaxLineLength");
});

test("features are deterministic, in vector order, with a one-hot language slice", () => {
  const A = ExtractEduFeatures(Documented, "typescript");
  const B = ExtractEduFeatures(Documented, "typescript");
  expect(A).toEqual(B);
  expect(Object.keys(A)).toEqual([...EduFeatureNames]);
  expect(EduFeatureVector(A).length).toBe(EduFeatureNames.length);
  expect(ScoreEducational(Documented, "ts").Score).toBe(ScoreEducational(Documented, "ts").Score);
  expect(A.Lang_ts).toBe(1);
  expect(EduFeatureNames.filter((Name) => Name.startsWith("Lang_")).reduce((Sum, Name) => Sum + A[Name], 0)).toBe(1);
  expect(A.DocCommentCoverage).toBeGreaterThan(0.5);
  expect(A.HasTypeAnnotations).toBe(1);
  expect(A.HasErrorHandling).toBe(1);
  expect(A.CommentRatio).toBeGreaterThan(0.05);
  expect(ExtractEduFeatures(Boilerplate, "js").ConsoleLogDensity).toBeGreaterThan(50);
  expect(ExtractEduFeatures(Boilerplate, "js").TodoDensity).toBeGreaterThan(50);
  expect(ExtractEduFeatures("func TestMax(t *testing.T) {\n\tif Max(1, 2) != 2 {\n\t\tt.Fatal(\"max\")\n\t}\n}\n", "go").HasTestSignals).toBe(1);
  for (const [Lang, Slice] of [["typescript", "ts"], ["tsx", "ts"], ["javascript", "js"], ["vue", "html"], ["scss", "css"], ["go", "go"], ["sql", "sql"], ["python", "other"]] as const) {
    expect(EduSliceForLang(Lang)).toBe(Slice);
  }
});

test("the prior is internally consistent and the shipped weights file loads", () => {
  expect(ParseEduWeights(DefaultEduWeights)).toEqual(DefaultEduWeights);
  const Shipped = LoadEduWeights(DefaultEduWeightsPath); // prior or trained: either must validate
  expect(Shipped.FeatureNames).toEqual([...EduFeatureNames]);
  expect(Shipped.Threshold).toBeGreaterThan(0);
  expect(Shipped.Threshold).toBeLessThan(1);
});

test("weights file round-trip; a missing file falls back to the prior; a mismatched one throws", () => {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-edu-"));
  try {
    const Path = join(Dir, "Nested", "EduWeights.json");
    const Custom = { ...DefaultEduWeights, Version: "test", Bias: -0.25, Threshold: 0.42, Weights: DefaultEduWeights.Weights.map((W) => W * 2) };
    SaveEduWeights(Custom, Path);
    expect(LoadEduWeights(Path)).toEqual(Custom);
    expect(LoadEduWeights(join(Dir, "absent.json"))).toBe(DefaultEduWeights);

    const Stale = join(Dir, "Stale.json");
    writeFileSync(Stale, JSON.stringify({ ...DefaultEduWeights, FeatureNames: ["OldFeature", ...EduFeatureNames.slice(1)] }));
    expect(() => LoadEduWeights(Stale)).toThrow(/FeatureNames/);
    const Short = join(Dir, "Short.json");
    writeFileSync(Short, JSON.stringify({ ...DefaultEduWeights, Mean: [0] }));
    expect(() => LoadEduWeights(Short)).toThrow(/Mean/);
    expect(() => SaveEduWeights({ ...Custom, Std: Custom.Std.map(() => 0) }, join(Dir, "Bad.json"))).toThrow();
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
});

// A synthetic, linearly separable labelling over standard-normal features (with a margin).
function SyntheticSamples(Count: number, Seed: number): EduSample[] {
  const Rng = new SeededRng(Seed);
  const Dim = EduFeatureNames.length;
  const Samples: EduSample[] = [];
  while (Samples.length < Count) {
    const Vector = Array.from({ length: Dim }, () => Rng.NextGaussian() * 3 + 5);
    const Margin = (Vector[0]! - 5) - 0.8 * (Vector[9]! - 5) + 0.5 * (Vector[3]! - 5);
    if (Math.abs(Margin) < 0.3) continue;
    Samples.push({ Vector, Label: Margin > 0 ? 1 : 0 });
  }
  return Samples;
}

test("the trainer reaches >90% held-out accuracy on a separable set, deterministically", () => {
  const Samples = SyntheticSamples(500, 11);
  const { Train, Test } = SplitHoldOut(Samples, 0.2, 7);
  expect(Test.length).toBe(100);
  expect(Train.length).toBe(400);
  const Weights = FitEduLogistic(Train, { Steps: 400, Threshold: 0.5 });
  const Metrics = EvaluateEdu(Test, Weights);
  expect(Metrics.Accuracy).toBeGreaterThan(0.9);
  expect(Metrics.Precision).toBeGreaterThan(0.85);
  expect(Metrics.Recall).toBeGreaterThan(0.85);
  // The informative features carry the largest weights, with the right signs.
  expect(Weights.Weights[0]!).toBeGreaterThan(0);
  expect(Weights.Weights[9]!).toBeLessThan(0);
  // Same data + seed → bit-identical weights; the fitted file validates.
  expect(FitEduLogistic(Train, { Steps: 400, Threshold: 0.5 })).toEqual(Weights);
  expect(ParseEduWeights(Weights)).toEqual(Weights);
  expect(ScoreEduVector(Test[0]!.Vector, Weights)).toBeGreaterThanOrEqual(0);
  // Split is a seeded permutation: same seed → same split, all items kept.
  expect(SplitHoldOut(Samples, 0.2, 7)).toEqual({ Train, Test });
  expect(() => FitEduLogistic([])).toThrow(/no labelled samples/);
  expect(() => FitEduLogistic([{ Vector: [1, 2], Label: 1 }])).toThrow(/features/);
});
