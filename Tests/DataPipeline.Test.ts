import { test, expect } from "bun:test";
import { LicenseManifest } from "../Brain/Data/LicenseManifest.ts";
import { NearDuplicateGroups, DedupedIndices } from "../Brain/Data/NearDedup.ts";
import { ScoreCodeQuality, ScoreTextQuality } from "../Brain/Data/QualityFilter.ts";
import { ToFim, FromFim, FimTokens } from "../Brain/Data/FimReformat.ts";
import { Decontaminate } from "../Brain/Data/Decontamination.ts";
import { SplitAndEncodeDocuments } from "../Brain/Data/TrainValSplit.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

test("license manifest filters to permissive and summarizes", () => {
  const Manifest = new LicenseManifest();
  Manifest.Add({ Source: "repoA", License: "MIT", Path: "a.ts", Bytes: 100, IngestedAt: "2026-07-13" });
  Manifest.Add({ Source: "repoB", License: "GPL-3.0", Path: "b.ts", Bytes: 50, IngestedAt: "2026-07-13" });
  expect(Manifest.PermissiveOnly().length).toBe(1);
  expect(Manifest.Summary().TotalBytes).toBe(150);
  expect(LicenseManifest.FromJson(Manifest.ToJson()).Entries.length).toBe(2);
});

test("near-dedup groups near-identical documents and keeps distinct ones", () => {
  const A = "function add(a, b) { return a + b; } // small helper used across the codebase";
  const B = "function add(a, b) { return a + b; } // small helper used across the whole codebase";
  const C = "class Server { constructor() { this.port = 8080; } listen() { startAll(); } }";
  const Groups = NearDuplicateGroups([A, B, C], 0.6);
  const GroupOf = (Index: number): number => Groups.findIndex((G) => G.includes(Index));
  expect(GroupOf(0)).toBe(GroupOf(1));
  expect(GroupOf(2)).not.toBe(GroupOf(0));
  expect(DedupedIndices([A, B, C], 0.6).length).toBe(2);
});

test("quality filter keeps clean code and drops minified/binary junk", () => {
  const Clean = "function add(a, b) {\n  return a + b;\n}\n\nconst x = add(1, 2);\n";
  expect(ScoreCodeQuality(Clean).Passed).toBe(true);
  expect(ScoreCodeQuality("a".repeat(2000)).Passed).toBe(false);
  let Binary = "";
  for (let I = 0; I < 200; I++) Binary += String.fromCharCode(I);
  expect(ScoreCodeQuality(Binary).Passed).toBe(false);
});

test("comment-ratio and tests signals apply to code only, never to prose", () => {
  const Prose = "The river runs through the old city. Its bridges were rebuilt after the flood of 1902.\n".repeat(4);
  const Text = ScoreTextQuality(Prose);
  expect(Text.Score).toBe(1);
  expect(Text.Reasons).toEqual([]);
  const AsCode = ScoreCodeQuality(Prose);
  expect(AsCode.Reasons.join(";")).toContain("no comments"); // code scoring sees zero comment lines
  expect(AsCode.Passed).toBe(true);
});

test("FIM PSM round-trips to the original", () => {
  const Doc = "function f(x) {\n  return x * 2;\n}\n";
  const Fim = ToFim(Doc, 10, 25, "Psm");
  expect(Fim).toContain(FimTokens.Prefix);
  expect(Fim).toContain(FimTokens.Middle);
  expect(FromFim(Fim)).toBe(Doc);
});

test("SplitAndEncodeDocuments splits at the document boundary (no leak) with EOS separators", () => {
  const Docs = ["alpha alpha", "beta beta", "gamma gamma", "delta delta", "epsilon epsilon", "zeta zeta"];
  const EosId = 9999;
  const Encode = (T: string): number[] => Array.from(T).map((C) => C.charCodeAt(0)); // deterministic char encode
  const { Train, Val } = SplitAndEncodeDocuments(Docs, 0.25, new SeededRng(7), Encode, EosId);
  expect(Train.length).toBeGreaterThan(0);
  expect(Val.length).toBeGreaterThan(0);
  expect(Train[Train.length - 1]).toBe(EosId); // every doc ends in EOS
  expect(Val[Val.length - 1]).toBe(EosId);
  const ToDocs = (Stream: number[]): string[] => {
    const Out: string[] = [];
    let Cur = "";
    for (const Id of Stream) {
      if (Id === EosId) { Out.push(Cur); Cur = ""; } else Cur += String.fromCharCode(Id);
    }
    return Out;
  };
  const TrainDocs = new Set(ToDocs(Train));
  const ValDocs = ToDocs(Val);
  for (const D of ValDocs) expect(TrainDocs.has(D)).toBe(false); // NO leak: disjoint document sets
  expect(TrainDocs.size + ValDocs.length).toBe(Docs.length); // every document accounted for
});

test("SplitAndEncodeDocuments requires at least two documents", () => {
  expect(() => SplitAndEncodeDocuments(["only one"], 0.1, new SeededRng(1), (T) => [T.length], 0)).toThrow(/>= 2/);
});

test("decontamination removes train docs overlapping the eval set", () => {
  const EvalSet = ["the quick brown fox jumps over the lazy dog again and once more today please"];
  const Train = [
    "the quick brown fox jumps over the lazy dog again and once more today please",
    "unrelated snippet about servers databases networking stacks here now",
  ];
  const Result = Decontaminate(Train, EvalSet, 13);
  expect(Result.Removed).toContain(0);
  expect(Result.Kept).toContain(1);
});

// Decontamination fires on shared CONTENT (13 consecutive words), never on shared idioms: a Go
// handler signature is punctuation-heavy and common to every handler, so it must not remove the
// document, while a verbatim copy of an eval reference must.
test("decontamination ignores punctuation idioms but catches verbatim eval content", () => {
  const EvalDoc = [
    "function slugify(input: string): string {",
    "  const lowered = input.trim().toLowerCase();",
    "  const dashed = lowered.replace(/[^a-z0-9]+/g, '-');",
    "  return dashed.replace(/^-+|-+$/g, '');",
    "}",
  ].join("\n");
  const Idiom = "func Health(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }";
  const IdiomDoc = `package api\n${Idiom}\nfunc Ready(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) }`;
  const CopyDoc = `// utils\n${EvalDoc}\nexport const Other = 1;`;
  const Result = Decontaminate([IdiomDoc, CopyDoc, "const unrelated = 42;"], [EvalDoc, "func HealthHandler(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK); w.Write([]byte(\"ok\")) }"]);
  expect(Result.Removed).toEqual([1]);
  expect(Result.Kept).toEqual([0, 2]);
});
