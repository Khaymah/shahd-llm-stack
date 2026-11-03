import { test, expect } from "bun:test";
import {
  ClassifyDocument,
  HashingEmbedding,
  CosineSimilarity,
  InMemoryDocumentStore,
  IngestDocuments,
  ExportTrainingText,
  BuildReport,
  RenderReportText,
} from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, DocumentRecord } from "../Foundry/FoundryBarrel.ts";
import { IngestFailureBudget, IngestAbortError, MaxConsecutiveIngestFailures } from "../Foundry/Ingest.ts";

const Clean = "export function add(a, b) {\n  return a + b;\n}\n";
const GoSnippet = "package main\n\nfunc Max(a, b int) int {\n\tif a > b {\n\t\treturn a\n\t}\n\treturn b\n}\n";
const CSnippet = "#include <stdio.h>\nint main(void) {\n  printf(\"hi\\n\");\n  return 0;\n}\n";
const Minified = "var a=1;" + "x=x+1;".repeat(90);

test("tiering: permissive+clean -> Filtered, non-permissive -> Rejected, general web -> Raw", () => {
  expect(ClassifyDocument("MIT", Clean, "local").Tier).toBe("Filtered");
  expect(ClassifyDocument("GPL-3.0", Clean, "local").Tier).toBe("Rejected");
  expect(ClassifyDocument("MIT", Minified, "local").Tier).toBe("Rejected"); // low quality
  const Web = ClassifyDocument("MIT", Clean, "web-general");
  expect(Web.Tier).toBe("Raw"); // isolated regardless of license
  expect(String(Web.RejectReason)).toContain("isolated");
});

test("embedding: identical text is more similar than unrelated text", () => {
  const A = HashingEmbedding(Clean);
  const B = HashingEmbedding(Clean + " // a comment");
  const C = HashingEmbedding("def multiply(x, y):\n    return x * y\n");
  expect(CosineSimilarity(A, A)).toBeCloseTo(1, 6);
  expect(CosineSimilarity(A, B)).toBeGreaterThan(CosineSimilarity(A, C));
});

test("ingest tiers documents into the store and export yields only Filtered", async () => {
  const Store = new InMemoryDocumentStore();
  const Inputs: SourceInput[] = [
    { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
    { Source: "s", License: "GPL-3.0", Lang: "c", Content: CSnippet, Provenance: "b.c", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "js", Content: Minified, Provenance: "m.js", Origin: "local" },
    { Source: "web", License: "unknown", Lang: "ts", Content: "const q = fetchFromWeb();\n", Provenance: "http://x", Origin: "web-general" },
  ];
  const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  expect(Stats.Ingested).toBe(4);
  expect(Stats.ByTier.Filtered).toBe(1);
  expect(Stats.ByTier.Rejected).toBe(2);
  expect(Stats.ByTier.Raw).toBe(1);

  const Text = await ExportTrainingText(Store);
  expect(Text).toContain("export function add"); // filtered content present
  expect(Text).not.toContain("fetchFromWeb"); // isolated web content excluded from training
});

test("quality report aggregates tiers, licenses, and langs", async () => {
  const Store = new InMemoryDocumentStore();
  await IngestDocuments(
    [
      { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
      { Source: "s", License: "Apache-2.0", Lang: "go", Content: GoSnippet, Provenance: "c.go", Origin: "local" },
      { Source: "s", License: "GPL-3.0", Lang: "c", Content: CSnippet, Provenance: "b.c", Origin: "local" },
    ],
    Store,
    "2026-07-13T00:00:00.000Z",
  );
  const Report = BuildReport(await Store.All());
  expect(Report.Total).toBe(3);
  expect(Report.ByTier.Filtered).toBe(2);
  expect(Report.ByLicense["GPL-3.0"]).toBe(1);
  expect(Report.FilteredBytes).toBeGreaterThan(0);
  expect(RenderReportText(Report)).toContain("Filtered=2");
});

test("dedup by content hash: re-ingesting identical content does not duplicate", async () => {
  const Store = new InMemoryDocumentStore();
  const One: SourceInput = { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" };
  const Stats = await IngestDocuments([One, One], Store, "2026-07-13T00:00:00.000Z");
  expect(await Store.Count()).toBe(1);
  // Honest accounting: the first is a NEW row, the second is a content-hash duplicate — not fresh data.
  expect(Stats.New).toBe(1);
  expect(Stats.Duplicate).toBe(1);
  expect(Stats.Ingested).toBe(2); // Ingested stays New + Duplicate (successful upserts) for back-compat
});

test("ingest is resilient: one failing Upsert doesn't abort the batch", async () => {
  // A store that rejects the 2nd row (mimics Postgres refusing an unstorable value): the run must
  // continue and report the failure, not throw and lose the other rows.
  class FlakyStore extends InMemoryDocumentStore {
    Calls = 0;
    override async Upsert(Doc: DocumentRecord): Promise<boolean> {
      this.Calls++;
      if (this.Calls === 2) throw new Error("simulated store failure");
      return super.Upsert(Doc);
    }
  }
  const Store = new FlakyStore();
  const Inputs: SourceInput[] = [
    { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "go", Content: GoSnippet, Provenance: "b.go", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "c", Content: CSnippet, Provenance: "c.c", Origin: "local" },
  ];
  const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  expect(Stats.Ingested).toBe(2);
  expect(Stats.Failed).toBe(1);
  expect(await Store.Count()).toBe(2);
});

test("ReclassifyBySource promotes quality NOASSERTION docs to Filtered, keeps low-quality rejected", async () => {
  const Store = new InMemoryDocumentStore();
  await IngestDocuments(
    [
      { Source: "acme/lib", License: "NOASSERTION", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "web-permissive" },
      { Source: "acme/lib", License: "NOASSERTION", Lang: "js", Content: Minified, Provenance: "m.js", Origin: "web-permissive" }, // low quality
      { Source: "other/lib", License: "NOASSERTION", Lang: "go", Content: GoSnippet, Provenance: "g.go", Origin: "web-permissive" }, // different repo — untouched
    ],
    Store,
    "2026-07-13T00:00:00.000Z",
  );
  expect((await Store.ByTier("Rejected")).length).toBe(3); // all NOASSERTION => Rejected on license

  const Res = await Store.ReclassifyBySource("acme/lib", "MIT", 0.6);
  expect(Res).toEqual({ Promoted: 1, KeptLowQuality: 1 });
  const Filtered = await Store.ByTier("Filtered");
  expect(Filtered.length).toBe(1);
  expect(Filtered[0]!.License).toBe("MIT");
  expect(Filtered[0]!.RejectReason).toBeNull();
  // the other repo is untouched (scoped by source); low-quality doc stays Rejected with a quality reason
  expect((await Store.ByTier("Rejected")).length).toBe(2);
  const StillRejected = (await Store.ByTier("Rejected")).find((D) => D.Source === "acme/lib");
  expect(String(StillRejected?.RejectReason)).toContain("low quality");
});

test("provenance-aware dedup: a Raw/Rejected copy never blocks or clobbers the training copy", async () => {
  // Raw first, then the local MIT copy of the same bytes: the training copy is still stored.
  const RawFirst = new InMemoryDocumentStore();
  const RawStats = await IngestDocuments(
    [
      { Source: "web", License: "unknown", Lang: "ts", Content: Clean, Provenance: "http://x", Origin: "web-general" },
      { Source: "local", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
    ],
    RawFirst,
    "2026-07-13T00:00:00.000Z",
  );
  expect(RawStats).toMatchObject({ New: 2, Duplicate: 0 });
  expect((await RawFirst.ByTier("Filtered")).length).toBe(1); // local MIT is training-eligible
  expect((await RawFirst.ByTier("Raw")).length).toBe(1); // general web stays isolated

  // Filtered first: the later Raw copy of the same bytes is a content duplicate (not stored) and the
  // Filtered row is untouched.
  const FilteredFirst = new InMemoryDocumentStore();
  const Stats = await IngestDocuments(
    [
      { Source: "local", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
      { Source: "web", License: "unknown", Lang: "ts", Content: Clean, Provenance: "http://x", Origin: "web-general" },
    ],
    FilteredFirst,
    "2026-07-13T00:00:00.000Z",
  );
  expect(Stats).toMatchObject({ New: 1, Duplicate: 1, Ingested: 2 });
  expect(await FilteredFirst.Count()).toBe(1);
  expect((await FilteredFirst.ByTier("Filtered"))[0]?.Provenance).toBe("a.ts");
});

test("exact dedup across licenses: the same bytes under MIT and Apache are stored once", async () => {
  const Store = new InMemoryDocumentStore();
  const Mit: SourceInput = { Source: "a/lib", License: "MIT", Lang: "go", Content: GoSnippet, Provenance: "a.go", Origin: "web-permissive" };
  const Apache: SourceInput = { ...Mit, Source: "b/fork", License: "Apache-2.0", Provenance: "b.go" };
  // In-batch: the second copy is a duplicate of the first.
  const First = await IngestDocuments([Mit, Apache], Store, "2026-07-13T00:00:00.000Z");
  expect(First).toMatchObject({ New: 1, Duplicate: 1, Ingested: 2, Failed: 0 });
  expect(First.ByTier.Filtered).toBe(2); // both copies were judged Filtered; only one row exists
  expect(await Store.Count()).toBe(1);
  // Across batches: the stored hash is found by the batched ExistingContentHashes lookup.
  const Second = await IngestDocuments([{ ...Mit, License: "BSD-3-Clause", Source: "c/fork", Provenance: "c.go" }], Store, "2026-07-14T00:00:00.000Z");
  expect(Second).toMatchObject({ New: 0, Duplicate: 1 });
  expect(await Store.Count()).toBe(1);
  expect((await Store.All())[0]?.Source).toBe("a/lib"); // first-seen provenance wins

  // A Rejected copy (GPL) of bytes already stored is a duplicate too: no redundant audit row.
  const Gpl = await IngestDocuments([{ ...Mit, License: "GPL-3.0", Provenance: "d.go" }], Store, "2026-07-14T00:00:00.000Z");
  expect(Gpl).toMatchObject({ New: 0, Duplicate: 1 });
  expect(Gpl.ByTier.Rejected).toBe(1);
});

test("a rejected copy stored first does not block the permissive training copy", async () => {
  const Store = new InMemoryDocumentStore();
  const Gpl: SourceInput = { Source: "g/lib", License: "GPL-3.0", Lang: "go", Content: GoSnippet, Provenance: "g.go", Origin: "web-permissive" };
  await IngestDocuments([Gpl], Store, "2026-07-13T00:00:00.000Z");
  const Stats = await IngestDocuments([{ ...Gpl, License: "MIT", Source: "m/lib", Provenance: "m.go" }], Store, "2026-07-13T00:00:00.000Z");
  expect(Stats).toMatchObject({ New: 1, Duplicate: 0 });
  expect((await Store.ByTier("Filtered")).length).toBe(1);
  expect((await Store.ByTier("Rejected")).length).toBe(1);
});

test("a failed content-hash lookup degrades to in-batch dedup with a warning", async () => {
  class LookupDownStore extends InMemoryDocumentStore {
    override async ExistingContentHashes(): Promise<Set<string>> {
      throw new Error("lookup down");
    }
  }
  const Warnings: string[] = [];
  const One: SourceInput = { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" };
  const Stats = await IngestDocuments([One, { ...One, License: "ISC" }], new LookupDownStore(), "2026-07-13T00:00:00.000Z", 256, undefined, new IngestFailureBudget((M) => Warnings.push(M)));
  expect(Stats).toMatchObject({ New: 1, Duplicate: 1, Failed: 0 });
  expect(Warnings.join("\n")).toContain("lookup down");
});

test("tiering: code below the educational threshold is Rejected with its two worst features", () => {
  const StringTable = `export const Messages = {\n${Array.from({ length: 40 }, (_V, I) => `  Key${I}: "This is the translated user interface message number ${I}",`).join("\n")}\n};\n`;
  const Decision = ClassifyDocument("MIT", StringTable, "web-permissive", { Lang: "typescript" });
  expect(Decision.Tier).toBe("Rejected");
  expect(String(Decision.RejectReason)).toMatch(/^low educational value: \w+=.*, \w+=/);
  expect(String(Decision.RejectReason)).toContain("StringLiteralRatio");
  expect(Decision.QualityScore).toBeLessThan(0.3);
  // Without a code Lang (or as curated prose) the educational gate does not apply.
  expect(ClassifyDocument("MIT", StringTable, "web-permissive").RejectReason ?? "").not.toContain("educational");
  expect(ClassifyDocument("MIT", StringTable, "curated", { Lang: "typescript" }).RejectReason ?? "").not.toContain("educational");
});

test("tiering: QualityScore is the educational score for code that passes the structural gate", () => {
  const Documented = `/**\n * Clamp a value into [Low, High].\n * @example Clamp(5, 0, 3) // 3\n */\nexport function Clamp(Value: number, Low: number, High: number): number {\n  if (Low > High) throw new RangeError("Low must not exceed High");\n  // Math.min/max keep NaN out of the happy path.\n  return Math.min(High, Math.max(Low, Value));\n}\n`;
  const Strict = ClassifyDocument("MIT", Documented, "web-permissive", { Lang: "ts" });
  expect(Strict.Tier).toBe("Filtered");
  const Structural = ClassifyDocument("MIT", Documented, "web-permissive");
  expect(Strict.QualityScore).not.toBe(Structural.QualityScore);
  // A license-rejected doc still records the educational score (what a license backfill judges).
  expect(ClassifyDocument("NOASSERTION", Documented, "web-permissive", { Lang: "ts" }).QualityScore).toBe(Strict.QualityScore);
});

// A store whose Upsert fails for every row the predicate selects (by 0-based call index).
class SelectiveFailStore extends InMemoryDocumentStore {
  Calls = 0;
  private FailWhen: (Call: number) => boolean;
  constructor(FailWhen: (Call: number) => boolean) {
    super();
    this.FailWhen = FailWhen;
  }
  override async Upsert(Doc: DocumentRecord): Promise<boolean> {
    const Call = this.Calls++;
    if (this.FailWhen(Call)) throw new Error(`simulated failure ${Call}`);
    return super.Upsert(Doc);
  }
}
const UniqueInputs = (Count: number): SourceInput[] =>
  Array.from({ length: Count }, (_, I) => ({ Source: "s", License: "MIT", Lang: "ts", Content: `${Clean}// ${I}\n`, Provenance: `f${I}.ts`, Origin: "local" as const }));

test("ingest aborts after MaxConsecutiveIngestFailures in a row, carrying the partial stats", async () => {
  const Warnings: string[] = [];
  const Store = new SelectiveFailStore((Call) => Call >= 3); // 3 good rows, then the store is "down"
  let Failure: unknown = null;
  try {
    await IngestDocuments(UniqueInputs(40), Store, "2026-07-13T00:00:00.000Z", 256, undefined, new IngestFailureBudget((M) => Warnings.push(M)));
  } catch (Caught) {
    Failure = Caught;
  }
  expect(Failure).toBeInstanceOf(IngestAbortError);
  const Abort = Failure as IngestAbortError;
  expect(Abort.message).toContain(`${MaxConsecutiveIngestFailures} consecutive`);
  expect(Abort.Stats.Ingested).toBe(3);
  expect(Abort.Stats.Failed).toBe(MaxConsecutiveIngestFailures);
  expect(Store.Calls).toBe(3 + MaxConsecutiveIngestFailures); // stopped right there, no churn
  expect(Warnings.length).toBe(1); // first failure logged in full; repeats suppressed
  expect(Warnings[0]).toContain("simulated failure 3");
});

test("ingest aborts when >20% of >=50 attempts failed, even with no long failure streak", async () => {
  const Store = new SelectiveFailStore((Call) => Call % 4 === 0); // every 4th row fails: 25%
  let Failure: unknown = null;
  try {
    await IngestDocuments(UniqueInputs(200), Store, "2026-07-13T00:00:00.000Z", 256, undefined, new IngestFailureBudget(() => {}));
  } catch (Caught) {
    Failure = Caught;
  }
  expect(Failure).toBeInstanceOf(IngestAbortError);
  expect((Failure as Error).message).toContain("upserts failed");
  expect(Store.Calls).toBeLessThan(200); // aborted early, at the first check past 50 attempts
  expect(Store.Calls).toBeGreaterThanOrEqual(50);
});

test("ingest failure budget is run-wide: failures accumulate across IngestDocuments calls", async () => {
  const Budget = new IngestFailureBudget(() => {});
  const Store = new SelectiveFailStore(() => true);
  // 6 + 6 consecutive failures over two batches trip the 10-in-a-row limit in the SECOND call.
  const First = await IngestDocuments(UniqueInputs(6), Store, "2026-07-13T00:00:00.000Z", 256, undefined, Budget);
  expect(First.Failed).toBe(6);
  let Failure: unknown = null;
  try {
    await IngestDocuments(UniqueInputs(6), Store, "2026-07-13T00:00:00.000Z", 256, undefined, Budget);
  } catch (Caught) {
    Failure = Caught;
  }
  expect(Failure).toBeInstanceOf(IngestAbortError);
  expect(Budget.Failed).toBe(MaxConsecutiveIngestFailures);
});
