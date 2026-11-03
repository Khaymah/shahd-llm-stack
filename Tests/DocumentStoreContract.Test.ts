// One behavioral contract, run against EVERY DocumentStore implementation: the in-memory store always,
// and the Postgres store when DATABASE_URL is set (each test gets a throwaway table that is dropped
// afterwards). If the two ever disagree — on dedup accounting, ordering, search semantics, projections,
// page clamping, delete counts, or reclassification — this suite fails for the one that drifted.

import { describe, test, expect } from "bun:test";
import postgres from "postgres";
import { randomBytes } from "node:crypto";
import { InMemoryDocumentStore } from "../Foundry/InMemoryDocumentStore.ts";
import { PostgresDocumentStore } from "../Foundry/PostgresDocumentStore.ts";
import { EnsureDocumentIndexes, EnsureDocumentTable, EnsureTrigramExtension } from "../Foundry/FoundrySchema.ts";
import { DefaultPageLimit, MaxPageLimit } from "../Foundry/DocumentStore.ts";
import type { DocumentStore } from "../Foundry/DocumentStore.ts";
import type { DocumentRecord } from "../Foundry/DocumentRecord.ts";
import { IngestDocuments } from "../Foundry/Ingest.ts";
import type { SourceInput } from "../Foundry/Ingest.ts";

const DatabaseUrl = process.env["DATABASE_URL"] ?? "";
const PostgresTimeoutMs = 60_000;

type StoreHandle = { Store: DocumentStore; Cleanup: () => Promise<void> };
type StoreFactory = { Name: string; Enabled: boolean; Create: () => Promise<StoreHandle> };

const Factories: StoreFactory[] = [
  {
    Name: "InMemoryDocumentStore",
    Enabled: true,
    Create: async () => ({ Store: new InMemoryDocumentStore(), Cleanup: async () => {} }),
  },
  {
    Name: "PostgresDocumentStore",
    Enabled: DatabaseUrl !== "",
    Create: async () => {
      const Sql = postgres(DatabaseUrl, { onnotice: () => {} });
      const Table = `documents_contract_${randomBytes(6).toString("hex")}`;
      await EnsureDocumentTable(Sql, Table);
      await EnsureDocumentIndexes(Sql, Table, await EnsureTrigramExtension(Sql));
      return {
        Store: new PostgresDocumentStore(Sql, Table),
        Cleanup: async () => {
          await Sql`DROP TABLE IF EXISTS ${Sql(Table)}`;
          await Sql.end();
        },
      };
    },
  },
];

// A 32-hex-char id like the real content-hash ids, ordered by N.
function HexId(N: number): string {
  return N.toString(16).padStart(32, "0");
}

function MakeDoc(N: number, Over: Partial<DocumentRecord> = {}): DocumentRecord {
  const Content = Over.Content ?? `export const Value${N} = ${N};`;
  return {
    Id: HexId(N),
    Tier: "Filtered",
    Origin: "local",
    Source: "acme/repo",
    License: "MIT",
    Lang: "ts",
    Content,
    Bytes: Buffer.byteLength(Content, "utf8"),
    QualityScore: 0.9,
    ContentHash: `hash-${N}`,
    Embedding: [1, 0, 0, N],
    RejectReason: null,
    Provenance: `src/File${N}.ts`,
    IngestedAt: "2026-07-01T00:00:00.000Z",
    ...Over,
  };
}

async function WithStore(Factory: StoreFactory, Body: (Store: DocumentStore) => Promise<void>): Promise<void> {
  const { Store, Cleanup } = await Factory.Create();
  try {
    await Body(Store);
  } finally {
    await Cleanup();
  }
}

for (const Factory of Factories) {
  describe.skipIf(!Factory.Enabled)(`DocumentStore contract: ${Factory.Name}`, () => {
    test("Upsert: new vs duplicate accounting; the first-seen row is never rewritten", async () => {
      await WithStore(Factory, async (Store) => {
        expect(await Store.Upsert(MakeDoc(1, { Provenance: "first/seen.ts", Source: "first" }))).toBe(true);
        expect(await Store.Upsert(MakeDoc(1, { Provenance: "second/seen.ts", Source: "second", Tier: "Rejected" }))).toBe(false);
        expect(await Store.Upsert(MakeDoc(2))).toBe(true);
        expect(await Store.Count()).toBe(2);
        const Stored = await Store.DocumentById(HexId(1));
        expect(Stored?.Provenance).toBe("first/seen.ts");
        expect(Stored?.Source).toBe("first");
        expect(Stored?.Tier).toBe("Filtered");
        expect(Stored?.Embedding).toEqual([1, 0, 0, 1]);
        expect(await Store.DocumentById(HexId(99))).toBeNull();
      });
    }, PostgresTimeoutMs);

    test("ByTier: ascending id, stable across calls, Limit is a prefix, no embeddings", async () => {
      await WithStore(Factory, async (Store) => {
        for (const N of [7, 3, 11, 1, 9, 5]) await Store.Upsert(MakeDoc(N));
        await Store.Upsert(MakeDoc(4, { Tier: "Rejected" }));
        const All = await Store.ByTier("Filtered");
        expect(All.map((D) => D.Id)).toEqual([1, 3, 5, 7, 9, 11].map(HexId));
        expect((await Store.ByTier("Filtered")).map((D) => D.Id)).toEqual(All.map((D) => D.Id));
        expect((await Store.ByTier("Filtered", 3)).map((D) => D.Id)).toEqual([1, 3, 5].map(HexId));
        expect(await Store.ByTier("Filtered", 0)).toEqual([]);
        expect(All[0]?.Content).toBe("export const Value1 = 1;");
        for (const Doc of All) expect("Embedding" in Doc).toBe(false);
      });
    }, PostgresTimeoutMs);

    test("search is a literal, case-insensitive substring over content or provenance", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { Content: "call my_func() here" }));
        await Store.Upsert(MakeDoc(2, { Content: "call myXfunc() here" }));
        await Store.Upsert(MakeDoc(3, { Content: "progress: 100% done" }));
        await Store.Upsert(MakeDoc(4, { Content: "path C:\\tools\\bin" }));
        await Store.Upsert(MakeDoc(5, { Content: "nothing special", Provenance: "lib/Special_Case.ts" }));
        const Ids = async (Search: string): Promise<string[]> => {
          const Page = await Store.Query({ Search }, 0, MaxPageLimit);
          const Summaries = await Store.ListSummaries({ Filter: { Search }, Limit: MaxPageLimit });
          expect(Page.Total).toBe(Page.Rows.length);
          expect(Summaries.Total).toBe(Page.Total);
          expect(Summaries.Rows.map((R) => R.Id).sort()).toEqual(Page.Rows.map((R) => R.Id).sort());
          return Page.Rows.map((R) => R.Id).sort();
        };
        expect(await Ids("my_func")).toEqual([HexId(1)]); // `_` is literal, not a wildcard
        expect(await Ids("MY_FUNC")).toEqual([HexId(1)]); // case-insensitive
        expect(await Ids("  my_func  ")).toEqual([HexId(1)]); // trimmed
        expect(await Ids("myxfunc")).toEqual([HexId(2)]);
        expect(await Ids("%")).toEqual([HexId(3)]); // `%` is literal, not match-all
        expect(await Ids("\\tools")).toEqual([HexId(4)]); // backslash is literal
        expect(await Ids("special_case")).toEqual([HexId(5)]); // provenance matches too
        expect(await Ids("no such text")).toEqual([]);
        expect((await Store.Query({ Search: "   " }, 0, MaxPageLimit)).Total).toBe(5); // blank = no filter
      });
    }, PostgresTimeoutMs);

    test("ListSummaries: no content/embedding, newest-first, filtered Total, keyset paging", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { IngestedAt: "2026-07-01T00:00:00.000Z" }));
        await Store.Upsert(MakeDoc(2, { IngestedAt: "2026-07-03T00:00:00.000Z" }));
        await Store.Upsert(MakeDoc(3, { IngestedAt: "2026-07-03T00:00:00.000Z" }));
        await Store.Upsert(MakeDoc(4, { IngestedAt: "2026-07-02T00:00:00.000Z", Tier: "Rejected", RejectReason: "low quality" }));
        const Page = await Store.ListSummaries({});
        expect(Page.Total).toBe(4);
        // Newest first; a shared ingestedAt is tie-broken by id DESC.
        expect(Page.Rows.map((R) => R.Id)).toEqual([3, 2, 4, 1].map(HexId));
        for (const Row of Page.Rows) {
          expect("Content" in Row).toBe(false);
          expect("Embedding" in Row).toBe(false);
        }
        const Rejected = Page.Rows.find((R) => R.Id === HexId(4));
        expect(Rejected).toMatchObject({ Tier: "Rejected", RejectReason: "low quality", Source: "acme/repo", License: "MIT", Lang: "ts", Origin: "local" });
        expect(Rejected?.Bytes).toBe(Buffer.byteLength("export const Value4 = 4;", "utf8"));

        const Second = await Store.ListSummaries({ Offset: 2, Limit: 2 });
        expect(Second.Rows.map((R) => R.Id)).toEqual([4, 1].map(HexId));
        const Filtered = await Store.ListSummaries({ Filter: { Tier: "Filtered" } });
        expect(Filtered.Total).toBe(3);

        // Keyset: ascending id strictly after the cursor; Total still counts the whole filter.
        const K1 = await Store.ListSummaries({ AfterId: "", Limit: 2 });
        expect(K1.Rows.map((R) => R.Id)).toEqual([1, 2].map(HexId));
        expect(K1.Total).toBe(4);
        const K2 = await Store.ListSummaries({ AfterId: HexId(2), Limit: 2 });
        expect(K2.Rows.map((R) => R.Id)).toEqual([3, 4].map(HexId));
        expect((await Store.ListSummaries({ AfterId: HexId(4) })).Rows).toEqual([]);

        const Contents = await Store.ContentByIds([HexId(1), HexId(3), HexId(99)]);
        expect([...Contents.keys()].sort()).toEqual([HexId(1), HexId(3)]);
        expect(Contents.get(HexId(3))).toBe("export const Value3 = 3;");
        expect((await Store.ContentByIds([])).size).toBe(0);
      });
    }, PostgresTimeoutMs);

    test("page limits are clamped to MaxPageLimit; non-positive limits use the default", async () => {
      await WithStore(Factory, async (Store) => {
        const Extra = 5;
        const N = MaxPageLimit + Extra;
        for (let I = 1; I <= N; I++) await Store.Upsert(MakeDoc(I, { Source: I % 2 === 0 ? "even" : "odd" }));
        const Big = await Store.Query({}, 0, MaxPageLimit * 10);
        expect(Big.Rows.length).toBe(MaxPageLimit);
        expect(Big.Total).toBe(N);
        expect((await Store.ListSummaries({ Limit: MaxPageLimit * 10 })).Rows.length).toBe(MaxPageLimit);
        expect((await Store.ListSummaries({ Limit: 0 })).Rows.length).toBe(DefaultPageLimit);
        expect((await Store.ListSummaries({ Limit: -3 })).Rows.length).toBe(DefaultPageLimit);
        expect((await Store.Query({}, 0, 0)).Rows.length).toBe(DefaultPageLimit);
        expect((await Store.Query({}, -10, 3)).Rows.length).toBe(3); // negative offset → 0
        expect((await Store.ListSummaries({ Offset: N - 2, Limit: 10 })).Rows.length).toBe(2);
        expect((await Store.DocumentsBySource("even", MaxPageLimit * 10)).length).toBeLessThanOrEqual(MaxPageLimit);
        const AllIds = Array.from({ length: N }, (_V, I) => HexId(I + 1));
        expect((await Store.ContentByIds(AllIds)).size).toBe(MaxPageLimit);
        // ByTier is a training read: its Limit is the caller's size budget, not a page — never clamped.
        expect((await Store.ByTier("Filtered")).length).toBe(N);
      });
    }, PostgresTimeoutMs);

    test("DeleteMatching / DeleteById report exactly how many rows were removed", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { Lang: "ts" }));
        await Store.Upsert(MakeDoc(2, { Lang: "ts", Content: "contains needle_x" }));
        await Store.Upsert(MakeDoc(3, { Lang: "go" }));
        await Store.Upsert(MakeDoc(4, { Lang: "go", Tier: "Rejected" }));
        expect(await Store.DeleteMatching({ Lang: "rust" })).toBe(0);
        expect(await Store.DeleteMatching({ Search: "needle%x" })).toBe(0); // literal %, so no match
        expect(await Store.DeleteMatching({ Search: "NEEDLE_X" })).toBe(1);
        expect(await Store.DeleteMatching({ Lang: "go", Tier: "Filtered" })).toBe(1);
        expect(await Store.Count()).toBe(2);
        expect(await Store.DeleteById(HexId(4))).toBe(1);
        expect(await Store.DeleteById(HexId(4))).toBe(0);
        expect((await Store.All()).map((D) => D.Id)).toEqual([HexId(1)]);
      });
    }, PostgresTimeoutMs);

    test("ReclassifyBySource promotes quality NOASSERTION docs and keeps the rest, once", async () => {
      await WithStore(Factory, async (Store) => {
        const Unresolved: Partial<DocumentRecord> = { Source: "acme/lib", License: "NOASSERTION", Tier: "Rejected", RejectReason: "license NOASSERTION" };
        await Store.Upsert(MakeDoc(1, { ...Unresolved, QualityScore: 0.9 }));
        await Store.Upsert(MakeDoc(2, { ...Unresolved, QualityScore: 0.6 }));
        await Store.Upsert(MakeDoc(3, { ...Unresolved, QualityScore: 0.2 }));
        await Store.Upsert(MakeDoc(4, { ...Unresolved, Source: "other/repo", QualityScore: 0.9 }));
        await Store.Upsert(MakeDoc(5, { Source: "acme/lib", License: "GPL-3.0", Tier: "Rejected", RejectReason: "copyleft" }));

        expect(await Store.ReclassifyBySource("acme/lib", "MIT", 0.6)).toEqual({ Promoted: 2, KeptLowQuality: 1 });
        const D1 = await Store.DocumentById(HexId(1));
        const D2 = await Store.DocumentById(HexId(2));
        const D3 = await Store.DocumentById(HexId(3));
        expect(D1).toMatchObject({ Tier: "Filtered", License: "MIT", RejectReason: null });
        expect(D2).toMatchObject({ Tier: "Filtered", License: "MIT", RejectReason: null });
        expect(D3).toMatchObject({ Tier: "Rejected", License: "MIT", RejectReason: "low quality (score < 0.6)" });
        expect(await Store.DocumentById(HexId(4))).toMatchObject({ License: "NOASSERTION", Tier: "Rejected" }); // other source
        expect(await Store.DocumentById(HexId(5))).toMatchObject({ License: "GPL-3.0", RejectReason: "copyleft" }); // not NOASSERTION
        expect(await Store.ReclassifyBySource("acme/lib", "MIT", 0.6)).toEqual({ Promoted: 0, KeptLowQuality: 0 }); // idempotent
      });
    }, PostgresTimeoutMs);

    test("Retier reclassifies one row explicitly; RepoSummaries and Stats honor it", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { Tier: "Rejected", RejectReason: "minified?", QualityScore: 0.1, Origin: "curated" }));
        await Store.Upsert(MakeDoc(2, { Source: "b/repo", License: "NOASSERTION" }));
        expect(await Store.Retier(HexId(1), "Filtered", null, 0.8)).toBe(true);
        expect(await Store.Retier(HexId(99), "Filtered", null)).toBe(false);
        expect(await Store.DocumentById(HexId(1))).toMatchObject({ Tier: "Filtered", RejectReason: null, QualityScore: 0.8 });
        expect(await Store.Retier(HexId(2), "Rejected", "manual")).toBe(true);
        expect(await Store.DocumentById(HexId(2))).toMatchObject({ Tier: "Rejected", RejectReason: "manual", QualityScore: 0.9 }); // score untouched

        expect(await Store.RepoSummaries({ License: "NOASSERTION" })).toEqual([{ Source: "b/repo", Files: 1, Bytes: MakeDoc(2).Bytes }]);
        expect((await Store.RepoSummaries()).length).toBe(2);
        expect((await Store.ListSummaries({ Filter: { Origin: "curated" } })).Total).toBe(1);
        const Stats = await Store.Stats();
        expect(Stats.Total).toBe(2);
        expect(Stats.ByTier).toEqual({ Filtered: 1, Raw: 0, Rejected: 1 });
        expect(Stats.FilteredBytes).toBe(MakeDoc(1).Bytes);
      });
    }, PostgresTimeoutMs);

    test("ExistingContentHashes: batched, any tier or one tier, absent hashes omitted", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { ContentHash: "h-filtered" }));
        await Store.Upsert(MakeDoc(2, { ContentHash: "h-rejected", Tier: "Rejected" }));
        await Store.Upsert(MakeDoc(3, { ContentHash: "h-filtered", License: "Apache-2.0" })); // same bytes, other license
        expect(await Store.ExistingContentHashes([])).toEqual(new Set());
        expect(await Store.ExistingContentHashes(["h-filtered", "h-rejected", "h-absent", "h-filtered"])).toEqual(new Set(["h-filtered", "h-rejected"]));
        expect(await Store.ExistingContentHashes(["h-filtered", "h-rejected"], "Filtered")).toEqual(new Set(["h-filtered"]));
        expect(await Store.ExistingContentHashes(["h-filtered", "h-rejected"], "Raw")).toEqual(new Set());
        // More hashes than one lookup chunk: still answered completely.
        const Many = Array.from({ length: MaxPageLimit + 7 }, (_V, I) => `h-miss-${I}`);
        expect(await Store.ExistingContentHashes([...Many, "h-rejected"])).toEqual(new Set(["h-rejected"]));
      });
    }, PostgresTimeoutMs);

    test("IngestDocuments: the same bytes under two licenses are one row; duplicates are counted", async () => {
      await WithStore(Factory, async (Store) => {
        const Content = "package mathx\n\n// Max returns the larger of a and b.\nfunc Max(a, b int) int {\n\tif a > b {\n\t\treturn a\n\t}\n\treturn b\n}\n";
        const Mit: SourceInput = { Source: "a/lib", License: "MIT", Lang: "go", Content, Provenance: "a/max.go", Origin: "web-permissive" };
        const First = await IngestDocuments([Mit, { ...Mit, License: "Apache-2.0", Source: "b/fork", Provenance: "b/max.go" }], Store, "2026-07-01T00:00:00.000Z");
        expect(First).toMatchObject({ New: 1, Duplicate: 1, Ingested: 2, Failed: 0 });
        const Second = await IngestDocuments([{ ...Mit, License: "ISC", Source: "c/fork", Provenance: "c/max.go" }], Store, "2026-07-02T00:00:00.000Z");
        expect(Second).toMatchObject({ New: 0, Duplicate: 1, Failed: 0 });
        expect(await Store.Count()).toBe(1);
        const Stored = await Store.ByTier("Filtered");
        expect(Stored.map((D) => [D.Source, D.License])).toEqual([["a/lib", "MIT"]]);
        expect(await Store.ExistingContentHashes([Stored[0]!.ContentHash], "Filtered")).toEqual(new Set([Stored[0]!.ContentHash]));
      });
    }, PostgresTimeoutMs);

    test("FindSimilar ranks by cosine and returns full records", async () => {
      await WithStore(Factory, async (Store) => {
        await Store.Upsert(MakeDoc(1, { Embedding: [1, 0] }));
        await Store.Upsert(MakeDoc(2, { Embedding: [0, 1] }));
        await Store.Upsert(MakeDoc(3, { Embedding: [0.9, 0.1] }));
        const Hits = await Store.FindSimilar([1, 0], 2);
        expect(Hits.map((H) => H.Doc.Id)).toEqual([HexId(1), HexId(3)]);
        expect(Hits[0]?.Doc.Content).toBe("export const Value1 = 1;");
        expect(await Store.FindSimilar([1, 0], 0)).toEqual([]);
      });
    }, PostgresTimeoutMs);
  });
}

describe.skipIf(DatabaseUrl === "")("EnsureDocumentIndexes (Postgres)", () => {
  test("creates the content_hash index used by the cross-license dedup", async () => {
    const Sql = postgres(DatabaseUrl, { onnotice: () => {} });
    const Table = `documents_index_${randomBytes(6).toString("hex")}`;
    try {
      await EnsureDocumentTable(Sql, Table);
      await EnsureDocumentIndexes(Sql, Table, false);
      await EnsureDocumentIndexes(Sql, Table, false); // idempotent
      const Rows = await Sql<{ indexdef: string }[]>`select indexdef from pg_indexes where tablename = ${Table} and indexname = ${`${Table}_content_hash_idx`}`;
      expect(Rows.length).toBe(1);
      expect(Rows[0]!.indexdef).toContain("(content_hash)");
    } finally {
      await Sql`DROP TABLE IF EXISTS ${Sql(Table)}`;
      await Sql.end();
    }
  }, PostgresTimeoutMs);
});
