// Postgres implementation of DocumentStore (M3b). Same interface and semantics as the in-memory store
// (Tests/DocumentStoreContract runs one suite over both), so ingestion / reports / dashboard work
// unchanged against a real database. The table is a constructor parameter (default "documents") so
// each data kind gets its own physically-separate table (documents_code, documents_conversation, …)
// via the same store class. Tables + indexes are created by Scripts/FoundryMigrate (EnsureDocumentTable).
//
// Every read names its columns explicitly (never `select *`): browse pages skip content+embedding,
// training reads skip the embedding, aggregates touch neither.

import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { eq, sql, desc, and, or, inArray } from "drizzle-orm";
import type { SQL } from "drizzle-orm";
import { DocumentsTable, LegacyDocumentsTableName } from "./FoundrySchema.ts";
import type { DocumentRow, DocumentInsert, DocumentTable } from "./FoundrySchema.ts";
import { ClampOffset, ClampPageLimit, MaxPageLimit, NormalizeSearch } from "./DocumentStore.ts";
import type { DocumentStore, SimilarHit, RepoSummary, FoundryStats, DocumentFilter, DocumentPage, DocumentSummary, ListOptions, SummaryPage, TierDocument } from "./DocumentStore.ts";
import type { DocumentRecord, Tier } from "./DocumentRecord.ts";
import { CosineSimilarity } from "./Embedding.ts";

type SummaryRow = Omit<DocumentRow, "content" | "embedding">;
type TierRow = Omit<DocumentRow, "embedding">;

// Column projections (explicit lists, so adding a heavy column later can never leak into a page read).
function SummaryColumns(T: DocumentTable) {
  return {
    id: T.id,
    tier: T.tier,
    origin: T.origin,
    source: T.source,
    license: T.license,
    lang: T.lang,
    bytes: T.bytes,
    quality: T.quality,
    contentHash: T.contentHash,
    rejectReason: T.rejectReason,
    provenance: T.provenance,
    ingestedAt: T.ingestedAt,
  };
}

function TierColumns(T: DocumentTable) {
  return { ...SummaryColumns(T), content: T.content };
}

function FullColumns(T: DocumentTable) {
  return { ...TierColumns(T), embedding: T.embedding };
}

function ToRow(Doc: DocumentRecord): DocumentInsert {
  return {
    id: Doc.Id,
    tier: Doc.Tier,
    origin: Doc.Origin,
    source: Doc.Source,
    license: Doc.License,
    lang: Doc.Lang,
    content: Doc.Content,
    bytes: Doc.Bytes,
    quality: Doc.QualityScore,
    contentHash: Doc.ContentHash,
    embedding: Doc.Embedding,
    rejectReason: Doc.RejectReason,
    provenance: Doc.Provenance,
    ingestedAt: Doc.IngestedAt,
  };
}

function FromSummaryRow(Row: SummaryRow): DocumentSummary {
  return {
    Id: Row.id,
    Tier: Row.tier as Tier,
    Origin: Row.origin as DocumentRecord["Origin"],
    Source: Row.source,
    License: Row.license,
    Lang: Row.lang,
    Bytes: Row.bytes,
    QualityScore: Row.quality,
    ContentHash: Row.contentHash,
    RejectReason: Row.rejectReason,
    Provenance: Row.provenance,
    IngestedAt: Row.ingestedAt,
  };
}

function FromTierRow(Row: TierRow): TierDocument {
  return { ...FromSummaryRow(Row), Content: Row.content };
}

function FromRow(Row: DocumentRow): DocumentRecord {
  return { ...FromTierRow(Row), Embedding: Row.embedding };
}

/** Escape LIKE/ILIKE metacharacters so a user's search is matched literally (used with ESCAPE '\'). */
function EscapeLikePattern(Text: string): string {
  return Text.replace(/[\\%_]/g, (Char) => `\\${Char}`);
}

export class PostgresDocumentStore implements DocumentStore {
  private Sql: ReturnType<typeof postgres>;
  private Db: ReturnType<typeof drizzle>;
  private Table: DocumentTable;
  private OwnsSql: boolean;

  // Accepts a connection URL (opens+owns its own client) OR a shared postgres client (so many per-kind
  // stores can share ONE connection pool instead of opening one each — see FoundryStores).
  constructor(UrlOrSql: string | ReturnType<typeof postgres>, TableName = LegacyDocumentsTableName) {
    this.OwnsSql = typeof UrlOrSql === "string";
    this.Sql = typeof UrlOrSql === "string" ? postgres(UrlOrSql, { onnotice: () => {} }) : UrlOrSql;
    this.Db = drizzle(this.Sql);
    this.Table = DocumentsTable(TableName);
  }

  // Byte-order id comparison/ordering: locale-independent, identical to the in-memory store, and
  // served by the (tier, id COLLATE "C") index.
  private IdOrder(): SQL {
    return sql`${this.Table.id} collate "C"`;
  }

  async Upsert(Doc: DocumentRecord): Promise<boolean> {
    // First-seen wins: a re-collected document (same content-hash id) is NOT rewritten — its original
    // provenance/source/ingested_at stay. RETURNING yields a row only for a real insert, which is
    // exactly the New-vs-Duplicate signal ingestion reports. Reclassification goes through Retier.
    const Returned = await this.Db
      .insert(this.Table)
      .values(ToRow(Doc))
      .onConflictDoNothing({ target: this.Table.id })
      .returning({ Id: this.Table.id });
    return Returned.length > 0;
  }

  async All(): Promise<DocumentRecord[]> {
    return (await this.Db.select(FullColumns(this.Table)).from(this.Table)).map(FromRow);
  }

  async ByTier(Tier: Tier, Limit?: number): Promise<TierDocument[]> {
    // ORDER BY id makes a capped training read deterministic (the same Limit always returns the same
    // documents), which is what lets a resumed run see the same corpus.
    const Query = this.Db.select(TierColumns(this.Table)).from(this.Table).where(eq(this.Table.tier, Tier)).orderBy(this.IdOrder());
    const Rows = Limit !== undefined ? await Query.limit(Math.max(0, Math.floor(Limit))) : await Query;
    return Rows.map(FromTierRow);
  }

  async FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]> {
    // In-app cosine ranking (portable: no pgvector required). Rank over (id, embedding) only, then
    // load full records for the winners — never materializes every document's content.
    const Vectors = await this.Db.select({ Id: this.Table.id, Embedding: this.Table.embedding }).from(this.Table);
    const Top = Vectors.map((V) => ({ Id: V.Id, Score: CosineSimilarity(Embedding, V.Embedding) }))
      .sort((A, B) => B.Score - A.Score)
      .slice(0, Math.max(0, Limit));
    if (Top.length === 0) return [];
    const Rows = await this.Db.select(FullColumns(this.Table)).from(this.Table).where(inArray(this.Table.id, Top.map((T) => T.Id)));
    const ById = new Map(Rows.map((R) => [R.id, FromRow(R)]));
    const Hits: SimilarHit[] = [];
    for (const T of Top) {
      const Doc = ById.get(T.Id);
      if (Doc !== undefined) Hits.push({ Doc, Score: T.Score }); // deleted between the two reads → skip
    }
    return Hits;
  }

  async Count(): Promise<number> {
    const Result = await this.Db.select({ Count: sql<number>`count(*)::int` }).from(this.Table);
    return Number(Result[0]?.Count ?? 0);
  }

  async Sources(): Promise<string[]> {
    return (await this.Db.selectDistinct({ Source: this.Table.source }).from(this.Table)).map((R) => R.Source);
  }

  async RepoSummaries(Filter: DocumentFilter = {}): Promise<RepoSummary[]> {
    const Cond = this.WhereFor(Filter);
    const Select = this.Db
      .select({ Source: this.Table.source, Files: sql<number>`count(*)::int`, Bytes: sql<number>`coalesce(sum(${this.Table.bytes}),0)::bigint` })
      .from(this.Table);
    const Rows = await (Cond ? Select.where(Cond) : Select).groupBy(this.Table.source).orderBy(desc(sql`count(*)`));
    return Rows.map((R) => ({ Source: R.Source, Files: Number(R.Files), Bytes: Number(R.Bytes) }));
  }

  async DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]> {
    const Rows = await this.Db.select(FullColumns(this.Table)).from(this.Table).where(eq(this.Table.source, Source)).orderBy(this.IdOrder()).limit(ClampPageLimit(Limit));
    return Rows.map(FromRow);
  }

  async DocumentById(Id: string): Promise<DocumentRecord | null> {
    const Rows = await this.Db.select(FullColumns(this.Table)).from(this.Table).where(eq(this.Table.id, Id)).limit(1);
    return Rows[0] ? FromRow(Rows[0]) : null;
  }

  async ContentByIds(Ids: readonly string[]): Promise<Map<string, string>> {
    const Wanted = Ids.slice(0, MaxPageLimit);
    if (Wanted.length === 0) return new Map();
    const Rows = await this.Db.select({ Id: this.Table.id, Content: this.Table.content }).from(this.Table).where(inArray(this.Table.id, Wanted));
    return new Map(Rows.map((R) => [R.Id, R.Content]));
  }

  // THE single WHERE builder for every filtered read/delete/count, so a page's rows and its Total can
  // never disagree. Search is a literal case-insensitive substring match on provenance OR content:
  // %, _ and \ in the user's text are escaped and matched with ESCAPE '\' (served by the pg_trgm GIN
  // indexes when the extension is installed; a seq scan otherwise — same results either way).
  private WhereFor(Filter: DocumentFilter): SQL | undefined {
    const Conds: SQL[] = [];
    if (Filter.Tier) Conds.push(eq(this.Table.tier, Filter.Tier));
    if (Filter.Lang) Conds.push(eq(this.Table.lang, Filter.Lang));
    if (Filter.License) Conds.push(eq(this.Table.license, Filter.License));
    if (Filter.Origin) Conds.push(eq(this.Table.origin, Filter.Origin));
    if (Filter.Source) Conds.push(eq(this.Table.source, Filter.Source));
    const Search = NormalizeSearch(Filter.Search);
    if (Search !== null) {
      const Like = `%${EscapeLikePattern(Search)}%`;
      const Match = or(sql`${this.Table.provenance} ilike ${Like} escape '\\'`, sql`${this.Table.content} ilike ${Like} escape '\\'`);
      if (Match) Conds.push(Match);
    }
    return Conds.length === 0 ? undefined : and(...Conds);
  }

  private async CountWhere(Cond: SQL | undefined): Promise<number> {
    const Counting = this.Db.select({ Count: sql<number>`count(*)::int` }).from(this.Table);
    const CountRows = await (Cond ? Counting.where(Cond) : Counting);
    return Number(CountRows[0]?.Count ?? 0);
  }

  async Query(Filter: DocumentFilter, Offset: number, Limit: number): Promise<DocumentPage> {
    const Cond = this.WhereFor(Filter);
    const Select = this.Db.select(FullColumns(this.Table)).from(this.Table);
    // Tie-break on the unique id: a whole collection run shares ONE ingestedAt, so ordering by it
    // alone gives Postgres no stable order across paged OFFSET/LIMIT queries — pages could repeat or
    // skip rows. The id (primary key) makes the order total and pagination deterministic.
    const Rows = await (Cond ? Select.where(Cond) : Select)
      .orderBy(desc(this.Table.ingestedAt), desc(this.IdOrder()))
      .offset(ClampOffset(Offset))
      .limit(ClampPageLimit(Limit));
    return { Rows: Rows.map(FromRow), Total: await this.CountWhere(Cond) };
  }

  async ListSummaries(Opts: ListOptions): Promise<SummaryPage> {
    const Cond = this.WhereFor(Opts.Filter ?? {});
    const Limit = ClampPageLimit(Opts.Limit);
    const Select = this.Db.select(SummaryColumns(this.Table)).from(this.Table);
    let Rows: SummaryRow[];
    if (Opts.AfterId !== undefined) {
      // Keyset paging: ascending id strictly after the cursor — stable while rows change tier.
      const After = sql`${this.IdOrder()} > ${Opts.AfterId}`;
      Rows = await Select.where(Cond ? and(Cond, After) : After).orderBy(this.IdOrder()).limit(Limit);
    } else {
      Rows = await (Cond ? Select.where(Cond) : Select)
        .orderBy(desc(this.Table.ingestedAt), desc(this.IdOrder()))
        .offset(ClampOffset(Opts.Offset))
        .limit(Limit);
    }
    return { Rows: Rows.map(FromSummaryRow), Total: await this.CountWhere(Cond) };
  }

  async DeleteById(Id: string): Promise<number> {
    const Deleted = await this.Db.delete(this.Table).where(eq(this.Table.id, Id)).returning({ Id: this.Table.id });
    return Deleted.length;
  }

  async DeleteMatching(Filter: DocumentFilter): Promise<number> {
    // One statement; the affected-row count comes from the DELETE itself (no count-then-delete race,
    // and no RETURNING that would materialize every id when a filter matches 100k+ rows).
    const Cond = this.WhereFor(Filter);
    const Del = this.Db.delete(this.Table);
    const Result = await (Cond ? Del.where(Cond) : Del);
    return Number(Result.count);
  }

  async Retier(Id: string, Tier: Tier, RejectReason: string | null, QualityScore?: number): Promise<boolean> {
    const Set = QualityScore !== undefined ? { tier: Tier, rejectReason: RejectReason, quality: QualityScore } : { tier: Tier, rejectReason: RejectReason };
    const Updated = await this.Db.update(this.Table).set(Set).where(eq(this.Table.id, Id)).returning({ Id: this.Table.id });
    return Updated.length > 0;
  }

  async ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }> {
    // Scoped to license='NOASSERTION' so only the previously-unresolved rows are relabeled. The id
    // (which encodes the old license) is intentionally left as-is — it is a dedup key, and SkipRepo
    // by source prevents any future re-ingest that would collide with it. Both UPDATEs run in ONE
    // transaction: a crash between them can no longer leave half a repo relabeled.
    const Unresolved = and(eq(this.Table.source, Source), eq(this.Table.license, "NOASSERTION"));
    return this.Db.transaction(async (Tx) => {
      const Promote = await Tx.update(this.Table)
        .set({ tier: "Filtered", license: NewLicense, rejectReason: null })
        .where(and(Unresolved, sql`${this.Table.quality} >= ${MinQuality}`))
        .returning({ Id: this.Table.id });
      const Keep = await Tx.update(this.Table)
        .set({ license: NewLicense, rejectReason: `low quality (score < ${MinQuality})` })
        .where(and(Unresolved, sql`${this.Table.quality} < ${MinQuality}`))
        .returning({ Id: this.Table.id });
      return { Promoted: Promote.length, KeptLowQuality: Keep.length };
    });
  }

  async Stats(): Promise<FoundryStats> {
    const CountExpr = sql<number>`count(*)::int`;
    const Tiers = await this.Db.select({ Key: this.Table.tier, Count: CountExpr }).from(this.Table).groupBy(this.Table.tier);
    const Langs = await this.Db.select({ Key: this.Table.lang, Count: CountExpr }).from(this.Table).groupBy(this.Table.lang);
    const Licenses = await this.Db.select({ Key: this.Table.license, Count: CountExpr }).from(this.Table).groupBy(this.Table.license);
    const Filtered = await this.Db.select({ Bytes: sql<number>`coalesce(sum(${this.Table.bytes}),0)::bigint` }).from(this.Table).where(eq(this.Table.tier, "Filtered"));

    const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
    for (const Row of Tiers) if (Row.Key in ByTier) ByTier[Row.Key as Tier] = Number(Row.Count);
    const ByLang: Record<string, number> = {};
    for (const Row of Langs) ByLang[Row.Key] = Number(Row.Count);
    const ByLicense: Record<string, number> = {};
    for (const Row of Licenses) ByLicense[Row.Key] = Number(Row.Count);
    const Total = ByTier.Filtered + ByTier.Raw + ByTier.Rejected;
    return { Total, ByTier, ByLang, ByLicense, FilteredBytes: Number(Filtered[0]?.Bytes ?? 0) };
  }

  async Close(): Promise<void> {
    if (this.OwnsSql) await this.Sql.end(); // don't close a shared client we didn't open
  }
}
