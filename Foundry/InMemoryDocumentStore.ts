// In-memory DocumentStore (M3) — the dependency-free implementation used by tests and small local
// runs. Same interface AND the same semantics as the Postgres store (insert-if-absent upsert, ascending
// id ByTier, literal case-insensitive substring search, clamped pages), so ingestion/report/export code
// validated here behaves identically against a live database — Tests/DocumentStoreContract pins that.
// FindSimilar is a linear cosine scan (fine at this scale).

import { ClampOffset, ClampPageLimit, MaxPageLimit, NormalizeSearch } from "./DocumentStore.ts";
import type { DocumentStore, SimilarHit, RepoSummary, FoundryStats, DocumentFilter, DocumentPage, DocumentSummary, ListOptions, SummaryPage, TierDocument } from "./DocumentStore.ts";
import type { DocumentRecord, Tier } from "./DocumentRecord.ts";
import { CosineSimilarity } from "./Embedding.ts";

// Same total order as Postgres `ORDER BY id` for the hex ids we store (plain code-unit comparison,
// not locale-aware localeCompare, which could disagree with the database collation on edge cases).
function ById(A: { Id: string }, B: { Id: string }): number {
  return A.Id < B.Id ? -1 : A.Id > B.Id ? 1 : 0;
}

// Newest-first by ingestion, id DESC tie-break — the Postgres browse order.
function NewestFirst(A: DocumentRecord, B: DocumentRecord): number {
  if (A.IngestedAt !== B.IngestedAt) return A.IngestedAt < B.IngestedAt ? 1 : -1;
  return -ById(A, B);
}

function ToSummary(Doc: DocumentRecord): DocumentSummary {
  const { Content: _Content, Embedding: _Embedding, ...Summary } = Doc;
  return { ...Summary };
}

function ToTierDocument(Doc: DocumentRecord): TierDocument {
  const { Embedding: _Embedding, ...Rest } = Doc;
  return { ...Rest };
}

export class InMemoryDocumentStore implements DocumentStore {
  private Docs = new Map<string, DocumentRecord>();

  async Upsert(Doc: DocumentRecord): Promise<boolean> {
    // Dedup by content-hash Id. First-seen wins: a re-ingest of the same bytes leaves the stored row
    // (and its provenance) untouched, exactly like Postgres ON CONFLICT DO NOTHING.
    if (this.Docs.has(Doc.Id)) return false;
    this.Docs.set(Doc.Id, { ...Doc });
    return true;
  }

  async All(): Promise<DocumentRecord[]> {
    return [...this.Docs.values()];
  }

  async ByTier(Tier: Tier, Limit?: number): Promise<TierDocument[]> {
    const Matching = [...this.Docs.values()].filter((D) => D.Tier === Tier).sort(ById);
    const Capped = Limit !== undefined ? Matching.slice(0, Math.max(0, Limit)) : Matching;
    return Capped.map(ToTierDocument);
  }

  async FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]> {
    return [...this.Docs.values()]
      .map((Doc) => ({ Doc, Score: CosineSimilarity(Embedding, Doc.Embedding) }))
      .sort((A, B) => B.Score - A.Score)
      .slice(0, Limit);
  }

  async Count(): Promise<number> {
    return this.Docs.size;
  }

  async Sources(): Promise<string[]> {
    return [...new Set([...this.Docs.values()].map((D) => D.Source))];
  }

  async RepoSummaries(Filter: DocumentFilter = {}): Promise<RepoSummary[]> {
    const By = new Map<string, RepoSummary>();
    for (const Doc of this.Docs.values()) {
      if (!this.Matches(Doc, Filter)) continue;
      const Entry = By.get(Doc.Source) ?? { Source: Doc.Source, Files: 0, Bytes: 0 };
      Entry.Files++;
      Entry.Bytes += Doc.Bytes;
      By.set(Doc.Source, Entry);
    }
    return [...By.values()].sort((A, B) => B.Files - A.Files);
  }

  async DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]> {
    return [...this.Docs.values()].filter((D) => D.Source === Source).slice(0, ClampPageLimit(Limit));
  }

  async DocumentById(Id: string): Promise<DocumentRecord | null> {
    return this.Docs.get(Id) ?? null;
  }

  async ExistingContentHashes(Hashes: readonly string[], Tier?: Tier): Promise<Set<string>> {
    const Wanted = new Set(Hashes);
    const Found = new Set<string>();
    if (Wanted.size === 0) return Found;
    for (const Doc of this.Docs.values()) {
      if (Wanted.has(Doc.ContentHash) && (Tier === undefined || Doc.Tier === Tier)) Found.add(Doc.ContentHash);
    }
    return Found;
  }

  async ContentByIds(Ids: readonly string[]): Promise<Map<string, string>> {
    const Out = new Map<string, string>();
    for (const Id of Ids.slice(0, MaxPageLimit)) {
      const Doc = this.Docs.get(Id);
      if (Doc !== undefined) Out.set(Id, Doc.Content);
    }
    return Out;
  }

  // Mirrors PostgresDocumentStore.WhereFor: exact match on the structured fields, and a literal,
  // case-insensitive substring match (ILIKE with %/_/\ escaped) on provenance OR content.
  private Matches(Doc: DocumentRecord, Filter: DocumentFilter): boolean {
    if (Filter.Tier && Doc.Tier !== Filter.Tier) return false;
    if (Filter.Lang && Doc.Lang !== Filter.Lang) return false;
    if (Filter.License && Doc.License !== Filter.License) return false;
    if (Filter.Origin && Doc.Origin !== Filter.Origin) return false;
    if (Filter.Source && Doc.Source !== Filter.Source) return false;
    const Search = NormalizeSearch(Filter.Search)?.toLowerCase();
    if (Search !== undefined) return Doc.Provenance.toLowerCase().includes(Search) || Doc.Content.toLowerCase().includes(Search);
    return true;
  }

  async Query(Filter: DocumentFilter, Offset: number, Limit: number): Promise<DocumentPage> {
    const Matching = [...this.Docs.values()].filter((D) => this.Matches(D, Filter)).sort(NewestFirst);
    const Start = ClampOffset(Offset);
    return { Rows: Matching.slice(Start, Start + ClampPageLimit(Limit)), Total: Matching.length };
  }

  async ListSummaries(Opts: ListOptions): Promise<SummaryPage> {
    const Filter = Opts.Filter ?? {};
    const Matching = [...this.Docs.values()].filter((D) => this.Matches(D, Filter));
    const Limit = ClampPageLimit(Opts.Limit);
    if (Opts.AfterId !== undefined) {
      const After = Opts.AfterId;
      const Page = Matching.filter((D) => D.Id > After).sort(ById).slice(0, Limit);
      return { Rows: Page.map(ToSummary), Total: Matching.length };
    }
    const Start = ClampOffset(Opts.Offset);
    return { Rows: Matching.sort(NewestFirst).slice(Start, Start + Limit).map(ToSummary), Total: Matching.length };
  }

  async DeleteById(Id: string): Promise<number> {
    return this.Docs.delete(Id) ? 1 : 0;
  }

  async DeleteMatching(Filter: DocumentFilter): Promise<number> {
    let N = 0;
    for (const [Id, Doc] of [...this.Docs.entries()]) {
      if (this.Matches(Doc, Filter)) {
        this.Docs.delete(Id);
        N++;
      }
    }
    return N;
  }

  async Retier(Id: string, Tier: Tier, RejectReason: string | null, QualityScore?: number): Promise<boolean> {
    const Doc = this.Docs.get(Id);
    if (Doc === undefined) return false;
    Doc.Tier = Tier;
    Doc.RejectReason = RejectReason;
    if (QualityScore !== undefined) Doc.QualityScore = QualityScore;
    return true;
  }

  async ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }> {
    let Promoted = 0;
    let KeptLowQuality = 0;
    for (const Doc of this.Docs.values()) {
      if (Doc.Source !== Source || Doc.License !== "NOASSERTION") continue;
      Doc.License = NewLicense;
      if (Doc.QualityScore >= MinQuality) {
        Doc.Tier = "Filtered";
        Doc.RejectReason = null;
        Promoted++;
      } else {
        Doc.RejectReason = `low quality (score < ${MinQuality})`;
        KeptLowQuality++;
      }
    }
    return { Promoted, KeptLowQuality };
  }

  async Stats(): Promise<FoundryStats> {
    const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
    const ByLang: Record<string, number> = {};
    const ByLicense: Record<string, number> = {};
    let FilteredBytes = 0;
    for (const Doc of this.Docs.values()) {
      ByTier[Doc.Tier]++;
      ByLang[Doc.Lang] = (ByLang[Doc.Lang] ?? 0) + 1;
      ByLicense[Doc.License] = (ByLicense[Doc.License] ?? 0) + 1;
      if (Doc.Tier === "Filtered") FilteredBytes += Doc.Bytes;
    }
    return { Total: this.Docs.size, ByTier, ByLang, ByLicense, FilteredBytes };
  }
}
