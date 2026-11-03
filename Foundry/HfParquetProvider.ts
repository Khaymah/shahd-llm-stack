// Generic Hugging Face parquet provider (data engine, Phase 2) — the reusable engine behind the bulk
// datasets that only ship as parquet (Wikipedia dumps now; Gutenberg / Stack Exchange later). The HF
// datasets-server /rows JSON API is unreachable from here, but the parquet FILES download fine via
// resolve/main, and hyparquet + hyparquet-compressors decode them (verified: codec is not a blocker).
//
// A dataset's shards are discovered from the HF tree API, then read via HTTP RANGE requests: hyparquet's
// asyncBufferFromUrl fetches only the parquet footer plus the row groups a window actually needs, so a
// run downloads a few MB — not the whole 157 MB shard (verified: 5 rows decoded from 2.8 MB in 3s). This
// also sidesteps the xethub CDN dropping long transfers (small requests rarely ECONNRESET). Rows are read
// in windows so memory stays bounded; a {Shard, Offset} cursor (persisted in collection_state) resumes
// across shards — the first real use of that Phase-1 cursor. open/decode/list are injected so the
// provider is testable with plain data (no network, no real parquet).

import { parquetReadObjects, asyncBufferFromUrl, cachedAsyncBuffer } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { DataKind } from "./DataKinds.ts";
import { FetchWithBackoff, HttpError } from "./HttpBackoff.ts";

export type ParquetRow = Record<string, unknown>;

// What hyparquet reads THROUGH: a total size plus a slice(start,end) that fetches only that byte range
// (an HTTP Range request for the URL case). This is how a shard is read without downloading all of it.
export type AsyncBuffer = {
  byteLength: number;
  // `slice` is hyparquet's AsyncBuffer contract (external API), so it cannot be PascalCase.
  // eslint-disable-next-line @typescript-eslint/naming-convention
  slice: (Start: number, End?: number) => Promise<ArrayBuffer> | ArrayBuffer;
};

// One HF parquet dataset the provider can collect. ConfigFor turns the run's Query (e.g. a Wikipedia
// language) into the HF config directory; MapRow turns a decoded row into a document body (or null to
// skip a stub). License/Kind are fixed per source and recorded on every document.
export type HfParquetSource = {
  Name: string;
  Kind: DataKind;
  License: string;
  Dataset: string; // "wikimedia/wikipedia"
  ConfigFor: (Query: string) => string; // lang -> "20231101.simple"
  MapRow: (Row: ParquetRow, Query: string) => { Content: string; Provenance: string; Lang: string } | null;
};

export type HfParquetOptions = {
  StartShard?: number; // resume: shard index to start at (from the persisted cursor)
  StartOffset?: number; // resume: row offset within that shard
  MaxPerRun?: number; // cap rows collected this run
  WindowRows?: number; // rows decoded per hyparquet call (memory vs. call-count trade-off)
  BatchSize?: number;
  ListShards?: (Dataset: string, Config: string) => Promise<string[]>; // injected in tests
  OpenShard?: (Url: string) => Promise<AsyncBuffer>; // injected in tests; range-backed reader by default
  ReadRows?: (File: AsyncBuffer, Start: number, End: number) => Promise<ParquetRow[]>; // injected in tests
  // Report the COMMITTED cursor (every row before it is stored) after each flushed batch, so the caller
  // can persist it write-through; awaited, so a Stop/crash right after a batch never loses its cursor.
  OnCursor?: (Shard: number, Offset: number) => void | Promise<void>;
  Sleep?: (Ms: number) => Promise<void>;
  OnRepoStart?: (Name: string) => void;
  OnRepoReady?: BatchSink;
  Log?: (Message: string) => void;
};

// Batch sink: stores one flushed batch. Resolving `false` (or throwing) means the batch was NOT fully
// stored — the cursor then stays at that batch's start so the next run re-reads it (content-hash dedup
// makes the re-read idempotent). A plain RepoSink (resolves void) counts as success.
export type BatchSink = (Source: string, Files: SourceInput[]) => Promise<boolean | void>;

const HfBase = "https://huggingface.co";

// Discover a dataset config's parquet shards (ordered) via the HF tree API — reachable even though the
// datasets-server is not. Returns full resolve URLs.
async function DefaultListShards(Dataset: string, Config: string, Sleep?: (Ms: number) => Promise<void>): Promise<string[]> {
  const Tree = (await FetchWithBackoff(
    async () => {
      const Response = await fetch(`${HfBase}/api/datasets/${Dataset}/tree/main/${Config}`, { headers: { "User-Agent": "shahd-foundry" } });
      if (!Response.ok) throw new HttpError(Response.status);
      return Response.json() as Promise<{ path?: string }[]>;
    },
    { Sleep },
  )) as { path?: string }[];
  return Tree.filter((E) => typeof E.path === "string" && E.path.endsWith(".parquet"))
    .map((E) => E.path!)
    .sort()
    .map((Path) => `${HfBase}/datasets/${Dataset}/resolve/main/${Path}`);
}

// Open a shard for RANGE reads: asyncBufferFromUrl does a HEAD for the size, then each slice() is an HTTP
// Range request (only the bytes hyparquet asks for). cachedAsyncBuffer memoizes fetched ranges so reading
// several windows out of the same row group doesn't re-download it. The whole 157 MB is never pulled.
async function DefaultOpenShard(Url: string): Promise<AsyncBuffer> {
  const Buffer = await asyncBufferFromUrl({ url: Url, requestInit: { headers: { "User-Agent": "shahd-foundry" } } });
  return cachedAsyncBuffer(Buffer) as AsyncBuffer;
}

async function DefaultReadRows(File: AsyncBuffer, Start: number, End: number): Promise<ParquetRow[]> {
  return parquetReadObjects({ file: File, compressors, rowStart: Start, rowEnd: End });
}

export function CreateHfParquetProvider(Source: HfParquetSource, Options: HfParquetOptions = {}): WebProvider {
  const MaxPerRun = Options.MaxPerRun ?? 200_000;
  const Window = Options.WindowRows ?? 1000;
  const BatchSize = Options.BatchSize ?? 500;
  const ListShards = Options.ListShards ?? ((D: string, C: string): Promise<string[]> => DefaultListShards(D, C, Options.Sleep));
  const OpenShard = Options.OpenShard ?? DefaultOpenShard;
  const ReadRows = Options.ReadRows ?? DefaultReadRows;
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));

  return {
    Name: Source.Name,
    Semantics: "streaming", // more shards / languages remain to collect; cursor advances across runs
    Fetch: async (Query: string, Limit: number, Signal?: AbortSignal): Promise<SourceInput[]> => {
      const Config = Source.ConfigFor(Query);
      Options.OnRepoStart?.(`${Source.Name} ${Config} shards`);
      const Shards = await ListShards(Source.Dataset, Config);
      Log(`[${Source.Name}] ${Config}: ${Shards.length} shard(s)`);

      const Shard = Options.StartShard ?? 0;
      let Offset = Options.StartOffset ?? 0;
      const Cap = Math.min(Math.max(1, Limit), MaxPerRun);

      if (Shard >= Shards.length) {
        Log(`[${Source.Name}] all ${Shards.length} shard(s) already collected (cursor past end)`);
        await Options.OnCursor?.(Shard, 0);
        return [];
      }

      Options.OnRepoStart?.(`${Source.Name} shard ${Shard + 1}/${Shards.length}`);
      Log(`[${Source.Name}] opening shard ${Shard + 1}/${Shards.length} (range reads)…`);
      const File = await OpenShard(Shards[Shard]!);

      // Committed = the row offset before which EVERY row is stored (or was a skipped stub). It only
      // moves past a batch that fully succeeded; once any batch fails it freezes (Frozen) for the rest
      // of the run, because a single {Shard, Offset} cursor cannot represent a hole — later batches are
      // still stored, and the next run re-reads from the hole (dedup makes that cheap and idempotent).
      let Committed = Offset;
      let Frozen = false;
      let Batch: SourceInput[] = [];
      const Unsunk: SourceInput[] = []; // no sink (CLI batch mode): collect and return everything
      let Collected = 0;
      const Stopped = (): boolean => Signal?.aborted === true;
      // Store the pending batch, whose rows end just before EndOffset, then report the committed cursor.
      const Flush = async (EndOffset: number): Promise<void> => {
        if (Batch.length === 0) {
          if (!Frozen) Committed = EndOffset; // only stubs since the last batch: nothing to lose
          return;
        }
        if (Options.OnRepoReady === undefined) {
          Unsunk.push(...Batch);
          Batch = [];
          if (!Frozen) Committed = EndOffset;
          return;
        }
        Options.OnRepoStart?.(`${Source.Name} ${Config} (${Collected}/${Cap})`); // Stop boundary between batches
        const Pending = Batch;
        Batch = [];
        let Stored = false;
        try {
          Stored = (await Options.OnRepoReady(`${Source.Name}-${Config}`, Pending)) !== false;
        } finally {
          // Success or not, persist where we are BEFORE anything propagates (abort/crash-safe).
          if (Stored && !Frozen) Committed = EndOffset;
          else if (!Stored) Frozen = true;
          await Options.OnCursor?.(Shard, Committed);
        }
        if (!Stored) Log(`[${Source.Name}] a batch was not fully stored — cursor held at shard ${Shard} offset ${Committed}`);
      };

      let ShardDone = false;
      while (Collected < Cap && !ShardDone && !Stopped()) {
        const Rows = await ReadRows(File, Offset, Offset + Window);
        if (Rows.length === 0) {
          ShardDone = true;
          break;
        }
        // Advance the offset by rows actually CONSUMED, not by the window size: when the MaxPerRun cap is
        // hit mid-window, the unconsumed rows must be re-read next run, not skipped.
        let Consumed = 0;
        for (const Row of Rows) {
          Consumed++;
          const Mapped = Source.MapRow(Row, Query);
          if (Mapped !== null) {
            Batch.push({ Source: Source.Name, License: Source.License, Lang: Mapped.Lang, Content: Mapped.Content, Provenance: Mapped.Provenance, Origin: "curated" });
            Collected++;
            if (Batch.length >= BatchSize) {
              if (Stopped()) break; // Stop: drop the unstored batch; the cursor stays before it
              await Flush(Offset + Consumed);
            }
          }
          if (Collected >= Cap) break;
        }
        Offset += Consumed;
        if (Rows.length < Window) ShardDone = true; // last (partial) window of the shard
      }

      if (Stopped()) {
        // A stop request: the pending (unstored) batch is dropped and the committed cursor is reported
        // so the next run resumes exactly after the last stored batch.
        await Options.OnCursor?.(Shard, Committed);
        Log(`[${Source.Name}] stopped; cursor -> shard ${Shard} offset ${Committed}`);
        return [];
      }
      await Flush(Offset);

      // Advance the cursor: to the next shard if this one is finished AND fully stored, else stay on it
      // at the committed offset (a mid-shard cap will re-download this shard next run).
      const NextShard = ShardDone && !Frozen ? Shard + 1 : Shard;
      const NextOffset = ShardDone && !Frozen ? 0 : Committed;
      await Options.OnCursor?.(NextShard, NextOffset);
      Log(`[${Source.Name}] collected ${Collected} docs; cursor -> shard ${NextShard} offset ${NextOffset}`);
      return Unsunk;
    },
  };
}

// Stack Exchange Q&A (CC-BY-SA) as conversation data — grows the dialogue corpus beyond OASST with real
// question/answer pairs across every SE site. donfu/oa-stackexchange is already cleaned + paired
// (columns INSTRUCTION / RESPONSE / SOURCE), parquet under data/. Query is ignored (one corpus); each
// row becomes a User/Assistant turn. The provider's cursor collects it shard by shard across runs.
export const StackExchangeSource: HfParquetSource = {
  Name: "stackexchange",
  Kind: "conversation",
  License: "CC-BY-SA-4.0",
  Dataset: "donfu/oa-stackexchange",
  ConfigFor: (): string => "data", // the parquet lives under data/ ; there is no per-query config
  MapRow: (Row: ParquetRow): { Content: string; Provenance: string; Lang: string } | null => {
    const Q = typeof Row["INSTRUCTION"] === "string" ? (Row["INSTRUCTION"] as string).trim() : "";
    const A = typeof Row["RESPONSE"] === "string" ? (Row["RESPONSE"] as string).trim() : "";
    if (Q.length < 10 || A.length < 20) return null; // skip empty / trivial pairs
    const Site = typeof Row["SOURCE"] === "string" ? (Row["SOURCE"] as string) : "stackexchange";
    return { Content: `User: ${Q}\n\nAssistant: ${A}`, Provenance: `stackexchange:${Site}`, Lang: "text-en" };
  },
};

// The first parquet source: Wikipedia dumps (CC-BY-SA), bulk knowledge to replace the live API's trickle.
// Query is the language ("simple" for Simple English by default — the smallest config, one shard).
export const WikiDumpSource: HfParquetSource = {
  Name: "wikidump",
  Kind: "knowledge",
  License: "CC-BY-SA-4.0",
  Dataset: "wikimedia/wikipedia",
  ConfigFor: (Query: string): string => `20231101.${(Query || "simple").trim().toLowerCase()}`,
  MapRow: (Row: ParquetRow, Query: string): { Content: string; Provenance: string; Lang: string } | null => {
    const Lang = (Query || "simple").trim().toLowerCase();
    const Title = typeof Row["title"] === "string" ? (Row["title"] as string) : "";
    const Text = typeof Row["text"] === "string" ? (Row["text"] as string).trim() : "";
    if (Text.length < 200) return null; // skip stubs / near-empty extracts
    return { Content: `${Title}\n\n${Text}`.trim(), Provenance: `wikipedia:${Lang}:${Row["id"] ?? "?"}`, Lang: `text-${Lang}` };
  },
};
