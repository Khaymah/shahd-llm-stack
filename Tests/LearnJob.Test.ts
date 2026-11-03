import { test, expect } from "bun:test";
import { LearnJob } from "../Foundry/LearnJob.ts";
import type { LearnDoneEvent } from "../Foundry/LearnJob.ts";
import { InMemoryDocumentStore } from "../Foundry/InMemoryDocumentStore.ts";
import { InMemoryCollectionStateStore } from "../Foundry/InMemoryCollectionStateStore.ts";
import { CreateHfParquetProvider, WikiDumpSource } from "../Foundry/HfParquetProvider.ts";
import { MaxConsecutiveIngestFailures } from "../Foundry/Ingest.ts";
import type { CollectionState } from "../Foundry/CollectionState.ts";
import type { DocumentRecord } from "../Foundry/DocumentRecord.ts";
import type { SourceInput } from "../Foundry/Ingest.ts";
import type { LearnEvent, LearnSettings } from "../Foundry/DashboardTypes.ts";
import type { ProviderFactory, ProviderHooks } from "../Foundry/KindRouting.ts";
import type { WebProvider } from "../Foundry/WebSource.ts";
import type { ParquetRow } from "../Foundry/HfParquetProvider.ts";

const Quiet = { Log: (): void => {}, Warn: (): void => {}, Now: (): string => "2026-11-02T00:00:00.000Z" };

const Settings = (Over: Partial<LearnSettings> = {}): LearnSettings => ({
  Source: "oasst", Query: "en", Repos: [], MinLevel: "low", MaxRepos: 1000, MaxFilesPerRepo: 100, MaxBytesPerRepo: 1_000_000, MaxContentBytes: 100_000, SkipLearned: false, ...Over,
});

const Doc = (I: number): SourceInput => ({ Source: "src", License: "MIT", Lang: "text-en", Content: `User: question ${I}?\n\nAssistant: a reasonably long answer number ${I}.`, Provenance: `doc:${I}`, Origin: "curated" });
const Docs = (From: number, Count: number): SourceInput[] => Array.from({ length: Count }, (_, I) => Doc(From + I));

// A store that fails upserts whose 0-based call index matches the predicate.
class FailingStore extends InMemoryDocumentStore {
  Calls = 0;
  private FailWhen: (Call: number) => boolean;
  constructor(FailWhen: (Call: number) => boolean) {
    super();
    this.FailWhen = FailWhen;
  }
  override async Upsert(Record: DocumentRecord): Promise<boolean> {
    const Call = this.Calls++;
    if (this.FailWhen(Call)) throw new Error(`store rejected row ${Call}`);
    return super.Upsert(Record);
  }
}

// A ledger that records every write so write-through can be asserted.
class SpyLedger extends InMemoryCollectionStateStore {
  Writes: CollectionState[] = [];
  override async Upsert(State: CollectionState): Promise<void> {
    this.Writes.push({ ...State });
    await super.Upsert(State);
  }
}

// A fake provider that streams the given batches into the run's sink, honoring the Stop boundary.
const BatchProvider = (Name: string, Batches: SourceInput[][], Semantics: "bounded" | "streaming" = "bounded") =>
  (Hooks: ProviderHooks): WebProvider => ({
    Name,
    Semantics,
    Fetch: async (): Promise<SourceInput[]> => {
      for (const [I, Batch] of Batches.entries()) {
        Hooks.OnRepoStart(`${Name} batch ${I}`);
        await Hooks.OnRepoReady(`${Name}-${I}`, Batch);
      }
      return [];
    },
  });

function Collect(): { Events: LearnEvent[]; OnEvent: (E: LearnEvent) => void; Dones: () => LearnDoneEvent[] } {
  const Events: LearnEvent[] = [];
  return { Events, OnEvent: (E) => Events.push(E), Dones: () => Events.filter((E) => E.kind === "done") as LearnDoneEvent[] };
}

test("(a) per-document failures surface in the terminal done event", async () => {
  const Store = new FailingStore((Call) => Call === 1);
  const Job = new LearnJob({ ...Quiet, KindStore: () => Store, CollectionState: new InMemoryCollectionStateStore(), CreateProviders: (_S, H) => [BatchProvider("p", [Docs(0, 5)])(H)] });
  const Sink = Collect();
  await Job.Start(Settings(), Sink.OnEvent);
  const [Done] = Sink.Dones();
  expect(Sink.Dones().length).toBe(1);
  expect(Done!.failedDocs).toBe(1);
  expect(Done!.ingested).toBe(4);
  expect(Done!.failed).toBe(false); // under the budget: a degraded run, not an aborted one
});

test("(b) the run aborts after 10 consecutive store failures and says so in done", async () => {
  const Store = new FailingStore(() => true);
  let SecondProviderRan = false;
  const Factory: ProviderFactory = (_S, H) => [
    BatchProvider("first", [Docs(0, 4), Docs(4, 4), Docs(8, 4), Docs(12, 4)])(H),
    { Name: "second", Fetch: async () => { SecondProviderRan = true; return []; } },
  ];
  const Job = new LearnJob({ ...Quiet, KindStore: () => Store, CollectionState: new InMemoryCollectionStateStore(), CreateProviders: Factory });
  const Sink = Collect();
  await Job.Start(Settings(), Sink.OnEvent);
  const Dones = Sink.Dones();
  expect(Dones.length).toBe(1);
  expect(Dones[0]!.failed).toBe(true);
  expect(Dones[0]!.message).toContain("consecutive");
  expect(Dones[0]!.failedDocs).toBe(MaxConsecutiveIngestFailures);
  expect(Store.Calls).toBe(MaxConsecutiveIngestFailures); // stopped at the threshold, no churn
  expect(SecondProviderRan).toBe(false); // a broken store ends the whole run
  expect(Sink.Events.at(-1)!.kind).toBe("done"); // error events are never terminal
  expect(Sink.Events.some((E) => E.kind === "error")).toBe(true);
});

// The real parquet provider over 6 fake rows in 3 batches of 2, wired through the run's hooks.
const ParquetFactory: ProviderFactory = (Run, Hooks) => {
  const Data: ParquetRow[] = Array.from({ length: 6 }, (_, I) => ({ id: I, title: `T${I}`, text: "A long enough article body sentence. ".repeat(10) }));
  return [
    CreateHfParquetProvider(WikiDumpSource, {
      ListShards: async () => ["s0"],
      OpenShard: async () => ({ byteLength: 8, slice: async () => new ArrayBuffer(8) }),
      ReadRows: async (_F, Start, End) => Data.slice(Start, End),
      WindowRows: 2,
      BatchSize: 2,
      MaxPerRun: Run.MaxRepos,
      OnRepoStart: Hooks.OnRepoStart,
      OnRepoReady: Hooks.OnRepoReady,
      OnCursor: Hooks.OnCursor,
      Log: () => {},
    }),
  ];
};

test("(c) the ledger cursor is NOT advanced past a batch that failed to store", async () => {
  const Store = new FailingStore((Call) => Call === 2); // first row of batch 2 fails
  const Ledger = new InMemoryCollectionStateStore();
  const Job = new LearnJob({ ...Quiet, KindStore: () => Store, CollectionState: Ledger, CreateProviders: ParquetFactory });
  const Sink = Collect();
  await Job.Start(Settings({ Source: "wikidump", Query: "simple" }), Sink.OnEvent);
  const State = await Ledger.Get("wikidump:simple");
  expect(JSON.parse(State!.Cursor)).toEqual({ Shard: 0, Offset: 2 }); // held at batch 2's start
  expect(State!.Collected).toBe(5); // the rows that DID store still count
  expect(Sink.Dones()[0]!.failedDocs).toBe(1);
});

test("(d) cursor + lifetime count are persisted after EVERY batch, not only at the end", async () => {
  const Ledger = new SpyLedger();
  await Ledger.Upsert({ SourceKey: "wikidump:simple", Kind: "knowledge", Cursor: "{}", Collected: 100, Exhausted: false, UpdatedAt: "2026-11-01T00:00:00.000Z" });
  Ledger.Writes = [];
  const Job = new LearnJob({ ...Quiet, KindStore: () => new InMemoryDocumentStore(), CollectionState: Ledger, CreateProviders: ParquetFactory });
  await Job.Start(Settings({ Source: "wikidump", Query: "simple" }), () => {});
  const Cursors = Ledger.Writes.map((W) => W.Cursor);
  for (const Offset of [2, 4, 6]) expect(Cursors).toContain(JSON.stringify({ Shard: 0, Offset }));
  // The write that records offset 2 already carries batch 1's lifetime count (100 + 2).
  expect(Ledger.Writes.find((W) => W.Cursor === JSON.stringify({ Shard: 0, Offset: 2 }))!.Collected).toBe(102);
  const Final = await Ledger.Get("wikidump:simple");
  expect(JSON.parse(Final!.Cursor)).toEqual({ Shard: 1, Offset: 0 });
  expect(Final!.Collected).toBe(106);
});

test("(e) a second Start while running is rejected without touching the first run", async () => {
  let Release: () => void = () => {};
  const Gate = new Promise<void>((Resolve) => {
    Release = Resolve;
  });
  const Factory: ProviderFactory = (_S, Hooks) => [
    {
      Name: "slow",
      Fetch: async () => {
        await Hooks.OnRepoReady("slow-0", Docs(0, 2));
        await Gate; // hold the run open
        return [];
      },
    },
  ];
  const Job = new LearnJob({ ...Quiet, KindStore: () => new InMemoryDocumentStore(), CollectionState: new InMemoryCollectionStateStore(), CreateProviders: Factory });
  const First = Collect();
  const Running = Job.Start(Settings(), First.OnEvent);
  expect(Job.Running).toBe(true);
  const Second = Collect();
  await Job.Start(Settings(), Second.OnEvent);
  expect(Second.Events).toEqual([{ kind: "error", message: expect.stringContaining("already in progress") }]);
  expect(Second.Dones().length).toBe(0); // must not emit a done that would end the first run
  Release();
  await Running;
  expect(First.Dones().length).toBe(1);
  expect(First.Dones()[0]!.ingested).toBe(2);
  expect(Job.Running).toBe(false);
});

test("(f) exactly one done per run — provider error, runner error, and user stop", async () => {
  // Provider error: non-terminal error event, then one failed done.
  const Broken = new LearnJob({ ...Quiet, KindStore: () => new InMemoryDocumentStore(), CollectionState: new InMemoryCollectionStateStore(), CreateProviders: () => [{ Name: "net", Fetch: async () => { throw new Error("ECONNRESET"); } }] });
  const A = Collect();
  await Broken.Start(Settings(), A.OnEvent);
  expect(A.Dones().length).toBe(1);
  expect(A.Dones()[0]!.failed).toBe(true);
  expect(A.Dones()[0]!.message).toBe("net: ECONNRESET");

  // Unexpected runner error before any provider runs (store's Sources() throws).
  class NoSources extends InMemoryDocumentStore {
    override async Sources(): Promise<string[]> {
      throw new Error("db unreachable");
    }
  }
  const B = Collect();
  await new LearnJob({ ...Quiet, KindStore: () => new NoSources(), CollectionState: new InMemoryCollectionStateStore(), CreateProviders: () => [] }).Start(Settings({ SkipLearned: true }), B.OnEvent);
  expect(B.Dones().length).toBe(1);
  expect(B.Dones()[0]!.failed).toBe(true);
  expect(B.Dones()[0]!.message).toBe("db unreachable");

  // User stop between batches: clean stop, not a failure, still exactly one done.
  const Stop = new AbortController();
  const Factory: ProviderFactory = (_S, Hooks) => [
    {
      Name: "p",
      Fetch: async () => {
        Hooks.OnRepoStart("b0");
        await Hooks.OnRepoReady("p-0", Docs(0, 2));
        Stop.abort();
        Hooks.OnRepoStart("b1"); // the Stop boundary throws here
        await Hooks.OnRepoReady("p-1", Docs(2, 2));
        return [];
      },
    },
  ];
  const C = Collect();
  await new LearnJob({ ...Quiet, KindStore: () => new InMemoryDocumentStore(), CollectionState: new InMemoryCollectionStateStore(), CreateProviders: Factory }).Start(Settings(), C.OnEvent, Stop.signal);
  expect(C.Dones().length).toBe(1);
  expect(C.Dones()[0]!.stopped).toBe(true);
  expect(C.Dones()[0]!.failed).toBe(false);
  expect(C.Dones()[0]!.ingested).toBe(2);
  expect(C.Events.some((E) => E.kind === "error")).toBe(false);
});

test("exhaustion for a repo-capped source counts repos, not files", async () => {
  const Store = new InMemoryDocumentStore();
  const Ledger = new InMemoryCollectionStateStore();
  // Two repos of 30 files each, already stored: re-collecting yields only duplicates.
  const Repos = [Docs(0, 30), Docs(30, 30)];
  const Factory: ProviderFactory = (_S, H) => [BatchProvider("github-repo", Repos)(H)];
  const Job = new LearnJob({ ...Quiet, KindStore: () => Store, CollectionState: Ledger, CreateProviders: Factory });
  await Job.Start(Settings({ Source: "github", Query: "q", MaxRepos: 10 }), () => {});
  const Sink = Collect();
  await Job.Start(Settings({ Source: "github", Query: "q", MaxRepos: 10 }), Sink.OnEvent);
  // 2 repos < cap of 10 repos -> ran dry -> exhausted (60 files vs 10 would have said "capped").
  expect(Sink.Dones()[0]!.exhausted).toBe(true);
  expect((await Ledger.Get("github:q"))!.Exhausted).toBe(true);
});

test("a failed ledger read leaves the lifetime total untouched", async () => {
  class FlakyLedger extends InMemoryCollectionStateStore {
    override async Get(): Promise<CollectionState | null> {
      throw new Error("ledger down");
    }
  }
  const Ledger = new FlakyLedger();
  await Ledger.Upsert({ SourceKey: "oasst:en", Kind: "conversation", Cursor: "{}", Collected: 500, Exhausted: false, UpdatedAt: "2026-11-01T00:00:00.000Z" });
  const Job = new LearnJob({ ...Quiet, KindStore: () => new InMemoryDocumentStore(), CollectionState: Ledger, CreateProviders: (_S, H) => [BatchProvider("p", [Docs(0, 3)])(H)] });
  await Job.Start(Settings(), () => {});
  expect((await Ledger.All())[0]!.Collected).toBe(500);
});
