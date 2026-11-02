// The Learn (data-collection) runner — STAGE 1 of the Foundry pipeline, extracted from the dashboard
// script so it is unit-testable with in-memory stores and fake providers. One run: route the source to
// its kind table, resume from the collection ledger, stream batches from the providers into the store
// (bounded by a run-wide failure budget), persist cursor + lifetime counts write-through after EVERY
// batch, and finish with exactly ONE terminal `done` event — never a terminal `error`.
//
// Event contract (what OnEvent receives, in order):
//   scanning / repo / repo-progress  — transient progress (any number, any order)
//   error   { message }              — NON-terminal: one per provider failure; the run continues/ends
//   done    LearnDoneEvent           — terminal, exactly once per accepted run (failed/stopped flags)
// A Start while a run is in flight emits one `error` and returns WITHOUT `done` (the running job's
// terminal event is still pending; emitting a second `done` would end the wrong run).

import type { LearnEvent, LearnSettings, LearnFn } from "./DashboardTypes.ts";
import type { DataKind } from "./DataKinds.ts";
import type { DocumentStore } from "./DocumentStore.ts";
import type { CollectionState, CollectionStateStore } from "./CollectionState.ts";
import { ComputeExhausted } from "./CollectionState.ts";
import type { SourceInput, IngestStats } from "./Ingest.ts";
import { IngestDocuments, IngestFailureBudget, IngestAbortError } from "./Ingest.ts";
import type { ProviderFactory, ProviderHooks } from "./KindRouting.ts";
import { SourceKindFor, CapUnitFor, CreateLearnProviders } from "./KindRouting.ts";

/** Hashing-embedding width for collected documents (matches IngestDocuments' default). */
export const LearnEmbeddingDim = 256;
/** Target number of repo-progress events per stored batch (progress granularity vs. event volume). */
export const ProgressUpdatesPerBatch = 50;
/** The error a provider's OnRepoStart throws at a Stop boundary — recognized as a clean stop. */
export const StoppedByUserMessage = "stopped by user";

/** The terminal event. Extends the dashboard's `done` with failure/stop reporting:
 *  failedDocs = documents whose store upsert failed; failed = the run did not complete cleanly (an
 *  ingest-budget abort, a provider error, or an unexpected runner error) with `message` = the first
 *  cause; stopped = the user pressed Stop. */
export type LearnDoneEvent = Extract<LearnEvent, { kind: "done" }> & {
  failedDocs: number;
  failed: boolean;
  stopped: boolean;
  message?: string;
};

export type LearnJobDeps = {
  KindStore: (Kind: DataKind) => DocumentStore;
  CollectionState: CollectionStateStore;
  CreateProviders?: ProviderFactory; // default: the production network/disk providers
  GitHubToken?: string;
  Now?: () => string; // ISO timestamp source (injected for deterministic tests)
  Log?: (Message: string) => void;
  Warn?: (Message: string) => void;
};

// Batches = successfully stored provider batches; for repo providers one batch IS one repository.
type Totals = { Ingested: number; New: number; Duplicate: number; Failed: number; Batches: number };

/** Exhaustion in the cap's own unit: repos (stored repo batches) for GitHub, documents (attempted
 *  upserts) for datasets; sources whose providers ignore the cap compare against an unbounded cap. */
function RunExhausted(Settings: LearnSettings, Semantics: "bounded" | "streaming", Total: Totals): boolean {
  const Unit = CapUnitFor(Settings.Source);
  if (Unit === null) return ComputeExhausted(Semantics, Total.New, Total.Duplicate, { Count: 0, Unit: "documents" }, { Count: Number.POSITIVE_INFINITY, Unit: "documents" });
  const Produced = Unit === "repos" ? Total.Batches : Total.Ingested + Total.Failed;
  return ComputeExhausted(Semantics, Total.New, Total.Duplicate, { Count: Produced, Unit }, { Count: Settings.MaxRepos, Unit });
}

export class LearnJob {
  private Active = false;
  private Deps: LearnJobDeps;

  constructor(Deps: LearnJobDeps) {
    this.Deps = Deps;
  }

  /** Whether a run is in flight (the overlap guard's state). */
  get Running(): boolean {
    return this.Active;
  }

  /** The dashboard-facing runner (LearnFn) bound to this job. */
  AsLearnFn(): LearnFn {
    return (Settings, OnEvent, Signal) => this.Start(Settings, OnEvent, Signal);
  }

  /** Run one collect. Rejects (one `error`, no `done`) when a run is already in flight. */
  async Start(Settings: LearnSettings, OnEvent: (Event: LearnEvent) => void, Signal?: AbortSignal): Promise<void> {
    if (this.Active) {
      OnEvent({ kind: "error", message: "a learn run is already in progress — wait for it to finish or press Stop" });
      return;
    }
    this.Active = true;
    try {
      await this.Run(Settings, OnEvent, Signal);
    } finally {
      this.Active = false;
    }
  }

  private async Run(Settings: LearnSettings, OnEvent: (Event: LearnEvent) => void, Signal?: AbortSignal): Promise<void> {
    const Log = this.Deps.Log ?? ((Message: string): void => console.log(Message));
    const Warn = this.Deps.Warn ?? ((Message: string): void => console.warn(Message));
    const Now = this.Deps.Now ?? ((): string => new Date().toISOString());
    const Ledger = this.Deps.CollectionState;

    const Kind = SourceKindFor(Settings);
    const StateKey = `${Settings.Source}:${Settings.Query}`;
    const Total: Totals = { Ingested: 0, New: 0, Duplicate: 0, Failed: 0, Batches: 0 };
    let Semantics: "bounded" | "streaming" = "bounded";
    let FailMessage: string | null = null;
    let PrevState: CollectionState | null = null;
    let RunCursor = "{}";
    let LedgerLoaded = false; // false => the prior lifetime total is unknown; never overwrite it
    const IsStopped = (): boolean => Signal?.aborted === true;

    // Write-through ledger: cursor + LIFETIME new-doc total after every stored batch, so a Stop/crash/
    // restart resumes exactly where storage got to. Non-fatal — a ledger hiccup must not fail a collect.
    const Persist = async (Exhausted: boolean): Promise<void> => {
      if (!LedgerLoaded) return; // a failed read must not let this run clobber the lifetime total
      try {
        await Ledger.Upsert({ SourceKey: StateKey, Kind, Cursor: RunCursor, Collected: (PrevState?.Collected ?? 0) + Total.New, Exhausted, UpdatedAt: Now() });
      } catch (Caught) {
        Warn(`[learn] collection-state write failed (non-fatal): ${(Caught as Error).message}`);
      }
    };
    const Accumulate = (Stats: IngestStats): void => {
      Total.Ingested += Stats.Ingested;
      Total.New += Stats.New;
      Total.Duplicate += Stats.Duplicate;
      Total.Failed += Stats.Failed;
    };

    try {
      const Store = this.Deps.KindStore(Kind);
      const Learned = new Set(Settings.SkipLearned ? await Store.Sources() : []);
      // Ledger read: a streaming parquet source resumes from its persisted cursor. Non-fatal.
      try {
        PrevState = await Ledger.Get(StateKey);
        LedgerLoaded = true;
      } catch (Caught) {
        Warn(`[learn] collection-state read failed (non-fatal; ledger left untouched this run): ${(Caught as Error).message}`);
      }
      RunCursor = PrevState?.Cursor ?? "{}";

      const IngestedAt = Now();
      const Budget = new IngestFailureBudget(Warn); // ONE budget for the whole run (all batches)
      const Hooks: ProviderHooks = {
        Kind,
        PrevCursor: PrevState?.Cursor,
        GitHubToken: this.Deps.GitHubToken,
        Skip: (Repo) => Learned.has(Repo),
        // Fired before each (slow) download: show a "working" status, and stop cleanly at this boundary
        // if the user pressed Stop — already-stored batches stay (collect is incremental).
        OnRepoStart: (Name) => {
          if (IsStopped()) throw new Error(StoppedByUserMessage);
          OnEvent({ kind: "scanning", label: `downloading ${Name}…` });
        },
        OnRepo: (Info) => {
          OnEvent({ kind: "repo", repo: Info.Repo, level: Info.Assessment.Level, files: Info.Assessment.FileCount, bytes: Info.Assessment.TotalBytes, ingested: Info.Ingested, reason: Info.Reason ?? null });
        },
        // INCREMENTAL: store each batch the moment it arrives, then persist the ledger.
        OnRepoReady: async (Source: string, Files: SourceInput[]): Promise<boolean> => {
          OnEvent({ kind: "scanning", label: `storing ${Source} (${Files.length} files)…` });
          const Stride = Math.max(1, Math.floor(Files.length / ProgressUpdatesPerBatch));
          const OnEach = (Done: number, Count: number): void => {
            if (Done % Stride === 0 || Done === Count) OnEvent({ kind: "repo-progress", repo: Source, filesDone: Done, filesTotal: Count });
          };
          let Stats: IngestStats;
          try {
            Stats = await IngestDocuments(Files, Store, IngestedAt, LearnEmbeddingDim, OnEach, Budget);
          } catch (Caught) {
            if (Caught instanceof IngestAbortError) {
              Accumulate(Caught.Stats); // count what WAS stored before the budget tripped
              await Persist(false);
            }
            throw Caught;
          }
          Accumulate(Stats);
          Total.Batches++;
          await Persist(false);
          return Stats.Failed === 0;
        },
        OnCursor: async (Shard, Offset) => {
          RunCursor = JSON.stringify({ Shard, Offset });
          await Persist(false);
        },
      };

      const Providers = (this.Deps.CreateProviders ?? CreateLearnProviders)(Settings, Hooks);
      // Semantics tells the user whether running again can grow the corpus (streaming) or not (bounded).
      Semantics = Providers.some((P) => P.Semantics === "streaming") ? "streaming" : "bounded";
      OnEvent({ kind: "scanning", label: "searching for repos…" });
      Log(`[learn] start: source=${Settings.Source} query="${Settings.Query}" maxRepos=${Settings.MaxRepos} minLevel=${Settings.MinLevel} skipLearned=${Settings.SkipLearned} (already know ${Learned.size} repos)`);

      for (const Provider of Providers) {
        if (IsStopped()) break;
        try {
          await Provider.Fetch(Settings.Query, Settings.MaxRepos, Signal); // batches stream into OnRepoReady
        } catch (Caught) {
          const Message = (Caught as Error).message;
          if (IsStopped() || Message.includes(StoppedByUserMessage)) {
            Log(`[learn] stopped by user during ${Provider.Name}`);
            break;
          }
          // A provider failure is never swallowed: logged, surfaced as a NON-terminal error event, and
          // recorded for the terminal done. An ingest-budget abort ends the whole run (the store is
          // broken — other providers would fail the same way); any other failure moves to the next one.
          Warn(`[learn] provider ${Provider.Name} failed: ${Message}`);
          OnEvent({ kind: "error", message: `${Provider.Name}: ${Message}` });
          FailMessage ??= `${Provider.Name}: ${Message}`;
          if (Caught instanceof IngestAbortError) break;
        }
      }
    } catch (Caught) {
      // Unexpected runner failure (e.g. the store's Sources() query) — still ends with one done.
      const Message = (Caught as Error).message;
      Warn(`[learn] run failed: ${Message}`);
      FailMessage ??= Message;
    }

    // Exhausted ONLY when a clean, unstopped bounded run ran DRY (0 new, only dups) before its cap,
    // measured in the cap's own unit (repos for GitHub, documents for datasets) — see ComputeExhausted.
    const Stopped = IsStopped();
    const Exhausted = FailMessage === null && !Stopped && RunExhausted(Settings, Semantics, Total);
    await Persist(Exhausted);

    const Collected = (PrevState?.Collected ?? 0) + Total.New;
    Log(`[learn] done: ${Total.New} new · ${Total.Duplicate} duplicate · ${Total.Failed} failed · ${Total.Ingested} processed (${Semantics}${Exhausted ? ", exhausted" : ""}${Stopped ? ", stopped" : ""}${FailMessage !== null ? ", FAILED" : ""}; lifetime ${Collected})`);
    const Done: LearnDoneEvent = {
      kind: "done",
      ingested: Total.Ingested,
      new: Total.New,
      duplicate: Total.Duplicate,
      failedDocs: Total.Failed,
      semantics: Semantics,
      collected: Collected,
      exhausted: Exhausted,
      failed: FailMessage !== null,
      stopped: Stopped,
      ...(FailMessage !== null ? { message: FailMessage } : {}),
    };
    OnEvent(Done);
  }
}
