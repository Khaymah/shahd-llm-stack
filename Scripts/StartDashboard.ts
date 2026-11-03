// Serve the interactive Foundry control panel: a clear 3-stage pipeline — (1) COLLECT DATA into
// Postgres, (2) TRAIN the model on that data (a subprocess, so it never blocks the dashboard and can
// run alongside collection), (3) CHAT to test the trained model. Reads store + token from .env
// (Bun): Postgres when DATABASE_URL is set. The trained checkpoint lives in Postgres; after a Train
// run finishes it is hot-reloaded into the chat model with no restart.
//   bun run foundry:dashboard        # then open http://localhost:8090
//   bun run foundry:dashboard:dev    # same, restarting on source changes (--watch)
//
// This file is the COMPOSITION ROOT only: it resolves stores from the environment and wires the
// testable modules — LearnJob (collect), TrainJob (train subprocess), ModelHolder (live chat model).

import { StartDashboard, InMemoryDocumentStore, InMemoryCollectionStateStore, PostgresCheckpointStore } from "../Foundry/FoundryBarrel.ts";
import type { CollectionStateStore, CheckpointSummary, DataKind, DocumentStore } from "../Foundry/FoundryBarrel.ts";
import type { ChatStore } from "../Foundry/ChatStore.ts";
import { PostgresChatStore } from "../Foundry/PostgresChatStore.ts";
import { InMemoryChatStore } from "../Foundry/InMemoryChatStore.ts";
import { ChatService } from "../Foundry/ChatService.ts";
import { LearnJob } from "../Foundry/LearnJob.ts";
import { CreateTrainRunner } from "../Foundry/TrainJob.ts";
import { ModelHolder, CreateChatStream } from "../Foundry/ModelHolder.ts";
import { ResolveFoundryStores, GitHubToken, DefaultCheckpointName, CheckpointFile, SelfConsistencySamples } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

const DbUrl = process.env["DATABASE_URL"];
const HasDb = DbUrl !== undefined && DbUrl !== "";
// ONE shared checkpoint-store pool for the whole dashboard lifetime — reused by reload/list/delete
// instead of opening (and, on error paths, LEAKING) a fresh postgres pool on every call.
const SharedCkptStore = HasDb ? new PostgresCheckpointStore(DbUrl) : null;
// Per-kind stores (Postgres): collection routes each source to its own kind table, so data types stay
// separated. Without a database, fall back to one in-memory store (no kind separation).
const Stores = HasDb ? ResolveFoundryStores() : null;
const FallbackStore = new InMemoryDocumentStore();
const KindStore = (Kind: DataKind): DocumentStore => (Stores !== null ? Stores.Kind(Kind) : FallbackStore);
// Collection ledger (progress/exhausted per source) — on the shared Postgres pool, or in-memory without a DB.
const CollState: CollectionStateStore = Stores !== null ? Stores.CollectionState() : new InMemoryCollectionStateStore();
const InspectStore = KindStore("code"); // the dashboard's inspection/stats view (code is the bulk)

// STAGE 3 — the live chat model, swapped in place when a training run finishes or the user picks one.
const Model = new ModelHolder({ Store: SharedCkptStore, DefaultName: DefaultCheckpointName(), FallbackFile: CheckpointFile() });
const Chats: ChatStore = HasDb ? new PostgresChatStore(DbUrl) : new InMemoryChatStore();
const Chat = new ChatService(Chats, CreateChatStream(Model, { SelfConsistencySamples: SelfConsistencySamples() }));

// STAGE 1 — collect data; STAGE 2 — train (subprocess).
const Learn = new LearnJob({ KindStore, CollectionState: CollState, GitHubToken: GitHubToken() });
const Train = CreateTrainRunner();

// List saved checkpoints (the chat-model picker).
async function ListCheckpoints(): Promise<CheckpointSummary[]> {
  if (SharedCkptStore === null) return [];
  try {
    return await SharedCkptStore.List();
  } catch (Caught) {
    console.warn(`[dashboard] checkpoint list failed: ${(Caught as Error).message}`);
    return [];
  }
}

async function DeleteCheckpoint(Name: string): Promise<void> {
  if (SharedCkptStore === null) return;
  try {
    await SharedCkptStore.Delete(Name);
  } catch (Caught) {
    console.warn(`[dashboard] delete checkpoint "${Name}" failed: ${(Caught as Error).message}`);
  }
}

await Model.Reload(DefaultCheckpointName());

const Port = Number(ReadArg("--Port=", "8090"));
const Instance = StartDashboard(InspectStore, Port, Learn.AsLearnFn(), {
  Chat,
  GetModel: () => Model.Info,
  GetModelName: () => Model.Name,
  Train,
  OnTrained: async (Name: string) => {
    await Model.Reload(Name);
  },
  Checkpoints: ListCheckpoints,
  LoadModel: async (Name: string) => {
    await Model.Reload(Name);
  },
  KindStats: () => (Stores !== null ? Stores.Stats() : Promise.resolve([])),
  Collection: () => CollState.All(),
  // Data browser + cleanup: kind-aware, so it reaches every documents_<kind> table (Postgres) or the
  // single fallback store (in-memory). Each op resolves the right store and delegates to it.
  Browse: (Kind, Filter, Offset, Limit) => KindStore(Kind).ListSummaries({ Filter, Offset, Limit }),
  Facets: (Kind) => KindStore(Kind).Stats(),
  DocContent: (Kind, Id) => KindStore(Kind).DocumentById(Id),
  DeleteDoc: (Kind, Id) => KindStore(Kind).DeleteById(Id),
  DeleteMatching: (Kind, Filter) => KindStore(Kind).DeleteMatching(Filter),
  DeleteCheckpoint,
});
console.log(`Foundry control panel: ${Instance.url.href}  (store=${Stores !== null ? "postgres/per-kind" : "memory"}, ${await InspectStore.Count()} code docs, GitHub token: ${GitHubToken() ? "yes" : "no"})`);
console.log(`  chat model: ${Model.Source} (memory: ${HasDb ? "postgres" : "in-memory"})`);
