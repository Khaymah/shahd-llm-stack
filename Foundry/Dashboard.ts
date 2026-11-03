// Interactive Foundry control panel (M9/M11/M13). A Bun-served page + JSON API + a WebSocket for
// realtime push over a DocumentStore:
//   GET  /                     the control-panel page      GET  /chat            the chat page
//   GET  /api/stats            aggregate counts            GET  /api/repos       per-repo rollup
//   GET  /api/system           host + compute snapshot     GET  /api/model       loaded-model breakdown
//   GET  /api/documents?source files of one repo           GET  /api/file?id     one file's full content
//   GET  /api/chat/conversations   history list            GET  /api/chat/conversation?id  its messages
//   POST /api/chat/delete      remove a conversation       POST /api/learn       start a Learn run
//   GET  /api/learn/stream     SSE progress (fallback)     WS   /ws              realtime push + chat
// The WebSocket pushes system/stats snapshots on a ticker, streams Learn progress (incl. per-repo
// file progress), and streams chat token-by-token. Learn runner + chat service + model info are
// INJECTED so the handler is testable and never imports the network/model layers itself.
// Local-only by design (no login): it binds to loopback and every /api/* request and /ws upgrade must
// carry a loopback Origin/Host (DashboardGuard) — the boundary against cross-site pages + DNS rebinding.

import type { Server, ServerWebSocket } from "bun";
import type { DocumentStore, DocumentFilter, SummaryPage, FoundryStats } from "./DocumentStore.ts";
import type { DocumentRecord } from "./DocumentRecord.ts";
import type { DataKind } from "./DataKinds.ts";
import { IsDataKind } from "./DataKinds.ts";
import type { CollectionState } from "./CollectionState.ts";
import type { LearnSettings, LearnEvent, LearnFn, TrainSettings, TrainEvent, TrainFn } from "./DashboardTypes.ts";
import type { KindStat } from "./FoundryStores.ts";
import type { RepoLevel } from "./RepoQuality.ts";
import type { ChatService } from "./ChatService.ts";
import type { ModelInfo } from "./ModelInfo.ts";
import type { CheckpointSummary } from "./PostgresCheckpointStore.ts";
import { DashboardHtml } from "./DashboardHtml.ts";
import { RenderChatHtml } from "./ChatHtml.ts";
import { GetSystemInfo } from "./SystemInfo.ts";
import { DefaultConfig } from "../Brain/Config/Constants.ts";
import { IsLocalOrigin, IsLocalHost, IsJsonContentType, IsValidConvId, ResolveBindHost, BindHostAliases, BindEnvVar, DefaultDashboardPort, NewCspNonce, DashboardPageCsp, ChatPageCsp } from "./DashboardGuard.ts";

export type DashboardHandler = (Req: Request) => Promise<Response>;
export type DashboardOptions = {
  Chat?: ChatService;
  GetModel?: () => ModelInfo | null; // a getter so the panel reflects a model reloaded after training
  GetModelName?: () => string; // name of the currently loaded chat model (for the picker)
  Train?: TrainFn; // model training (subprocess) — distinct from Learn/data-collection
  OnTrained?: (Name: string) => Promise<void>; // load the freshly-trained checkpoint into the chat model
  Checkpoints?: () => Promise<CheckpointSummary[]>; // list saved models (the chat-model picker)
  LoadModel?: (Name: string) => Promise<void>; // switch the chat model to a saved checkpoint
  KindStats?: () => Promise<KindStat[]>; // per-kind document counts/bytes (the data-separation breakdown)
  Collection?: () => Promise<CollectionState[]>; // the collection ledger: lifetime/exhausted/cursor per source
  // Data browser (per-kind, paginated) — kind-aware so it reaches every documents_<kind> table, not
  // just the one InspectStore the panel is built with. All injected so Dashboard.ts stays DB-agnostic.
  Browse?: (Kind: DataKind, Filter: DocumentFilter, Offset: number, Limit: number) => Promise<SummaryPage>;
  Facets?: (Kind: DataKind) => Promise<FoundryStats>; // distinct tier/lang/license values for the filter dropdowns
  DocContent?: (Kind: DataKind, Id: string) => Promise<DocumentRecord | null>; // one doc's full content (viewer)
  DeleteDoc?: (Kind: DataKind, Id: string) => Promise<number>; // remove one document
  DeleteMatching?: (Kind: DataKind, Filter: DocumentFilter) => Promise<number>; // bulk corpus cleanup
  DeleteCheckpoint?: (Name: string) => Promise<void>; // remove a saved model
  // Server-side cap on tokens per chat generation (the client's maxTokens is clamped to it). Defaults
  // to Config.Limits.MaxNewTokens from Constants so the operator's limit is never exceeded over /ws.
  MaxNewTokens?: number;
};

const ReplayBufferCap = 500; // replayed Learn/Train events kept per run (first start event + the newest)
const MaxGlobalGenerations = 2; // chat generations in flight across ALL sockets (CPU-bound model)
const DefaultChatMaxTokens = 160; // per-turn token budget when the client sends none
const DefaultChatTemperature = 0.8;
const SnapshotIntervalMs = 3000; // system/stats push cadence

// Append to a run's replay buffer without unbounded growth: once over the cap, drop the OLDEST events
// after index 0 — index 0 is always the run's start event, which a late joiner needs to render the run.
function PushBounded<T>(Events: T[], Event: T): void {
  Events.push(Event);
  if (Events.length > ReplayBufferCap) Events.splice(1, Events.length - ReplayBufferCap);
}

// Build a document filter from a flat record (query params or a POST body use the same lowercase keys).
function ParseFilter(O: Record<string, unknown>): DocumentFilter {
  const Filter: DocumentFilter = {};
  const Tier = O["tier"];
  if (Tier === "Filtered" || Tier === "Raw" || Tier === "Rejected") Filter.Tier = Tier;
  if (typeof O["lang"] === "string" && O["lang"] !== "") Filter.Lang = O["lang"];
  if (typeof O["license"] === "string" && O["license"] !== "") Filter.License = O["license"];
  if (typeof O["q"] === "string" && O["q"] !== "") Filter.Search = O["q"];
  return Filter;
}

function SafeName(Value: unknown): string {
  const Raw = typeof Value === "string" ? Value.trim() : "";
  const Clean = Raw.replace(/[^a-zA-Z0-9_.-]/g, "-").slice(0, 40);
  return Clean === "" ? "foundry" : Clean; // safe as a checkpoint key
}

function ParseTrainSettings(Body: Record<string, unknown>): TrainSettings {
  return {
    Kind: Body["Kind"] === "chat" ? "chat" : "pretrain",
    Name: SafeName(Body["Name"]),
    Resume: Body["Resume"] === true,
    Steps: ToNum(Body["Steps"], 500),
    CorpusMb: ToNum(Body["CorpusMb"], 1.5),
    EmbedDim: ToNum(Body["EmbedDim"], 96),
    NumLayers: ToNum(Body["NumLayers"], 3),
    NumHeads: ToNum(Body["NumHeads"], 4),
    BlockSize: ToNum(Body["BlockSize"], 96),
    Merges: ToNum(Body["Merges"], 256),
    BatchSize: ToNum(Body["BatchSize"], 16),
    // The one field the sanitizer previously DROPPED — the client sent Workers, this whitelist ate
    // it, and TrainFn's ?? fallback quietly ran a 4-hour sequential pretrain. Default 8: dashboard
    // training should be fast even for an older cached client that doesn't send the field.
    Workers: ToNum(Body["Workers"], 8),
    Precision: Body["Precision"] === "F32" ? "F32" : "F64",
    // Chat warm start: SafeName never returns "" (it falls back to "foundry"), so gate on the RAW
    // value — an absent/empty From must stay undefined, not become a phantom base named "foundry".
    From: typeof Body["From"] === "string" && Body["From"].trim() !== "" ? SafeName(Body["From"]) : undefined,
    KnowledgeMb: ToNum(Body["KnowledgeMb"], 0),
    ConvCount: ToNum(Body["ConvCount"], 4000),
    CodeSamples: ToNum(Body["CodeSamples"], 4000),
    MultiTurn: ToNum(Body["MultiTurn"], 3000), // must pass the sanitizer or the CLI silently gets the default (the Workers lesson)
  };
}
type WsData = { Generating: boolean }; // per-connection: a chat reply is in flight on this socket
type Ws = ServerWebSocket<WsData>;
type ChatMsg = { convId?: unknown; message?: unknown; temperature?: unknown; maxTokens?: unknown };

// RunId distinguishes runs so a previous run's late settlement can never close the current one.
type JobState = { Running: boolean; RunId: number; Events: LearnEvent[]; Listeners: Set<(Event: LearnEvent) => void>; Controller: AbortController | null };

function ToNum(Value: unknown, Default: number): number {
  if (Value === null || Value === undefined || Value === "") return Default; // Number(null) is 0, not the default
  const N = Number(Value);
  return Number.isFinite(N) ? N : Default;
}

function IsLearnSource(Value: unknown): Value is LearnSettings["Source"] {
  return Value === "github" || Value === "local" || Value === "both" || Value === "oasst" || Value === "oasst2" || Value === "wikipedia" || Value === "gsm8k" || Value === "wikidump" || Value === "folder" || Value === "stackexchange";
}

function ParseSettings(Body: Record<string, unknown>): LearnSettings {
  const Source = Body["Source"];
  const Level = Body["MinLevel"];
  const Repos = Body["Repos"];
  return {
    // Accept EVERY valid source (github/local/both + the general text sources oasst/oasst2/wikipedia);
    // anything unknown falls back to github. The old check listed only local/both, so it silently
    // coerced oasst/oasst2/wikipedia to github — the "I picked OASST but it crawled GitHub" bug.
    Source: IsLearnSource(Source) ? Source : "github",
    Query: typeof Body["Query"] === "string" ? Body["Query"] : "language:typescript stars:>1000",
    Repos: Array.isArray(Repos) ? Repos.map(String) : ["."],
    MinLevel: (Level === "high" || Level === "low" ? Level : "medium") as RepoLevel,
    MaxRepos: ToNum(Body["MaxRepos"], 5),
    MaxFilesPerRepo: ToNum(Body["MaxFilesPerRepo"], 2000),
    MaxBytesPerRepo: ToNum(Body["MaxBytesPerRepo"], 32_000_000),
    MaxContentBytes: ToNum(Body["MaxContentBytes"], 512_000),
    SkipLearned: Body["SkipLearned"] !== false,
    // "folder" source: which kind its files become + the license to record (defaults suit Gutenberg books).
    Kind: IsDataKind(String(Body["Kind"])) ? (Body["Kind"] as LearnSettings["Kind"]) : undefined,
    License: typeof Body["License"] === "string" && Body["License"] !== "" ? Body["License"] : undefined,
  };
}

function Json(Data: unknown, Status = 200): Response {
  return Response.json(Data, { status: Status });
}

function HtmlResponse(Html: string, Csp: string): Response {
  return new Response(Html, { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": Csp, "X-Content-Type-Options": "nosniff" } });
}

// Everything the panel needs, built once and shared by the HTTP handler and the WebSocket handler.
export function CreateDashboardParts(Store: DocumentStore, Learn?: LearnFn, Options: DashboardOptions = {}) {
  const Job: JobState = { Running: false, RunId: 0, Events: [], Listeners: new Set(), Controller: null };
  let Bound: Server<WsData> | null = null;
  // Guard parameters: the real port + any concrete bind address arrive with AttachServer.
  let GuardPort = DefaultDashboardPort;
  let GuardHosts: string[] = [];
  let Clients = 0; // open WebSockets — background snapshots only run while someone is watching
  let ActiveGenerations = 0;
  const MaxNewTokens = Options.MaxNewTokens ?? DefaultConfig.Limits.MaxNewTokens;
  const IsLocalRequest = (Req: Request): boolean => IsLocalOrigin(Req.headers.get("origin"), GuardPort, GuardHosts) && IsLocalHost(Req.headers.get("host"), GuardPort, GuardHosts);
  const Publish = (Message: unknown): void => {
    Bound?.publish("all", JSON.stringify(Message));
  };
  const ModelMsg = (): { type: string; data: ModelInfo | null; name: string } => ({ type: "model", data: Options.GetModel?.() ?? null, name: Options.GetModelName?.() ?? "" });

  const Emit = (Event: LearnEvent): void => {
    // repo-progress and scanning are transient status — broadcast live but do NOT buffer in the replay
    // array (re-sent to every new/reconnecting client), so memory stays O(repos).
    if (Event.kind !== "repo-progress" && Event.kind !== "scanning") PushBounded(Job.Events, Event);
    for (const Listener of Job.Listeners) Listener(Event);
    Publish({ type: "learn", event: Event });
    // ONLY done is terminal: the runner reports non-fatal errors (one provider failed) mid-run and keeps
    // going, so flipping Running on error let a second run start on top of a live one.
    if (Event.kind === "done") Job.Running = false;
  };

  const StartLearn = (Settings: LearnSettings): boolean => {
    if (Learn === undefined || Job.Running) return false;
    Job.Running = true;
    Job.Events = [];
    Job.Controller = new AbortController();
    const RunId = ++Job.RunId;
    Emit({ kind: "start", query: Settings.Query, source: Settings.Source, repos: Settings.MaxRepos });
    void Learn(Settings, Emit, Job.Controller.signal)
      .catch((Caught) => Emit({ kind: "error", message: (Caught as Error).message }))
      .finally(() => {
        // Contract: the runner always ends with done (even after errors). If it settled without one
        // (it threw), close the run here so the panel can never stick in "running".
        if (Job.RunId === RunId && Job.Running) Emit({ kind: "done", ingested: 0, failed: true, message: "learn runner ended without a done event" });
      });
    return true;
  };
  const StopLearn = (): void => Job.Controller?.abort();

  // Model TRAINING — a separate pipeline stage from Learn/data-collection. On completion the freshly
  // trained checkpoint is reloaded into the live chat model, and the new model panel is broadcast.
  const TrainJob: { Running: boolean; Events: TrainEvent[]; Controller: AbortController | null; Name: string } = { Running: false, Events: [], Controller: null, Name: "foundry" };
  const TrainEmit = (Event: TrainEvent): void => {
    PushBounded(TrainJob.Events, Event); // buffered (bounded) so reconnects see progress
    Publish({ type: "train", event: Event });
    if (Event.kind === "train-done" || Event.kind === "train-error") {
      TrainJob.Running = false;
      // A graceful pause saved a checkpoint at the exact stop step — hot-reload it like a finished
      // run, so the chat model IS the paused weights (pause -> try in chat -> resume, no re-pick).
      if (Event.kind === "train-done" || Event.paused === true) {
        void (async () => {
          await Options.OnTrained?.(TrainJob.Name); // swap the freshly-trained checkpoint into the chat model
          Publish(ModelMsg());
          Publish({ type: "checkpoints", data: (await Options.Checkpoints?.()) ?? [] });
        })();
      }
    }
  };
  const StartTrain = (Settings: TrainSettings): boolean => {
    if (Options.Train === undefined || TrainJob.Running) return false;
    TrainJob.Running = true;
    TrainJob.Events = [];
    TrainJob.Controller = new AbortController();
    TrainJob.Name = Settings.Name;
    TrainEmit({ kind: "train-start", steps: Settings.Steps });
    void Options.Train(Settings, TrainEmit, TrainJob.Controller.signal).catch((Caught) => TrainEmit({ kind: "train-error", message: (Caught as Error).message }));
    return true;
  };
  const StopTrain = (): void => TrainJob.Controller?.abort();

  const LoadModelByName = async (Name: string, Socket?: Ws): Promise<void> => {
    await Options.LoadModel?.(SafeName(Name));
    const Msg = JSON.stringify(ModelMsg());
    if (Socket !== undefined) Socket.send(Msg);
    Publish(JSON.parse(Msg));
  };

  const Fetch = async (Req: Request): Promise<Response> => {
    const Url = new URL(Req.url);
    const Path = Url.pathname;

    if (Path === "/" || Path === "/index.html") return HtmlResponse(DashboardHtml, DashboardPageCsp());
    if (Path === "/chat" || Path === "/chat.html") {
      const Nonce = NewCspNonce();
      return HtmlResponse(RenderChatHtml(Nonce), ChatPageCsp(Nonce, Req.headers.get("host"), GuardPort, GuardHosts));
    }

    if (Path.startsWith("/api/")) {
      if (!IsLocalRequest(Req)) return Json({ error: "forbidden: the dashboard only answers local origins" }, 403);
      // JSON-only bodies: a cross-site <form> cannot send application/json without a CORS preflight.
      if (Req.method === "POST" && !IsJsonContentType(Req.headers.get("content-type"))) return Json({ error: "Content-Type must be application/json" }, 415);
    }

    if (Path === "/api/stats") return Json(await Store.Stats());
    if (Path === "/api/repos") return Json(await Store.RepoSummaries());
    if (Path === "/api/system") return Json(GetSystemInfo());
    if (Path === "/api/model") return Json(Options.GetModel?.() ?? null);
    if (Path === "/api/checkpoints") return Json((await Options.Checkpoints?.()) ?? []);
    if (Path === "/api/kinds") return Json((await Options.KindStats?.()) ?? []);
    if (Path === "/api/collection") return Json((await Options.Collection?.()) ?? []);
    if (Path === "/api/config") return Json({ learnEnabled: Learn !== undefined, trainEnabled: Options.Train !== undefined, chatEnabled: Options.Chat !== undefined, running: Job.Running });

    if (Path === "/api/documents") {
      const Source = Url.searchParams.get("source") ?? "";
      const Limit = ToNum(Url.searchParams.get("limit"), 500);
      const Docs = await Store.DocumentsBySource(Source, Limit);
      return Json(Docs.map((D) => ({ id: D.Id, path: D.Provenance.split("/").slice(-3).join("/"), tier: D.Tier, lang: D.Lang, bytes: D.Bytes, provenance: D.Provenance })));
    }
    if (Path === "/api/file") {
      const Doc = await Store.DocumentById(Url.searchParams.get("id") ?? "");
      if (Doc === null) return Json({ error: "not found" }, 404);
      return Json({ provenance: Doc.Provenance, lang: Doc.Lang, tier: Doc.Tier, bytes: Doc.Bytes, license: Doc.License, origin: Doc.Origin, reason: Doc.RejectReason, content: Doc.Content });
    }

    // ── Data browser (per-kind, paginated) — review + clean the collected corpus ──
    if (Path === "/api/browse") {
      if (Options.Browse === undefined) return Json({ error: "browse not wired" }, 501);
      const Kind = Url.searchParams.get("kind") ?? "code";
      if (!IsDataKind(Kind)) return Json({ error: "unknown kind" }, 400);
      const Page = Math.max(0, ToNum(Url.searchParams.get("page"), 0));
      const PageSize = Math.min(200, Math.max(1, ToNum(Url.searchParams.get("pageSize"), 50)));
      const Filter = ParseFilter({ tier: Url.searchParams.get("tier"), lang: Url.searchParams.get("lang"), license: Url.searchParams.get("license"), q: Url.searchParams.get("q") });
      const Result = await Options.Browse(Kind, Filter, Page * PageSize, PageSize);
      // Lightweight rows only (no content/embedding) so a page over a huge table stays small.
      const Rows = Result.Rows.map((D) => ({ id: D.Id, provenance: D.Provenance, tier: D.Tier, lang: D.Lang, license: D.License, bytes: D.Bytes, origin: D.Origin }));
      return Json({ Rows, Total: Result.Total, Page, PageSize });
    }
    if (Path === "/api/browse/facets") {
      if (Options.Facets === undefined) return Json({ error: "browse not wired" }, 501);
      const Kind = Url.searchParams.get("kind") ?? "code";
      if (!IsDataKind(Kind)) return Json({ error: "unknown kind" }, 400);
      const S = await Options.Facets(Kind);
      return Json({ Total: S.Total, Tiers: S.ByTier, Langs: Object.keys(S.ByLang).sort(), Licenses: Object.keys(S.ByLicense).sort() });
    }
    if (Path === "/api/browse/doc") {
      if (Options.DocContent === undefined) return Json({ error: "browse not wired" }, 501);
      const Kind = Url.searchParams.get("kind") ?? "code";
      if (!IsDataKind(Kind)) return Json({ error: "unknown kind" }, 400);
      const Doc = await Options.DocContent(Kind, Url.searchParams.get("id") ?? "");
      if (Doc === null) return Json({ error: "not found" }, 404);
      return Json({ provenance: Doc.Provenance, lang: Doc.Lang, tier: Doc.Tier, bytes: Doc.Bytes, license: Doc.License, origin: Doc.Origin, reason: Doc.RejectReason, content: Doc.Content });
    }
    if (Path === "/api/browse/delete" && Req.method === "POST") {
      if (Options.DeleteDoc === undefined) return Json({ error: "delete not wired" }, 501);
      const Body = (await Req.json().catch(() => ({}))) as { kind?: string; id?: string };
      if (!IsDataKind(Body.kind ?? "") || typeof Body.id !== "string" || Body.id === "") return Json({ error: "kind + id required" }, 400);
      const Deleted = await Options.DeleteDoc(Body.kind as DataKind, Body.id);
      return Json({ deleted: Deleted });
    }
    if (Path === "/api/browse/delete-matching" && Req.method === "POST") {
      if (Options.DeleteMatching === undefined) return Json({ error: "delete not wired" }, 501);
      const Body = (await Req.json().catch(() => ({}))) as Record<string, unknown>;
      const Kind = String(Body["kind"] ?? "");
      if (!IsDataKind(Kind)) return Json({ error: "kind required" }, 400);
      const Filter = ParseFilter(Body);
      // An empty filter wipes the WHOLE kind table — demand an explicit confirm for that case.
      if (Object.keys(Filter).length === 0 && Body["confirm"] !== true) return Json({ error: "empty filter deletes every document of this kind — send confirm: true" }, 400);
      const Deleted = await Options.DeleteMatching(Kind, Filter);
      return Json({ deleted: Deleted });
    }
    if (Path === "/api/checkpoint/delete" && Req.method === "POST") {
      if (Options.DeleteCheckpoint === undefined) return Json({ error: "checkpoint delete not wired" }, 501);
      const Body = (await Req.json().catch(() => ({}))) as { name?: string };
      if (typeof Body.name !== "string" || Body.name === "") return Json({ error: "name required" }, 400);
      await Options.DeleteCheckpoint(SafeName(Body.name));
      Publish({ type: "checkpoints", data: (await Options.Checkpoints?.()) ?? [] }); // refresh every client's model list
      return Json({ ok: true });
    }

    if (Path === "/api/chat/conversations") return Json(Options.Chat !== undefined ? await Options.Chat.ListConversations() : []);
    if (Path === "/api/chat/conversation") {
      const Id = Url.searchParams.get("id");
      if (!IsValidConvId(Id)) return Json({ error: "invalid conversation id" }, 400);
      return Json(Options.Chat !== undefined ? await Options.Chat.Messages(Id) : []);
    }
    if (Path === "/api/chat/delete" && Req.method === "POST") {
      const Body = (await Req.json().catch(() => ({}))) as { id?: unknown };
      if (!IsValidConvId(Body.id)) return Json({ error: "invalid conversation id" }, 400);
      if (Options.Chat !== undefined) await Options.Chat.Delete(Body.id);
      return Json({ ok: true });
    }

    if (Path === "/api/learn" && Req.method === "POST") {
      if (Learn === undefined) return Json({ error: "read-only dashboard (no learn runner wired)" }, 501);
      if (Job.Running) return Json({ error: "a learn run is already in progress" }, 409);
      const Body = (await Req.json().catch(() => ({}))) as Record<string, unknown>;
      StartLearn(ParseSettings(Body));
      return Json({ started: true }, 202);
    }

    if (Path === "/api/learn/stream") {
      let Listener: ((Event: LearnEvent) => void) | null = null;
      const Drop = (): void => {
        if (Listener !== null) {
          Job.Listeners.delete(Listener);
          Listener = null;
        }
      };
      const Stream = new ReadableStream<Uint8Array>({
        start: (Controller: ReadableStreamDefaultController<Uint8Array>): void => {
          const Encoder = new TextEncoder();
          let Closed = false;
          const End = (): void => {
            Closed = true;
            Drop();
            try {
              Controller.close();
            } catch {
              // already closed (idle timeout / client disconnect) — ignore
            }
          };
          const Send = (Event: LearnEvent): void => {
            if (Closed) return;
            try {
              Controller.enqueue(Encoder.encode(`data: ${JSON.stringify(Event)}\n\n`));
            } catch {
              End();
            }
          };
          for (const Event of Job.Events) Send(Event);
          if (!Job.Running) {
            End();
            return;
          }
          Listener = (Event: LearnEvent): void => {
            Send(Event);
            if (Event.kind === "done") End(); // only done is terminal (errors are non-fatal mid-run)
          };
          Job.Listeners.add(Listener);
        },
        cancel: Drop,
      });
      return new Response(Stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }

    return new Response("not found", { status: 404 });
  };

  // WebSocket: push snapshots on open, stream Learn progress, and stream chat token-by-token.
  const SendSnapshots = (Socket: Ws): void => {
    Socket.send(JSON.stringify({ type: "system", data: GetSystemInfo() }));
    Socket.send(JSON.stringify(ModelMsg()));
    void Options.Checkpoints?.()
      .then((C) => Socket.send(JSON.stringify({ type: "checkpoints", data: C })))
      .catch(() => undefined);
    void Store.Stats()
      .then((S) => Socket.send(JSON.stringify({ type: "stats", data: S })))
      .catch(() => undefined); // a transient store error must not become an unhandled rejection
    for (const Event of Job.Events) Socket.send(JSON.stringify({ type: "learn", event: Event }));
    for (const Event of TrainJob.Events) Socket.send(JSON.stringify({ type: "train", event: Event }));
  };

  const HandleChat = async (Socket: Ws, Msg: ChatMsg): Promise<void> => {
    const Chat = Options.Chat;
    if (!IsValidConvId(Msg.convId)) {
      // Never reflect a malformed id back (it is untrusted input, not a conversation).
      Socket.send(JSON.stringify({ type: "chat-error", convId: "", error: "invalid conversation id" }));
      return;
    }
    const ConvId = Msg.convId;
    if (Chat === undefined) {
      Socket.send(JSON.stringify({ type: "chat-error", convId: ConvId, error: "no model loaded — train a checkpoint first (see Checkpoints/)" }));
      return;
    }
    const Message = Msg.message;
    if (typeof Message !== "string" || Message.trim() === "") {
      Socket.send(JSON.stringify({ type: "chat-error", convId: ConvId, error: "empty message" }));
      return;
    }
    // Resource caps: one reply in flight per socket, MaxGlobalGenerations across the server — reject
    // (with a clear busy error) rather than queue, so a flood can never pile up CPU-bound work.
    if (Socket.data.Generating || ActiveGenerations >= MaxGlobalGenerations) {
      const Why = Socket.data.Generating ? "a reply is already generating on this connection" : `the server is already generating ${MaxGlobalGenerations} replies`;
      Socket.send(JSON.stringify({ type: "chat-error", convId: ConvId, code: "busy", error: `busy — ${Why}; try again when it finishes` }));
      return;
    }
    Socket.data.Generating = true;
    ActiveGenerations++;
    try {
      // Abort generation if the client goes away (readyState leaves OPEN=1) — don't burn CPU for nobody.
      const Opts = {
        Temperature: ToNum(Msg.temperature, DefaultChatTemperature),
        MaxTokens: Math.min(MaxNewTokens, Math.max(1, Math.floor(ToNum(Msg.maxTokens, DefaultChatMaxTokens)))), // server-side cap
        ShouldStop: (): boolean => Socket.readyState !== 1,
        OnTrace: (Lines: unknown): void => {
          Socket.send(JSON.stringify({ type: "chat-trace", convId: ConvId, lines: Lines })); // the visible reasoning steps
        },
        // Live reasoning: think text streams as it is generated (seg groups spans per agent step) and
        // each tool call surfaces when it executes — the user watches the model reason in real time.
        OnThink: (Delta: string, Seg: number): void => {
          Socket.send(JSON.stringify({ type: "chat-think-delta", convId: ConvId, delta: Delta, seg: Seg }));
        },
        OnToolStep: (Line: unknown): void => {
          Socket.send(JSON.stringify({ type: "chat-tool", convId: ConvId, line: Line }));
        },
      };
      await Chat.Turn(ConvId, Message, Opts, (Delta) => Socket.send(JSON.stringify({ type: "chat-delta", convId: ConvId, delta: Delta })));
      Socket.send(JSON.stringify({ type: "chat-done", convId: ConvId }));
    } catch (Caught) {
      Socket.send(JSON.stringify({ type: "chat-error", convId: ConvId, error: (Caught as Error).message }));
    } finally {
      Socket.data.Generating = false;
      ActiveGenerations--;
    }
  };

  const WebSocket = {
    open: (Socket: Ws): void => {
      Clients++;
      Socket.subscribe("all");
      SendSnapshots(Socket);
    },
    close: (): void => {
      Clients = Math.max(0, Clients - 1);
    },
    message: (Socket: Ws, Raw: string | Buffer): void => {
      let Msg: { type?: string; settings?: Record<string, unknown> } & Record<string, unknown>;
      try {
        Msg = JSON.parse(typeof Raw === "string" ? Raw : Raw.toString());
      } catch {
        return;
      }
      if (Msg.type === "learn") {
        if (!StartLearn(ParseSettings(Msg.settings ?? {}))) {
          Socket.send(JSON.stringify({ type: "learn", event: { kind: "error", message: Learn === undefined ? "no learn runner wired" : "a run is already in progress" } }));
        }
      } else if (Msg.type === "learn-stop") {
        StopLearn();
      } else if (Msg.type === "train") {
        if (!StartTrain(ParseTrainSettings(Msg.settings ?? {}))) {
          Socket.send(JSON.stringify({ type: "train", event: { kind: "train-error", message: Options.Train === undefined ? "training not available" : "a training run is already in progress" } }));
        }
      } else if (Msg.type === "train-stop") {
        StopTrain();
      } else if (Msg.type === "load-model") {
        void LoadModelByName(String(Msg["name"] ?? ""), Socket);
      } else if (Msg.type === "chat") {
        void HandleChat(Socket, Msg as ChatMsg);
      }
    },
  };

  const AttachServer = (S: Server<WsData>): void => {
    Bound = S;
    GuardPort = S.port ?? GuardPort;
    GuardHosts = BindHostAliases(S.hostname ?? "");
  };
  // The server's fetch: guard + upgrade /ws (a cross-site page must not hijack the socket), else HTTP.
  const Serve = (Req: Request, Upgrader: Pick<Server<WsData>, "upgrade">): Response | Promise<Response> | undefined => {
    if (new URL(Req.url).pathname === "/ws") {
      if (!IsLocalRequest(Req)) return new Response("forbidden: the dashboard only answers local origins", { status: 403 });
      if (Upgrader.upgrade(Req, { data: { Generating: false } })) return undefined;
      return new Response("websocket upgrade failed", { status: 400 });
    }
    return Fetch(Req);
  };
  const Snapshot = (): void => {
    if (Clients === 0) return; // nobody watching — skip the stats query entirely
    Publish({ type: "system", data: GetSystemInfo() });
    void Store.Stats()
      .then((S) => Publish({ type: "stats", data: S }))
      .catch(() => undefined); // transient store error must not crash the ticker
  };

  return { Fetch, Serve, WebSocket, AttachServer, Snapshot };
}

export function CreateDashboardHandler(Store: DocumentStore, Learn?: LearnFn, Options: DashboardOptions = {}): DashboardHandler {
  return CreateDashboardParts(Store, Learn, Options).Fetch;
}

export function StartDashboard(Store: DocumentStore, Port = DefaultDashboardPort, Learn?: LearnFn, Options: DashboardOptions = {}): Server<WsData> {
  const Parts = CreateDashboardParts(Store, Learn, Options);
  const Hostname = ResolveBindHost(); // loopback unless FOUNDRY_BIND says otherwise
  // idleTimeout 0 disables Bun's 10s request timeout, which would otherwise kill a long-lived SSE
  // stream mid-learn (a whole-repo ingest can take much longer than 10s).
  const Instance = Bun.serve<WsData>({
    hostname: Hostname,
    port: Port,
    idleTimeout: 0,
    fetch: (Req: Request, Serv: Server<WsData>): Response | Promise<Response> | undefined => Parts.Serve(Req, Serv),
    websocket: Parts.WebSocket,
  });
  Parts.AttachServer(Instance);
  setInterval(() => Parts.Snapshot(), SnapshotIntervalMs); // realtime system + stats push (only while clients are connected)
  console.log(`[dashboard] listening on ${Instance.url.href} (bind ${Instance.hostname ?? Hostname}; set ${BindEnvVar} to change)`);
  return Instance;
}
