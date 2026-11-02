import { test, expect } from "bun:test";
import { InMemoryDocumentStore, IngestDocuments, CreateDashboardHandler, CreateDashboardParts, ChatService, InMemoryChatStore } from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, LearnFn, ChatOpts, DocumentFilter, DashboardOptions, LearnEvent } from "../Foundry/FoundryBarrel.ts";

const JsonHeaders = { "Content-Type": "application/json" };

async function SeededStore() {
  const Store = new InMemoryDocumentStore();
  const Inputs: SourceInput[] = [
    { Source: "acme/good", License: "MIT", Lang: "ts", Content: "export function add(a, b) {\n  return a + b;\n}\n", Provenance: "https://github.com/acme/good/blob/main/src/Add.ts", Origin: "web-permissive" },
    { Source: "acme/good", License: "MIT", Lang: "ts", Content: "export const pi = 3.14159;\n", Provenance: "https://github.com/acme/good/blob/main/src/Pi.ts", Origin: "web-permissive" },
    { Source: "bad/repo", License: "GPL-3.0", Lang: "c", Content: "int main(void) {\n  return 0;\n}\n", Provenance: "https://github.com/bad/repo/blob/main/m.c", Origin: "web-permissive" },
  ];
  await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  return Store;
}

test("dashboard serves the control-panel page", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Res = await Handler(new Request("http://x/"));
  expect(Res.headers.get("Content-Type")).toContain("text/html");
  expect(await Res.text()).toContain("Control panel"); // the SPA rebuild renamed the page title
});

test("/api/stats aggregates tiers/langs efficiently", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Stats = (await (await Handler(new Request("http://x/api/stats"))).json()) as { Total: number; ByTier: Record<string, number>; ByLang: Record<string, number> };
  expect(Stats.Total).toBe(3);
  expect(Stats.ByTier.Filtered).toBe(2);
  expect(Stats.ByTier.Rejected).toBe(1);
  expect(Stats.ByLang.ts).toBe(2);
});

test("/api/repos + /api/documents power the per-repo accordion", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Repos = (await (await Handler(new Request("http://x/api/repos"))).json()) as { Source: string; Files: number }[];
  expect(Repos[0].Source).toBe("acme/good"); // most files first
  expect(Repos[0].Files).toBe(2);
  const Docs = (await (await Handler(new Request("http://x/api/documents?source=acme/good"))).json()) as { path: string; tier: string }[];
  expect(Docs.length).toBe(2);
  expect(Docs[0].path).toContain("Add.ts");
});

test("POST /api/learn starts a run and SSE streams its progress", async () => {
  const MockLearn: LearnFn = async (_Settings, OnEvent) => {
    OnEvent({ kind: "repo", repo: "acme/new", level: "high", files: 7, bytes: 2000, ingested: true, reason: null });
    OnEvent({ kind: "done", ingested: 7 });
  };
  const Handler = CreateDashboardHandler(await SeededStore(), MockLearn);
  const Post = await Handler(new Request("http://x/api/learn", { method: "POST", headers: JsonHeaders, body: JSON.stringify({ Source: "github", Query: "q" }) }));
  expect(Post.status).toBe(202);
  const Stream = await Handler(new Request("http://x/api/learn/stream"));
  const Text = await Stream.text();
  expect(Text).toContain("acme/new");
  expect(Text).toContain('"kind":"done"');
});

test("dashboard is read-only when no Learn runner is wired", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Config = (await (await Handler(new Request("http://x/api/config"))).json()) as { learnEnabled: boolean };
  expect(Config.learnEnabled).toBe(false);
  const Post = await Handler(new Request("http://x/api/learn", { method: "POST", headers: JsonHeaders, body: "{}" }));
  expect(Post.status).toBe(501);
});

// ── security hardening (local-only guard, CSRF, XSS, resource caps) ──

type Parts = ReturnType<typeof CreateDashboardParts>;
type Socket = Parameters<Parts["WebSocket"]["open"]>[0];
type SentMessage = { type: string; convId?: string; error?: string; code?: string; event?: LearnEvent };

// A stand-in for Bun's ServerWebSocket: records what the dashboard sends.
function FakeSocket(): { Socket: Socket; Sent: SentMessage[] } {
  const Sent: SentMessage[] = [];
  const Fake = { data: { Generating: false }, readyState: 1, send: (Text: string): void => void Sent.push(JSON.parse(Text) as SentMessage), subscribe: (): void => undefined };
  return { Socket: Fake as unknown as Socket, Sent };
}

const Settle = (): Promise<void> => new Promise((Resolve) => setTimeout(Resolve, 5));

// A chat service whose generation is controlled by the test (Release resolves every pending turn).
function GatedChat(): { Chat: ChatService; Seen: ChatOpts[]; Release: () => void } {
  const Seen: ChatOpts[] = [];
  const Waiters: (() => void)[] = [];
  const Chat = new ChatService(new InMemoryChatStore(), async (_Messages, Opts) => {
    Seen.push(Opts);
    await new Promise<void>((Resolve) => Waiters.push(Resolve));
    return "ok";
  });
  return { Chat, Seen, Release: (): void => Waiters.splice(0).forEach((Resolve) => Resolve()) };
}

function ChatMessageFor(ConvId: string, MaxTokens?: number): string {
  return JSON.stringify({ type: "chat", convId: ConvId, message: "hi", temperature: 0.5, maxTokens: MaxTokens });
}

test("WS upgrade from a foreign Origin is refused with 403 (cross-site WebSocket hijack)", async () => {
  const Parts = CreateDashboardParts(await SeededStore());
  let Upgrades = 0;
  const Upgrader = { upgrade: (): boolean => ++Upgrades > 0 };
  const Evil = Parts.Serve(new Request("http://localhost:8090/ws", { headers: { Origin: "https://evil.example" } }), Upgrader) as Response;
  expect(Evil.status).toBe(403);
  expect(Upgrades).toBe(0);
  const Local = Parts.Serve(new Request("http://localhost:8090/ws", { headers: { Origin: "http://localhost:8090" } }), Upgrader);
  expect(Local).toBeUndefined(); // upgraded
  expect(Upgrades).toBe(1);
});

test("/api/* refuses a foreign Origin or a rebound Host", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  expect((await Handler(new Request("http://x/api/stats", { headers: { Origin: "https://evil.example" } }))).status).toBe(403);
  expect((await Handler(new Request("http://x/api/stats", { headers: { Origin: "null" } }))).status).toBe(403);
  expect((await Handler(new Request("http://x/api/stats", { headers: { Host: "evil.example:8090" } }))).status).toBe(403);
  expect((await Handler(new Request("http://x/api/stats", { headers: { Origin: "http://127.0.0.1:8090", Host: "127.0.0.1:8090" } }))).status).toBe(200);
});

test("POST with a non-JSON Content-Type is refused with 415 (form CSRF)", async () => {
  const Handler = CreateDashboardHandler(await SeededStore(), async () => undefined);
  const Res = await Handler(new Request("http://x/api/learn", { method: "POST", headers: { "Content-Type": "text/plain" }, body: "{}" }));
  expect(Res.status).toBe(415);
  const NoType = await Handler(new Request("http://x/api/chat/delete", { method: "POST", body: new Uint8Array([123, 125]) }));
  expect(NoType.status).toBe(415);
});

test("delete-matching with an empty filter requires confirm: true", async () => {
  const Calls: DocumentFilter[] = [];
  const Options: DashboardOptions = { DeleteMatching: async (_Kind, Filter) => (Calls.push(Filter), 3) };
  const Handler = CreateDashboardHandler(await SeededStore(), undefined, Options);
  const Post = (Body: unknown): Promise<Response> => Handler(new Request("http://x/api/browse/delete-matching", { method: "POST", headers: JsonHeaders, body: JSON.stringify(Body) }));
  expect((await Post({ kind: "code" })).status).toBe(400);
  expect((await Post({ kind: "code", tier: "", q: "" })).status).toBe(400); // blank fields are still an empty filter
  expect(Calls.length).toBe(0);
  expect((await Post({ kind: "code", confirm: true })).status).toBe(200);
  expect((await Post({ kind: "code", tier: "Rejected" })).status).toBe(200); // a real filter needs no confirm
  expect(Calls).toEqual([{}, { Tier: "Rejected" }]);
});

test("a hostile convId is rejected on the WebSocket and the REST endpoints", async () => {
  const Gate = GatedChat();
  const Parts = CreateDashboardParts(await SeededStore(), undefined, { Chat: Gate.Chat });
  const { Socket, Sent } = FakeSocket();
  const Hostile = '&quot;);alert(1)//';
  Parts.WebSocket.message(Socket, ChatMessageFor(Hostile));
  await Settle();
  expect(Sent).toEqual([{ type: "chat-error", convId: "", error: "invalid conversation id" }]);
  expect(Gate.Seen.length).toBe(0); // never reached the model
  expect((await Parts.Fetch(new Request("http://x/api/chat/conversation?id=" + encodeURIComponent(Hostile)))).status).toBe(400);
  const Del = await Parts.Fetch(new Request("http://x/api/chat/delete", { method: "POST", headers: JsonHeaders, body: JSON.stringify({ id: Hostile }) }));
  expect(Del.status).toBe(400);
});

test("chat maxTokens is clamped server-side and generations are capped per socket + globally", async () => {
  const Gate = GatedChat();
  const Parts = CreateDashboardParts(await SeededStore(), undefined, { Chat: Gate.Chat, MaxNewTokens: 64 });
  const A = FakeSocket();
  const B = FakeSocket();
  const C = FakeSocket();
  Parts.WebSocket.message(A.Socket, ChatMessageFor("conv-a", 999999));
  await Settle();
  expect(Gate.Seen[0]?.MaxTokens).toBe(64);
  Parts.WebSocket.message(A.Socket, ChatMessageFor("conv-a2")); // same socket, one already in flight
  Parts.WebSocket.message(B.Socket, ChatMessageFor("conv-b")); // second global slot
  Parts.WebSocket.message(C.Socket, ChatMessageFor("conv-c")); // over the global cap
  await Settle();
  expect(A.Sent.filter((M) => M.code === "busy").length).toBe(1);
  expect(B.Sent.some((M) => M.code === "busy")).toBe(false);
  expect(C.Sent.filter((M) => M.code === "busy").length).toBe(1);
  Gate.Release();
  await Settle();
  expect(A.Sent.some((M) => M.type === "chat-done" && M.convId === "conv-a")).toBe(true);
  Parts.WebSocket.message(C.Socket, ChatMessageFor("conv-c")); // slots freed: accepted now
  await Settle();
  expect(Gate.Seen.length).toBe(3);
  Gate.Release();
  await Settle();
});

test("a mid-run learn error is not terminal; only done ends the run", async () => {
  let Finish: () => void = () => undefined;
  const Learn: LearnFn = async (_Settings, OnEvent) => {
    OnEvent({ kind: "error", message: "github: rate limited" }); // non-fatal provider failure
    await new Promise<void>((Resolve) => (Finish = Resolve));
    OnEvent({ kind: "done", ingested: 1 });
  };
  const Handler = CreateDashboardHandler(await SeededStore(), Learn);
  const Start = (): Promise<Response> => Handler(new Request("http://x/api/learn", { method: "POST", headers: JsonHeaders, body: "{}" }));
  expect((await Start()).status).toBe(202);
  expect((await Start()).status).toBe(409); // still running after the error
  Finish();
  await Settle();
  expect((await Start()).status).toBe(202);
});

test("a learn runner that throws still closes the run with done", async () => {
  const Handler = CreateDashboardHandler(await SeededStore(), async () => {
    throw new Error("boom");
  });
  await Handler(new Request("http://x/api/learn", { method: "POST", headers: JsonHeaders, body: "{}" }));
  await Settle();
  const Text = await (await Handler(new Request("http://x/api/learn/stream"))).text();
  expect(Text).toContain('"kind":"error"');
  expect(Text).toContain('"kind":"done"');
});

test("the learn replay buffer is bounded and keeps the start event", async () => {
  const Learn: LearnFn = async (_Settings, OnEvent) => {
    for (let I = 0; I < 800; I++) OnEvent({ kind: "repo", repo: "r" + I, level: "high", files: 1, bytes: 1, ingested: true, reason: null });
    OnEvent({ kind: "done", ingested: 800 });
  };
  const Handler = CreateDashboardHandler(await SeededStore(), Learn);
  await Handler(new Request("http://x/api/learn", { method: "POST", headers: JsonHeaders, body: "{}" }));
  const Lines = (await (await Handler(new Request("http://x/api/learn/stream"))).text()).split("\n\n").filter((L) => L.startsWith("data: "));
  expect(Lines.length).toBe(500);
  expect(Lines[0]).toContain('"kind":"start"');
  expect(Lines[Lines.length - 1]).toContain('"kind":"done"');
});

test("background stats snapshots only run while a client is connected", async () => {
  const Store = await SeededStore();
  let StatsCalls = 0;
  const Original = Store.Stats.bind(Store);
  Store.Stats = (): ReturnType<typeof Store.Stats> => (StatsCalls++, Original());
  const Parts = CreateDashboardParts(Store);
  Parts.Snapshot();
  expect(StatsCalls).toBe(0);
  const { Socket } = FakeSocket();
  Parts.WebSocket.open(Socket); // initial snapshot for the new client
  Parts.Snapshot();
  expect(StatsCalls).toBe(2);
  Parts.WebSocket.close();
  Parts.Snapshot();
  expect(StatsCalls).toBe(2);
});

test("the chat page ships a nonce CSP and no inline handlers", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Res = await Handler(new Request("http://localhost:8090/chat", { headers: { Host: "localhost:8090" } }));
  const Csp = Res.headers.get("Content-Security-Policy") ?? "";
  const Nonce = /'nonce-([^']+)'/.exec(Csp)?.[1] ?? "";
  expect(Nonce.length).toBeGreaterThan(10);
  const Html = await Res.text();
  expect(Html).toContain('<script nonce="' + Nonce + '">');
  expect(Html).not.toMatch(/\son[a-z]+=/);
  expect(Csp).toContain("ws://localhost:8090");
  const Panel = await Handler(new Request("http://x/"));
  expect(Panel.headers.get("Content-Security-Policy")).toContain("frame-ancestors 'none'");
});
