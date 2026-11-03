// In-process TS inference server (Phase 6, C4: serving stays Node/Bun). An OpenAI-compatible surface
// over Bun.serve: POST /v1/chat/completions (JSON or real token-by-token SSE), GET /v1/models, GET
// /health. Bound to loopback by default.
//
// Prompt building never round-trips through a string re-encode for a special-token model: chat
// checkpoints render the messages straight to ids with the chat template (RenderChatToIds — user,
// system and client-supplied assistant text all base-encoded), so a message can't smuggle a control
// token. Plain base models keep the role-tagged transcript (RenderMessages).
//
// Generation goes through the one guarded ids-level entry point, GuardedGenerateIdsStream (Safety/):
// (1) EnforceInput on the rendered transcript before any compute, (2) the Limits caps on new tokens and
// context, (3) EnforceOutput on every delta BEFORE it is delivered plus a final full vet. The completion
// is computed from the generated IDS, never by string-prefix matching against the prompt. Generation is
// async and KV-cached (CachedGenerateAsync), so the event loop keeps serving other requests between
// tokens; a semaphore bounds concurrent generations.

import { randomUUID } from "node:crypto";
import type { Shahd } from "../Nn/Shahd.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";
import { DefaultSampling } from "../Sampling/Sampler.ts";
import type { SamplingOptions } from "../Sampling/Sampler.ts";
import { SeededRng } from "../Random/SeededRng.ts";
import { SafetyBlockedError } from "../Safety/SafetyPolicy.ts";
import { GuardedGenerateIdsStream, CapPromptIds } from "../Safety/GuardedGenerate.ts";
import type { TokenGenerator } from "../Safety/GuardedGenerate.ts";
import { RenderChatToIds, ChatTokens } from "../Sft/ChatTemplate.ts";
import type { ChatMessage, ChatRole } from "../Sft/ChatTemplate.ts";
import { RenderMessages } from "./RenderChat.ts";
import { ThinkRouter, SplitThinkingText } from "./ThinkRouter.ts";
import type { RenderableMessage } from "./RenderChat.ts";

export type ChatHandler = (Req: Request) => Promise<Response>;

// The generation seam (defined once, in Safety/): CachedGenerateAsync by default; tests inject a
// slow/stub generator with the same shape (prompt ids in, all ids out, OnToken per new id, ShouldStop
// polled each step). Re-exported so serving callers keep importing it from here.
export type { TokenGenerator };

export type ChatHandlerOptions = {
  MaxInFlight?: number; // concurrent generations allowed before answering 429 busy
  ModelId?: string; // the id reported by /v1/models and in responses
  Generate?: TokenGenerator;
};

export type ServeOptions = ChatHandlerOptions & { Hostname?: string; Port?: number };

export const DefaultMaxInFlight = 2; // concurrent generations before 429 busy (CPU inference is compute-bound)
const DefaultModelId = "shahd";
const DefaultMaxTokens = 128;
const DefaultTemperature = 0.8;
const MaxTemperature = 2; // the OpenAI-compatible range is [0, 2]
const MaxRequestBodyBytes = 1024 * 1024; // a chat request is text; 1 MB is far beyond any real prompt
const DefaultBindHost = "127.0.0.1"; // loopback: exposing the model is an explicit operator decision
const DefaultPort = 8080;
const MaxPort = 65535;
const BindEnvVar = "SHAHD_SERVE_BIND";
const PortEnvVar = "SHAHD_SERVE_PORT";
// Bun closes a connection idle for longer than idleTimeout (default 10 s) — a non-streaming CPU
// generation can easily exceed that before the first byte. 255 s is Bun's maximum.
const ServeIdleTimeoutSeconds = 255;

const PublicErrors = {
  InvalidRequest: "invalid request",
  TooLarge: "request too large",
  Busy: "busy",
  Blocked: "blocked by safety policy",
  Failed: "generation failed",
  NotFound: "not found",
  MethodNotAllowed: "method not allowed",
} as const;

const RoleMap: Record<string, ChatRole> = { system: "System", user: "User", tool: "User", assistant: "Assistant" };

type ChatRequest = { Messages: RenderableMessage[]; MaxTokens: number; Temperature: number; Stream: boolean };

// Validate the untrusted JSON body into a typed request, or null (=> 400). Never throws.
function ParseChatRequest(Value: unknown): ChatRequest | null {
  if (typeof Value !== "object" || Value === null || Array.isArray(Value)) return null;
  const Body = Value as Record<string, unknown>;
  const Messages = Body["messages"];
  if (!Array.isArray(Messages)) return null;
  const Parsed: RenderableMessage[] = [];
  for (const Item of Messages) {
    if (typeof Item !== "object" || Item === null) return null;
    const Role = (Item as Record<string, unknown>)["role"];
    const Content = (Item as Record<string, unknown>)["content"];
    if (typeof Role !== "string" || !Object.hasOwn(RoleMap, Role) || typeof Content !== "string") return null;
    Parsed.push({ role: Role, content: Content });
  }
  const MaxTokens = Body["max_tokens"] ?? DefaultMaxTokens;
  if (typeof MaxTokens !== "number" || !Number.isInteger(MaxTokens) || MaxTokens < 1) return null;
  const Temperature = Body["temperature"] ?? DefaultTemperature;
  if (typeof Temperature !== "number" || !Number.isFinite(Temperature) || Temperature < 0 || Temperature > MaxTemperature) return null;
  const Stream = Body["stream"] ?? false;
  if (typeof Stream !== "boolean") return null;
  return { Messages: Parsed, MaxTokens, Temperature, Stream };
}

// The prompt as ids (what the model sees), the text the safety policy scans, and the id that ends the
// assistant's turn (chat models only).
type PreparedPrompt = { PromptIds: number[]; ScanText: string; StopId: number | null };

function PreparePrompt(Messages: RenderableMessage[], Tok: Tokenizer): PreparedPrompt {
  const Transcript = RenderMessages(Messages);
  if (Tok instanceof SpecialTokenizer && Tok.TryId(ChatTokens.User) !== null) {
    const Chat: ChatMessage[] = Messages.map((M) => ({ Role: RoleMap[M.role] ?? "User", Content: M.content }));
    return { PromptIds: RenderChatToIds(Chat, Tok, true), ScanText: Transcript, StopId: Tok.TryId(ChatTokens.EndOfTurn) };
  }
  // A special tokenizer WITHOUT chat roles (e.g. FIM-only) still must not let message text become its
  // control tokens: base-encode the whole transcript. A plain tokenizer has no control tokens at all.
  const PromptIds = Tok instanceof SpecialTokenizer ? Tok.EncodeBase(Transcript) : Tok.Encode(Transcript);
  return { PromptIds, ScanText: Transcript, StopId: null };
}

type GenerationOutcome = { Text: string; PromptTokens: number; CompletionTokens: number; FinishReason: "stop" | "length" };

type GenerationRequest = {
  Model: Shahd;
  Tok: Tokenizer;
  Config: ResolvedConfig;
  Generate: TokenGenerator | undefined; // undefined = the guard's default (CachedGenerateAsync)
  Request: ChatRequest;
  Rng: SeededRng;
  OnDelta: (Delta: string) => void;
  ShouldStop: () => boolean;
};

// The guarded generation core shared by the JSON and SSE paths. Throws SafetyBlockedError on a blocked
// prompt/output (before the blocked text is delivered) and propagates any generation error.
async function RunGuardedGeneration(G: GenerationRequest): Promise<GenerationOutcome> {
  const Prepared = PreparePrompt(G.Request.Messages, G.Tok);
  const Sampling: SamplingOptions = { ...DefaultSampling, Temperature: G.Request.Temperature };
  let Text = "";
  const Ids = await GuardedGenerateIdsStream(
    G.Model,
    G.Tok,
    Prepared.PromptIds,
    G.Request.MaxTokens,
    Sampling,
    G.Rng,
    G.Config,
    (): void => undefined, // per-id hook unused: the HTTP surface delivers text deltas
    G.ShouldStop,
    {
      InputScanText: Prepared.ScanText,
      StopIds: Prepared.StopId === null ? undefined : new Set([Prepared.StopId]),
      OnDelta: (Delta: string): void => {
        Text += Delta; // the vetted deltas concatenate to exactly the decoded completion
        G.OnDelta(Delta);
      },
      Generator: G.Generate,
    },
  );
  // A natural stop leaves the end-of-turn id as the last completion id; it is not content.
  const HitStop = Prepared.StopId !== null && Ids.length > 0 && Ids[Ids.length - 1] === Prepared.StopId;
  return {
    Text,
    PromptTokens: CapPromptIds(Prepared.PromptIds, G.Config).length,
    CompletionTokens: HitStop ? Ids.length - 1 : Ids.length,
    FinishReason: HitStop ? "stop" : "length",
  };
}

// The scratchpad never ships as `content`: it goes out as `reasoning` (omitted when empty).
function AssistantMessage(Text: string): Record<string, string> {
  const Split = SplitThinkingText(Text);
  return Split.Reasoning.length > 0 ? { role: "assistant", content: Split.Content, reasoning: Split.Reasoning } : { role: "assistant", content: Split.Content };
}

function ErrorResponse(Status: number, Message: string, Headers?: Record<string, string>): Response {
  return Response.json({ error: Message }, { status: Status, headers: Headers });
}

// Map an internal failure to a fixed public message; the detail is logged server-side only.
function PublicFailure(Err: unknown, RequestId: string): { Status: number; Message: string } {
  if (Err instanceof SafetyBlockedError) {
    console.warn(`[serve] ${RequestId} blocked: ${Err.Categories.join(", ") || "none"}`);
    return { Status: 400, Message: PublicErrors.Blocked };
  }
  console.error(`[serve] ${RequestId} generation failed:`, Err);
  return { Status: 500, Message: PublicErrors.Failed };
}

export function CreateChatHandler(Model: Shahd, Tok: Tokenizer, Config: ResolvedConfig, Options: ChatHandlerOptions = {}): ChatHandler {
  const MaxInFlight = Math.max(1, Math.floor(Options.MaxInFlight ?? DefaultMaxInFlight));
  const ModelId = Options.ModelId ?? DefaultModelId;
  const Generate = Options.Generate;
  let Counter = 0;
  let InFlight = 0;

  const ModelsResponse = (): Response =>
    Response.json({ object: "list", data: [{ id: ModelId, object: "model", created: 0, owned_by: "shahd" }] });

  const Completions = async (Req: Request): Promise<Response> => {
    const DeclaredLength = Number(Req.headers.get("content-length") ?? "0");
    if (DeclaredLength > MaxRequestBodyBytes) return ErrorResponse(413, PublicErrors.TooLarge);
    const Raw = await Req.text().catch(() => null);
    if (Raw === null) return ErrorResponse(400, PublicErrors.InvalidRequest);
    if (Raw.length > MaxRequestBodyBytes) return ErrorResponse(413, PublicErrors.TooLarge);
    let Json: unknown;
    try {
      Json = JSON.parse(Raw);
    } catch {
      return ErrorResponse(400, PublicErrors.InvalidRequest);
    }
    const ChatReq = ParseChatRequest(Json);
    if (ChatReq === null) return ErrorResponse(400, PublicErrors.InvalidRequest);

    if (InFlight >= MaxInFlight) return ErrorResponse(429, PublicErrors.Busy, { "Retry-After": "1" });
    InFlight++;
    let Released = false;
    const Release = (): void => {
      if (!Released) {
        Released = true;
        InFlight--;
      }
    };
    const RequestId = `chatcmpl-${randomUUID()}`;
    const Created = Math.floor(Date.now() / 1000);
    const Rng = new SeededRng(Config.Training.Seed + Counter++);
    const Base = { Model, Tok, Config, Generate, Request: ChatReq, Rng };

    if (!ChatReq.Stream) {
      try {
        const Outcome = await RunGuardedGeneration({ ...Base, OnDelta: () => undefined, ShouldStop: () => Req.signal.aborted });
        return Response.json({
          id: RequestId,
          object: "chat.completion",
          created: Created,
          model: ModelId,
          choices: [{ index: 0, message: AssistantMessage(Outcome.Text), finish_reason: Outcome.FinishReason }],
          usage: { prompt_tokens: Outcome.PromptTokens, completion_tokens: Outcome.CompletionTokens, total_tokens: Outcome.PromptTokens + Outcome.CompletionTokens },
        });
      } catch (Err) {
        const Failure = PublicFailure(Err, RequestId);
        return ErrorResponse(Failure.Status, Failure.Message);
      } finally {
        Release();
      }
    }

    // SSE: one event per vetted delta as tokens are produced (the generator yields between tokens, so
    // each enqueue reaches the client), then the finish chunk and [DONE]. A client disconnect cancels
    // the stream and stops generation at the next token.
    const Encoder = new TextEncoder();
    let Cancelled = false;
    const Chunk = (Delta: Record<string, string>, FinishReason: string | null): Record<string, unknown> => ({
      id: RequestId,
      object: "chat.completion.chunk",
      created: Created,
      model: ModelId,
      choices: [{ index: 0, delta: Delta, finish_reason: FinishReason }],
    });
    const Stream = new ReadableStream<Uint8Array>({
      start: (Controller: ReadableStreamDefaultController<Uint8Array>): void => {
        const Send = (Payload: unknown): void => {
          if (Cancelled) return;
          try {
            Controller.enqueue(Encoder.encode(`data: ${typeof Payload === "string" ? Payload : JSON.stringify(Payload)}\n\n`));
          } catch {
            Cancelled = true; // the consumer went away
          }
        };
        const Run = async (): Promise<void> => {
          try {
            Send(Chunk({ role: "assistant" }, null));
            const Router = new ThinkRouter();
            const SendRouted = (Routed: { Content: string; Reasoning: string }): void => {
              if (Routed.Reasoning.length > 0) Send(Chunk({ reasoning: Routed.Reasoning }, null));
              if (Routed.Content.length > 0) Send(Chunk({ content: Routed.Content }, null));
            };
            const Outcome = await RunGuardedGeneration({
              ...Base,
              OnDelta: (Delta) => SendRouted(Router.Push(Delta)),
              ShouldStop: () => Cancelled || Req.signal.aborted,
            });
            SendRouted(Router.Flush());
            Send(Chunk({}, Outcome.FinishReason));
          } catch (Err) {
            Send({ error: PublicFailure(Err, RequestId).Message });
          } finally {
            Release();
            Send("[DONE]");
            try {
              Controller.close();
            } catch {
              // already closed by a cancel
            }
          }
        };
        void Run();
      },
      cancel: (): void => {
        Cancelled = true;
      },
    });
    return new Response(Stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
  };

  return async (Req: Request): Promise<Response> => {
    const Path = new URL(Req.url).pathname;
    if (Path === "/health") return Req.method === "GET" ? new Response("ok") : ErrorResponse(405, PublicErrors.MethodNotAllowed);
    if (Path === "/v1/models") return Req.method === "GET" ? ModelsResponse() : ErrorResponse(405, PublicErrors.MethodNotAllowed);
    if (Path === "/v1/chat/completions") {
      if (Req.method !== "POST") return ErrorResponse(405, PublicErrors.MethodNotAllowed);
      try {
        return await Completions(Req);
      } catch (Err) {
        console.error("[serve] unhandled request failure:", Err);
        return ErrorResponse(500, PublicErrors.Failed);
      }
    }
    return ErrorResponse(404, PublicErrors.NotFound);
  };
}

/** Bind address from the environment: SHAHD_SERVE_BIND (default 127.0.0.1) and SHAHD_SERVE_PORT
 *  (default 8080; 0 = an ephemeral port). Throws on a malformed port rather than binding somewhere odd. */
export function ResolveServeAddress(Env: Record<string, string | undefined> = process.env): { Hostname: string; Port: number } {
  const Hostname = Env[BindEnvVar]?.trim() || DefaultBindHost;
  const RawPort = Env[PortEnvVar]?.trim();
  if (RawPort === undefined || RawPort === "") return { Hostname, Port: DefaultPort };
  const Port = Number(RawPort);
  if (!Number.isInteger(Port) || Port < 0 || Port > MaxPort) throw new Error(`${PortEnvVar} must be an integer in [0, ${MaxPort}], got '${RawPort}'`);
  return { Hostname, Port };
}

export function StartServer(Model: Shahd, Tok: Tokenizer, Config: ResolvedConfig, Options: ServeOptions = {}): ReturnType<typeof Bun.serve> {
  const Address = ResolveServeAddress();
  return Bun.serve({
    hostname: Options.Hostname ?? Address.Hostname,
    port: Options.Port ?? Address.Port,
    maxRequestBodySize: MaxRequestBodyBytes,
    idleTimeout: ServeIdleTimeoutSeconds,
    fetch: CreateChatHandler(Model, Tok, Config, Options),
  });
}
