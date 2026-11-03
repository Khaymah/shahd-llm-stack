// The SFT teacher over HTTP: an OpenAI-compatible /v1/chat/completions endpoint (llama.cpp's
// llama-server, Ollama, vLLM …) serving a LOCAL open-weights model — never a commercial API (Plan §8).
// Lives in the Foundry (the layer that owns network access); the Brain only sees the SftTeacher
// contract. Each request has a hard timeout and is retried with exponential backoff on transient
// failures (429 / 5xx / network / timeout) via the shared FetchWithBackoff. JSON mode is requested as
// response_format {type: "json_object"}; a server that rejects that field with HTTP 400 is asked again
// once without it and JSON mode stays off for the rest of the run (the generators extract JSON from
// plain text anyway — Brain/Sft/Synthetic/TeacherJson.ts).

import { z } from "zod";
import type { SftTeacher, TeacherRequest } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { TeacherError } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import type { BackoffOptions } from "./HttpBackoff.ts";
import { FetchWithBackoff, HttpError, IsTransient, RetryAfterToMs } from "./HttpBackoff.ts";

/** Where llama-server listens in the synthesize-sft workflow (and the local default). */
export const DefaultTeacherUrl = "http://127.0.0.1:8081";
/** A completion on a 4-vCPU runner (prompt processing + up to ~1k generated tokens) stays well below this. */
export const DefaultTeacherTimeoutMs = 300_000;

export type TeacherFetch = (Url: string, Init: RequestInit) => Promise<Response>;

export type TeacherClientOptions = {
  Url: string; // base URL, e.g. http://127.0.0.1:8081
  Model: string;
  TimeoutMs?: number;
  Backoff?: BackoffOptions;
  Fetch?: TeacherFetch; // injected in tests
};

const CompletionSchema = z.object({
  choices: z.array(z.object({ message: z.object({ content: z.string().nullable() }) })).min(1),
});

const ModelsSchema = z.object({ data: z.array(z.object({ id: z.string().min(1) })).min(1) });

const HttpBadRequest = 400;

function Endpoint(Base: string, Path: string): string {
  return `${Base.replace(/\/+$/, "")}${Path}`;
}

/** The model id to record in provenance: the explicit one, else the first id the server lists. */
export async function ResolveTeacherModel(Url: string, Explicit: string | undefined, Fetch: TeacherFetch = fetch): Promise<string> {
  if (Explicit !== undefined && Explicit.trim() !== "") return Explicit.trim();
  const Reply = await Fetch(Endpoint(Url, "/v1/models"), { method: "GET", signal: AbortSignal.timeout(DefaultTeacherTimeoutMs) });
  if (!Reply.ok) throw new TeacherError(`teacher /v1/models: HTTP ${Reply.status} (set SHAHD_TEACHER_MODEL)`);
  const Parsed = ModelsSchema.safeParse(await Reply.json());
  if (!Parsed.success) throw new TeacherError("teacher /v1/models: unexpected response (set SHAHD_TEACHER_MODEL)");
  return Parsed.data.data[0].id.split("/").pop()!.replace(/\.gguf$/i, "");
}

/** An SftTeacher backed by an OpenAI-compatible chat-completions endpoint. */
export function CreateOpenAiTeacher(Options: TeacherClientOptions): SftTeacher {
  const Fetch = Options.Fetch ?? fetch;
  const TimeoutMs = Options.TimeoutMs ?? DefaultTeacherTimeoutMs;
  let JsonSupported = true;

  async function Once(Request: TeacherRequest, Json: boolean): Promise<Response> {
    const Body: Record<string, unknown> = {
      model: Options.Model,
      messages: Request.Messages,
      temperature: Request.Temperature,
      max_tokens: Request.MaxTokens,
      seed: Request.Seed,
      stream: false,
    };
    if (Json) Body["response_format"] = { type: "json_object" };
    return Fetch(Endpoint(Options.Url, "/v1/chat/completions"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(Body),
      signal: AbortSignal.timeout(TimeoutMs),
    });
  }

  async function Attempt(Request: TeacherRequest): Promise<string> {
    const UseJson = Request.Json && JsonSupported;
    let Reply = await Once(Request, UseJson);
    if (Reply.status === HttpBadRequest && UseJson) {
      JsonSupported = false; // the server does not take response_format: plain text from now on
      Reply = await Once(Request, false);
    }
    if (!Reply.ok) throw new HttpError(Reply.status, RetryAfterToMs(Reply.headers.get("retry-after")));
    const Parsed = CompletionSchema.safeParse(await Reply.json());
    if (!Parsed.success) throw new TeacherError(`teacher: unexpected completion shape (${Parsed.error.issues[0]?.message ?? "invalid"})`);
    return Parsed.data.choices[0].message.content ?? "";
  }

  return {
    Model: Options.Model,
    Complete: async (Request: TeacherRequest): Promise<string> => {
      try {
        // A malformed completion is the server's answer, not a transient fault: never retried.
        return await FetchWithBackoff(() => Attempt(Request), Options.Backoff, (Caught) => !(Caught instanceof TeacherError) && IsTransient(Caught));
      } catch (Caught) {
        if (Caught instanceof TeacherError) throw Caught;
        throw new TeacherError(`teacher ${Options.Url}: ${(Caught as Error).message}`);
      }
    },
  };
}
