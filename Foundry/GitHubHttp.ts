// Shared GitHub HTTP fetchers (rule #4: one home for auth + fetch). Both the file-sampling provider
// and the whole-repo provider use these. Injected in tests; the real fetch (with an optional token)
// by default. JSON for search/tree/contents; bytes for the repo tarball.

export type HttpJson = (Url: string, Signal?: AbortSignal) => Promise<unknown>;
export type FetchBytes = (Url: string, Signal?: AbortSignal) => Promise<Uint8Array>;

function AuthHeaders(Token: string | undefined, Accept: string): Record<string, string> {
  const Headers: Record<string, string> = { "User-Agent": "shahd-foundry", Accept };
  if (Token !== undefined) Headers["Authorization"] = `Bearer ${Token}`;
  return Headers;
}

// Rate limits: a spec of hundreds of repos spends ~2 API calls per repo, past the hourly budget of a
// CI token. A 403/429 that GitHub marks as rate limiting (x-ratelimit-remaining: 0, or Retry-After) is
// waited out and retried instead of failing the repo — up to MaxRateLimitWaitMs per wait.
export const MaxRateLimitWaitMs = 65 * 60 * 1000;
const RateLimitRetries = 3;
const RateLimitSlackMs = 5000;
const MsPerSecond = 1000;

/** How long to wait before retrying a response, or null when it is not a rate-limit response. */
export function RateLimitWaitMs(Status: number, Headers: Pick<globalThis.Headers, "get">, NowMs: number): number | null {
  if (Status !== 403 && Status !== 429) return null;
  const RetryAfter = Number(Headers.get("retry-after") ?? "");
  if (Number.isFinite(RetryAfter) && RetryAfter > 0) return Math.min(RetryAfter * MsPerSecond, MaxRateLimitWaitMs);
  if (Headers.get("x-ratelimit-remaining") !== "0") return null;
  const ResetAt = Number(Headers.get("x-ratelimit-reset") ?? "") * MsPerSecond;
  if (!Number.isFinite(ResetAt) || ResetAt <= 0) return null;
  return Math.min(Math.max(ResetAt - NowMs, 0) + RateLimitSlackMs, MaxRateLimitWaitMs);
}

/** fetch() that waits out GitHub rate limiting (see RateLimitWaitMs). */
async function FetchWithRateLimit(Url: string, Init: { headers: Record<string, string>; signal?: AbortSignal }): Promise<Response> {
  for (let Attempt = 0; ; Attempt++) {
    const Response = await fetch(Url, Init);
    const WaitMs = Attempt < RateLimitRetries ? RateLimitWaitMs(Response.status, Response.headers, Date.now()) : null;
    if (WaitMs === null) return Response;
    console.warn(`[github] rate limited on ${Url} — waiting ${Math.round(WaitMs / MsPerSecond)}s`);
    await Response.body?.cancel();
    await new Promise<void>((Resolve, Reject) => {
      const Timer = setTimeout(Resolve, WaitMs);
      Init.signal?.addEventListener("abort", () => {
        clearTimeout(Timer);
        Reject(new Error("aborted while waiting for the GitHub rate limit"));
      }, { once: true });
    });
  }
}

export function DefaultGitHubJson(Token?: string): HttpJson {
  return async (Url: string, Signal?: AbortSignal): Promise<unknown> => {
    const Response = await FetchWithRateLimit(Url, { headers: AuthHeaders(Token, "application/vnd.github+json"), signal: Signal });
    if (!Response.ok) throw new Error(`GitHub API ${Response.status}`);
    return Response.json();
  };
}

// Hard ceiling on RAW (compressed) tarball bytes buffered into memory. A repo whose declared or
// streamed size exceeds this is rejected BEFORE it can exhaust the single Bun process (which also
// serves the dashboard) — the per-file MaxBytes/MaxFiles caps in RepoArchive only apply AFTER full
// download + decompression, so they are no defence against an oversized/adversarial tarball. This
// bounds the compressed transfer; a gzip bomb (tiny compressed, huge decompressed) is a residual
// risk that would need streaming decompression with an output cap (nanotar exposes no such hook).
export const MaxTarballBytes = 256_000_000;

export function DefaultGitHubBytes(Token?: string, MaxBytes: number = MaxTarballBytes): FetchBytes {
  return async (Url: string, Signal?: AbortSignal): Promise<Uint8Array> => {
    const Response = await FetchWithRateLimit(Url, { headers: AuthHeaders(Token, "application/vnd.github+json"), signal: Signal });
    if (!Response.ok) throw new Error(`GitHub API ${Response.status}`);
    const Declared = Number(Response.headers.get("content-length") ?? "");
    if (Number.isFinite(Declared) && Declared > MaxBytes) {
      throw new Error(`GitHub tarball too large: ${Declared} bytes > cap ${MaxBytes}`);
    }
    // Stream with a running byte cap so a missing/lying Content-Length can't smuggle an unbounded
    // body past the check above.
    const Body = Response.body;
    if (Body === null) return new Uint8Array(await Response.arrayBuffer());
    const Reader = Body.getReader();
    const Chunks: Uint8Array[] = [];
    let Total = 0;
    for (;;) {
      const { done: Done, value: Value } = await Reader.read();
      if (Done) break;
      if (Value !== undefined) {
        Total += Value.byteLength;
        if (Total > MaxBytes) {
          await Reader.cancel();
          throw new Error(`GitHub tarball exceeded cap ${MaxBytes} bytes mid-stream`);
        }
        Chunks.push(Value);
      }
    }
    const Out = new Uint8Array(Total);
    let Offset = 0;
    for (const Chunk of Chunks) {
      Out.set(Chunk, Offset);
      Offset += Chunk.byteLength;
    }
    return Out;
  };
}

// Fetch a repo's LICENSE file (raw text) via GET /repos/{owner}/{repo}/license. Returns the SPDX id
// GitHub's own matcher assigned (often "NOASSERTION" — GitHub can't classify a deviant file) plus the
// decoded license text, which is what a stricter local matcher (SpdxDetector) then classifies. Null
// when the repo has no license file (404) or on any API error — the caller treats that as unresolved.
export type RepoLicense = { Spdx: string; Text: string };
export async function FetchRepoLicense(FullName: string, Token?: string): Promise<RepoLicense | null> {
  const Response = await FetchWithRateLimit(`https://api.github.com/repos/${FullName}/license`, { headers: AuthHeaders(Token, "application/vnd.github+json") });
  if (!Response.ok) return null;
  const Body = (await Response.json()) as { license?: { spdx_id?: string }; content?: string; encoding?: string };
  const Text = Body.content !== undefined && Body.encoding === "base64" ? Buffer.from(Body.content, "base64").toString("utf8") : "";
  return { Spdx: Body.license?.spdx_id ?? "NOASSERTION", Text };
}
