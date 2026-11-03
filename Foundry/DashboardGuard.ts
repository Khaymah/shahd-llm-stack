// Local-only request guard for the Foundry dashboard. The dashboard is a LOCAL developer tool with no
// login by design, so its security boundary is "only this machine may drive it":
//   1. bind to loopback by default (FOUNDRY_BIND overrides, e.g. 0.0.0.0 inside a container);
//   2. refuse any browser request whose Origin is not a loopback origin — a malicious web page open in
//      the same browser can otherwise POST to the API or hijack the WebSocket (cross-site WS);
//   3. refuse any request whose Host is not a loopback name — that is what defeats DNS rebinding
//      (an attacker domain re-resolved to 127.0.0.1 still sends its own name in Host).
// Plus small pure validators shared by the HTTP and WebSocket handlers (JSON content type, chat ids)
// and the page Content-Security-Policy. Everything here is pure so it is unit-testable in isolation.

import { randomBytes } from "node:crypto";

export const DefaultBindHost = "127.0.0.1"; // loopback only — nothing on the LAN can reach the panel
export const DefaultDashboardPort = 8090; // the guard's port until a real server is attached
export const BindEnvVar = "FOUNDRY_BIND";

const LoopbackNames: ReadonlySet<string> = new Set(["localhost", "127.0.0.1", "[::1]"]);
const WildcardBinds: ReadonlySet<string> = new Set(["", "0.0.0.0", "::", "[::]"]);
const AuthorityPattern = /^(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(?::(\d{1,5}))?$/; // name or [ipv6], optional :port
const ConvIdPattern = /^[A-Za-z0-9_-]{1,64}$/; // covers hex ids, UUIDs, and the client's "c-<hex>-<hex>" fallback
const CspNonceBytes = 16; // 128 bits of entropy per response

type Authority = { Name: string; Port: number | null };

function ParseAuthority(Value: string): Authority | null {
  const Match = AuthorityPattern.exec(Value.trim().toLowerCase());
  if (Match === null) return null;
  return { Name: Match[1] ?? "", Port: Match[2] === undefined ? null : Number(Match[2]) };
}

// A port-less authority is accepted (browsers omit the scheme-default port; some clients omit it).
function IsAllowedAuthority(Value: string, Port: number, ExtraHosts: readonly string[]): boolean {
  const Parsed = ParseAuthority(Value);
  if (Parsed === null) return false;
  const NameAllowed = LoopbackNames.has(Parsed.Name) || ExtraHosts.includes(Parsed.Name);
  return NameAllowed && (Parsed.Port === null || Parsed.Port === Port);
}

/** True when the Host header names this machine's loopback (localhost / 127.0.0.1 / [::1], with or
 *  without the dashboard port). A missing Host is allowed: only programmatic, in-process requests lack
 *  it — every browser (the DNS-rebinding vector) always sends one. ExtraHosts = a concrete bind address. */
export function IsLocalHost(Host: string | null, Port: number, ExtraHosts: readonly string[] = []): boolean {
  if (Host === null) return true;
  return IsAllowedAuthority(Host, Port, ExtraHosts);
}

/** True when the Origin header is a loopback http(s) origin. A MISSING Origin is allowed (curl,
 *  scripts and same-origin GETs send none); a PRESENT non-local Origin — including the opaque "null"
 *  of sandboxed frames and file:// pages — is rejected. */
export function IsLocalOrigin(Origin: string | null, Port: number, ExtraHosts: readonly string[] = []): boolean {
  if (Origin === null) return true;
  let Parsed: URL;
  try {
    Parsed = new URL(Origin);
  } catch {
    return false; // "null" or garbage
  }
  if (Parsed.protocol !== "http:" && Parsed.protocol !== "https:") return false;
  return IsAllowedAuthority(Parsed.host, Port, ExtraHosts);
}

/** The interface to bind: FOUNDRY_BIND when set and non-empty, else loopback. */
export function ResolveBindHost(Env: Readonly<Record<string, string | undefined>> = process.env): string {
  const Value = Env[BindEnvVar]?.trim() ?? "";
  return Value === "" ? DefaultBindHost : Value;
}

/** Extra Host/Origin names to accept for a bind address: a concrete non-loopback address (e.g. a LAN
 *  IP the operator chose) is reachable by that name; a wildcard bind adds nothing — reach it via
 *  localhost (e.g. a container port published on the host keeps Host = localhost). */
export function BindHostAliases(BindHost: string): string[] {
  const Name = BindHost.trim().toLowerCase();
  if (WildcardBinds.has(Name) || LoopbackNames.has(Name)) return [];
  return [Name.includes(":") && !Name.startsWith("[") ? "[" + Name + "]" : Name];
}

/** True for a JSON request body — the only kind the API accepts. A cross-site <form> can only send
 *  urlencoded/multipart/text-plain without a CORS preflight, so this blocks form-based CSRF. */
export function IsJsonContentType(Value: string | null): boolean {
  return (Value ?? "").trim().toLowerCase().startsWith("application/json");
}

/** True for a well-formed conversation id (it is echoed into pages and used as a storage key). */
export function IsValidConvId(Value: unknown): Value is string {
  return typeof Value === "string" && ConvIdPattern.test(Value);
}

/** A fresh, unguessable per-response nonce for the chat page's inline script. */
export function NewCspNonce(): string {
  return randomBytes(CspNonceBytes).toString("base64");
}

// Hardening every page gets: no framing (clickjacking), no plugins, no <base> hijack.
const BaseCsp = "frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'";

/** CSP for the control panel. Script-restricting directives are deliberately absent: its markup
 *  (DashboardHtml) still uses inline on* attributes, which a nonce cannot authorize. */
export function DashboardPageCsp(): string {
  return BaseCsp;
}

/** Strict CSP for the chat page: only the nonce'd inline script runs (no inline handlers, no injected
 *  scripts). Styles stay inline-capable (the page's own <style> + style attributes). The WebSocket
 *  source is listed explicitly only when Host is a verified loopback name (older engines do not map
 *  'self' to ws:), so a hostile Host header can never inject a directive. */
export function ChatPageCsp(Nonce: string, Host: string | null, Port: number, ExtraHosts: readonly string[] = []): string {
  const WsSources = Host !== null && IsLocalHost(Host, Port, ExtraHosts) ? " ws://" + Host + " wss://" + Host : "";
  return [
    "default-src 'self'",
    "script-src 'nonce-" + Nonce + "'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "connect-src 'self'" + WsSources,
    BaseCsp,
  ].join("; ");
}
