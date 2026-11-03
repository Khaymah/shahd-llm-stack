import { test, expect } from "bun:test";
import { IsLocalOrigin, IsLocalHost, IsJsonContentType, IsValidConvId, ResolveBindHost, BindHostAliases, ChatPageCsp, DefaultBindHost } from "../Foundry/DashboardGuard.ts";

const Port = 8090;

test("IsLocalOrigin accepts loopback origins and a missing Origin, rejects everything else", () => {
  for (const Origin of ["http://localhost:8090", "http://127.0.0.1:8090", "http://[::1]:8090", "http://localhost", "https://LOCALHOST:8090"]) {
    expect(IsLocalOrigin(Origin, Port)).toBe(true);
  }
  expect(IsLocalOrigin(null, Port)).toBe(true); // curl / scripts send no Origin
  for (const Origin of ["https://evil.example", "null", "", "http://localhost:9999", "http://localhost.evil.example:8090", "http://127.0.0.1.nip.io:8090", "file://", "chrome-extension://abc"]) {
    expect(IsLocalOrigin(Origin, Port)).toBe(false);
  }
});

test("IsLocalHost blocks DNS-rebinding Host names", () => {
  for (const Host of ["localhost:8090", "127.0.0.1:8090", "[::1]:8090", "localhost", "127.0.0.1"]) expect(IsLocalHost(Host, Port)).toBe(true);
  expect(IsLocalHost(null, Port)).toBe(true);
  for (const Host of ["evil.example:8090", "evil.example", "", "localhost:1", "localhost:8090; script-src *", "192.168.1.5:8090"]) expect(IsLocalHost(Host, Port)).toBe(false);
  expect(IsLocalHost("192.168.1.5:8090", Port, BindHostAliases("192.168.1.5"))).toBe(true); // explicit LAN bind
});

test("bind host resolution and aliases", () => {
  expect(ResolveBindHost({})).toBe(DefaultBindHost);
  expect(ResolveBindHost({ FOUNDRY_BIND: "  " })).toBe(DefaultBindHost);
  expect(ResolveBindHost({ FOUNDRY_BIND: "0.0.0.0" })).toBe("0.0.0.0");
  expect(BindHostAliases("0.0.0.0")).toEqual([]);
  expect(BindHostAliases("127.0.0.1")).toEqual([]);
  expect(BindHostAliases("fe80::1")).toEqual(["[fe80::1]"]);
});

test("JSON content type and conversation id validators", () => {
  expect(IsJsonContentType("application/json")).toBe(true);
  expect(IsJsonContentType("Application/JSON; charset=utf-8")).toBe(true);
  expect(IsJsonContentType("text/plain;charset=UTF-8")).toBe(false);
  expect(IsJsonContentType(null)).toBe(false);
  expect(IsValidConvId("3f2a9c1e-0b7d-4c55-9a1e-2f6d8b0c4e11")).toBe(true);
  expect(IsValidConvId("c-18f3a2b-1a2b3c")).toBe(true);
  expect(IsValidConvId('&quot;);alert(1)//')).toBe(false);
  expect(IsValidConvId("")).toBe(false);
  expect(IsValidConvId("a".repeat(65))).toBe(false);
  expect(IsValidConvId(42)).toBe(false);
});

test("chat CSP lists the WebSocket source only for a verified loopback Host", () => {
  expect(ChatPageCsp("abc", "localhost:8090", Port)).toContain("connect-src 'self' ws://localhost:8090 wss://localhost:8090");
  const Hostile = ChatPageCsp("abc", "evil.example; script-src *", Port);
  expect(Hostile).not.toContain("evil");
  expect(Hostile).toContain("script-src 'nonce-abc'");
});
