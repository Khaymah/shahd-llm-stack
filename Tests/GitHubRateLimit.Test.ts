import { test, expect } from "bun:test";
import { MaxRateLimitWaitMs, RateLimitWaitMs } from "../Foundry/GitHubHttp.ts";

const Headers = (Values: Record<string, string>): globalThis.Headers => new globalThis.Headers(Values);
const Now = 1_700_000_000_000;

test("an exhausted rate limit waits until the reset time plus slack", () => {
  const Wait = RateLimitWaitMs(403, Headers({ "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(Now / 1000 + 120) }), Now);
  expect(Wait).toBe(125_000);
});

test("Retry-After (secondary limits) wins and the wait is capped", () => {
  expect(RateLimitWaitMs(429, Headers({ "retry-after": "60" }), Now)).toBe(60_000);
  expect(RateLimitWaitMs(403, Headers({ "retry-after": "999999" }), Now)).toBe(MaxRateLimitWaitMs);
});

test("ordinary errors and a forbidden repo with budget left are not retried", () => {
  expect(RateLimitWaitMs(404, Headers({ "x-ratelimit-remaining": "0" }), Now)).toBeNull();
  expect(RateLimitWaitMs(403, Headers({ "x-ratelimit-remaining": "41" }), Now)).toBeNull();
  expect(RateLimitWaitMs(200, Headers({}), Now)).toBeNull();
});
