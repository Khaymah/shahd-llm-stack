import { test, expect } from "bun:test";
import {
  AssertSafeName,
  ReadArg,
  ReadBoolArg,
  ReadFlag,
  ReadFloatArg,
  ReadIntArg,
  ReadNameArg,
} from "../Scripts/ScriptArgs.ts";

test("ReadIntArg parses integers, applies the default, and rejects junk with the flag's name", () => {
  expect(ReadIntArg("--Steps", 600, {}, ["--Steps=250"])).toBe(250);
  expect(ReadIntArg("--Steps=", 600, {}, ["--Other=1"])).toBe(600); // legacy "=" form works too
  expect(() => ReadIntArg("--Steps", 600, {}, ["--Steps=abc"])).toThrow("--Steps expects an integer, got 'abc'");
  expect(() => ReadIntArg("--Steps", 600, {}, ["--Steps=1.5"])).toThrow(/expects an integer/);
  expect(() => ReadIntArg("--Steps", 600, {}, ["--Steps="])).toThrow(/expects an integer, got ''/);
  expect(() => ReadIntArg("--Steps", 600, {}, ["--Steps=Infinity"])).toThrow(/expects an integer/);
  expect(() => ReadIntArg("--Steps", 600, {}, ["--Steps"])).toThrow(/expects a value/);
});

test("numeric readers enforce Min/Max bounds (including on the default)", () => {
  expect(() => ReadIntArg("--Workers", 0, { Min: 0, Max: 64 }, ["--Workers=-1"])).toThrow("--Workers must be >= 0, got '-1'");
  expect(() => ReadIntArg("--Workers", 0, { Min: 0, Max: 64 }, ["--Workers=65"])).toThrow(/<= 64/);
  expect(() => ReadIntArg("--Workers", -3, { Min: 0 }, [])).toThrow(/default -3/);
  expect(ReadIntArg("--Workers", 0, { Min: 0, Max: 64 }, ["--Workers=64"])).toBe(64);
});

test("ReadFloatArg accepts finite numbers and rejects NaN/Infinity/text", () => {
  expect(ReadFloatArg("--Lr", 0.002, { Min: 0 }, ["--Lr=3e-4"])).toBeCloseTo(3e-4, 12);
  expect(ReadFloatArg("--Lr", 0.002, {}, [])).toBe(0.002);
  expect(() => ReadFloatArg("--Lr", 0.002, {}, ["--Lr=NaN"])).toThrow("--Lr expects a number, got 'NaN'");
  expect(() => ReadFloatArg("--Lr", 0.002, {}, ["--Lr=Infinity"])).toThrow(/expects a number/);
  expect(() => ReadFloatArg("--Lr", 0.002, {}, ["--Lr=fast"])).toThrow(/expects a number/);
});

test("ReadBoolArg / ReadFlag: bare flag, explicit values, default, and junk", () => {
  expect(ReadBoolArg("--Fresh", false, ["--Fresh"])).toBe(true);
  expect(ReadBoolArg("--Fresh", true, ["--Fresh=false"])).toBe(false);
  expect(ReadBoolArg("--Fresh", false, ["--Fresh=yes"])).toBe(true);
  expect(ReadBoolArg("--Fresh", true, [])).toBe(true);
  expect(() => ReadBoolArg("--Fresh", false, ["--Fresh=maybe"])).toThrow("--Fresh expects a boolean (true/false), got 'maybe'");
  expect(ReadFlag("--DryRun", ["--DryRun"])).toBe(true);
  expect(ReadFlag("--DryRun", [])).toBe(false);
});

test("ReadArg keeps its raw-string behavior for existing callers", () => {
  expect(ReadArg("--Name=", "Chat", ["--Name=Mine"])).toBe("Mine");
  expect(ReadArg("--Name=", "Chat", [])).toBe("Chat");
  expect(ReadArg("--Prompt=", "x", ["--Prompt=a=b"])).toBe("a=b"); // only the first "=" splits
});

test("AssertSafeName rejects names that a sanitizer would collide instead of rewriting them", () => {
  expect(AssertSafeName("my-model")).toBe("my-model");
  expect(AssertSafeName("Chat_v2.1")).toBe("Chat_v2.1");
  // "my model" would sanitize to "my-model" and overwrite that checkpoint — must be refused.
  expect(() => AssertSafeName("my model")).toThrow(/no spaces.*'my-model'/);
  expect(() => AssertSafeName("a/b")).toThrow(/may only contain/);
  expect(() => AssertSafeName("")).toThrow(/must not be empty/);
  expect(() => AssertSafeName("x".repeat(41))).toThrow(/longer than 40/);
  expect(ReadNameArg("--Name", "Chat", ["--Name=Run-7"])).toBe("Run-7");
  expect(() => ReadNameArg("--Name", "Chat", ["--Name=my model"])).toThrow(/--Name 'my model'/);
});
