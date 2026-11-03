// The frozen held-out set is a committed JSONL file (one whole source file per line, with its kind), so
// BPB is reported per language and source files are never cut at blank lines like plain-text input.

import { test, expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { LoadHeldOutFromFile } from "../Brain/Eval/Suite/SuiteCorpora.ts";

function WithTempFile(Name: string, Text: string, Body: (Path: string) => void): void {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-heldout-"));
  try {
    const Path = join(Dir, Name);
    writeFileSync(Path, Text);
    Body(Path);
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
}

test("a .jsonl held-out file keeps whole documents and their kinds", () => {
  const Code = "export function Add(A: number, B: number): number {\n  return A + B;\n}\n\nexport const Zero = 0;\n";
  const Lines = [JSON.stringify({ Kind: "ts", Id: "a/b:src/add.ts", Content: Code }), "", JSON.stringify({ Kind: "go", Id: "c/d:main.go", Content: "package main\n\nfunc main() {}\n" })];
  WithTempFile("held.jsonl", Lines.join("\n") + "\n", (Path) => {
    const HeldOut = LoadHeldOutFromFile(Path);
    expect(HeldOut.Docs.length).toBe(2);
    expect(HeldOut.Docs[0]).toEqual({ Kind: "ts", Id: "a/b:src/add.ts", Content: Code }); // blank line inside kept
    expect(HeldOut.Docs.map((D) => D.Kind)).toEqual(["ts", "go"]);
    expect(HeldOut.Note).toContain("ts=1");
  });
});

test("a malformed .jsonl line is reported with its line number", () => {
  WithTempFile("bad.jsonl", JSON.stringify({ Kind: "ts", Id: "x", Content: "a" }) + "\n" + JSON.stringify({ Kind: "ts", Content: "no id" }) + "\n", (Path) => {
    expect(() => LoadHeldOutFromFile(Path)).toThrow(/bad\.jsonl:2/);
  });
});

test("the committed frozen set HeldOutV1 parses and covers the code slices", () => {
  const HeldOut = LoadHeldOutFromFile("Data/Eval/HeldOutV1.jsonl");
  const Kinds = new Set(HeldOut.Docs.map((D) => D.Kind));
  for (const Kind of ["ts", "js", "go", "css"]) expect(Kinds.has(Kind)).toBe(true);
  expect(new Set(HeldOut.Docs.map((D) => D.Id)).size).toBe(HeldOut.Docs.length); // ids unique
});
