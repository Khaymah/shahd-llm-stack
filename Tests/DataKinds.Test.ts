// Pins the source/provider -> kind-table routing (KindForProvider) to the full truth table, so the
// dashboard's collect routing and any other caller can never drift apart again (oasst2 once fell
// through to "code"). Also checks the parquet sources' declared Kind agrees with the routing.

import { test, expect } from "bun:test";
import { DataKinds, DefaultFolderKind, IsDataKind, KindForProvider, TableForKind } from "../Foundry/DataKinds.ts";
import type { DataKind } from "../Foundry/DataKinds.ts";
import { StackExchangeSource, WikiDumpSource } from "../Foundry/HfParquetProvider.ts";

// Every dashboard source id (LearnSettings.Source) and every provider Name, with the kind it must route to.
const TruthTable: [string, DataKind][] = [
  // dashboard source ids
  ["oasst", "conversation"],
  ["oasst2", "conversation"],
  ["stackexchange", "conversation"],
  ["wikipedia", "knowledge"],
  ["wikidump", "knowledge"],
  ["gsm8k", "instruction"],
  ["folder", "books"],
  ["github", "code"],
  ["local", "code"],
  ["both", "code"],
  // provider Name values
  ["github-repo", "code"],
  ["local-repo", "code"],
  ["local-folder", "books"],
  ["web-search", "web"],
];

test("KindForProvider routes every known source/provider to the right kind", () => {
  for (const [Source, Kind] of TruthTable) {
    expect({ Source, Kind: KindForProvider(Source) }).toEqual({ Source, Kind });
  }
});

test("KindForProvider: folder sources honor the chosen kind, defaulting to books", () => {
  expect(DefaultFolderKind).toBe("books");
  for (const Kind of DataKinds) {
    expect(KindForProvider("folder", Kind)).toBe(Kind);
    expect(KindForProvider("local-folder", Kind)).toBe(Kind);
  }
  // A chosen kind only applies to folder sources — every other source has a fixed kind.
  expect(KindForProvider("oasst2", "books")).toBe("conversation");
  expect(KindForProvider("github", "knowledge")).toBe("code");
});

test("KindForProvider: unknown sources fall back to code (the dashboard's historical default)", () => {
  expect(KindForProvider("some-new-provider")).toBe("code");
  expect(KindForProvider("")).toBe("code");
});

test("parquet sources' declared Kind agrees with the routing table", () => {
  for (const Source of [StackExchangeSource, WikiDumpSource]) {
    expect(KindForProvider(Source.Name)).toBe(Source.Kind);
  }
});

test("every kind has its own table and is recognized", () => {
  const Tables = DataKinds.map(TableForKind);
  expect(new Set(Tables).size).toBe(DataKinds.length);
  for (const Kind of DataKinds) {
    expect(IsDataKind(Kind)).toBe(true);
    expect(TableForKind(Kind)).toBe(`documents_${Kind}`);
  }
  expect(IsDataKind("documents")).toBe(false);
});
