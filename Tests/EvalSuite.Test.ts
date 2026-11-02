// The fixed evaluation suite (Brain/Eval/Suite/): end-to-end on a tiny in-test chat model, the judges
// in isolation (tool call, format, coherence), report comparison, Meta.Eval persistence, the held-out
// split rule, the memorization scan, and decontamination against the suite.

import { test, expect } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { ChatTokens, ChatTokenList } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokens, ToolTokenList } from "../Brain/Serving/ToolProtocol.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { SaveCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { LoadCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import { LoadRunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import type { RunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { Decontaminate } from "../Brain/Data/Decontamination.ts";
import { DocumentSplitBucket, IsValidationDocument, ValidationPercent } from "../Brain/Data/DocumentSplit.ts";
import {
  RunSuite, CompareReports, FormatReport, WriteEvalReportToFile, ReadEvalReport, ParseSuiteReport, SuiteEvalDocs,
  LoadHeldOutFromStore, LoadTrainCorpusFromStore, LoadHeldOutFromSeed,
} from "../Brain/Eval/Suite/SuiteBarrel.ts";
import type { EvalCaseSet, GeneratorFactory, HeldOutSet, MetricResult, SuiteDocumentSource, SuiteReport, TrainCorpus } from "../Brain/Eval/Suite/SuiteBarrel.ts";
import { ReadTurnIds } from "../Brain/Eval/Suite/SuiteGeneration.ts";
import { FormatFailures } from "../Brain/Eval/Suite/FormatCompliance.ts";
import { JudgeToolCall } from "../Brain/Eval/Suite/ToolCallAccuracy.ts";
import { CoherenceFailures } from "../Brain/Eval/Suite/MultiTurnCoherence.ts";
import { ExtractCode } from "../Brain/Eval/Suite/CodeExecution.ts";
import { MarkFound } from "../Brain/Eval/Suite/MemorizationProbe.ts";
import { CapCases } from "../Brain/Eval/Suite/SuiteSettings.ts";

const Specials = [...ChatTokenList, ...ToolTokenList];
const PrintableFirst = 32;
const PrintableLast = 126;

function TinyTokenizer(): SpecialTokenizer {
  const Chars = ["\n"];
  for (let C = PrintableFirst; C <= PrintableLast; C++) Chars.push(String.fromCharCode(C));
  return new SpecialTokenizer(new CharTokenizer(Chars, { Lenient: true }), Specials);
}

function TinyRunnable(Chat: boolean): RunnableModel {
  const Tokenizer = TinyTokenizer();
  const Config = LoadConfig({ Overrides: { Model: { VocabSize: Tokenizer.VocabSize, EmbedDim: 16, NumLayers: 1, NumHeads: 2, BlockSize: 32 } }, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  return { Model: new Shahd(Config, Rng.InitRng), Tokenizer, Config, Rng, Chat };
}

const CalcCall = { name: "calculator", arguments: { a: 7, op: "+", b: 5 } };

const CaseSet: EvalCaseSet = {
  Version: "test",
  Code: [
    { Id: "ts-1", Slice: "ts", Prompt: "Write add(a, b).", Tests: "if (add(2, 3) !== 5) throw new Error('add');", Reference: "function add(a, b) { return a + b; }" },
    { Id: "sql-1", Slice: "sql", Prompt: "Select every user.", StaticChecks: ["select\\s+\\*", "from\\s+users"], Reference: "SELECT * FROM users;" },
  ],
  Format: [{ Id: "fmt-1", Slice: "general", Messages: [{ Role: "User", Content: "hi" }] }],
  Tools: [
    { Id: "tool-1", Slice: "general", User: "What is 7 + 5?", Expected: { Name: "calculator", Arguments: { a: 7, op: "+", b: 5 } } },
    { Id: "tool-2", Slice: "ts", User: "What is 7 + 6?", Expected: { Name: "calculator", Arguments: { a: 7, op: "+", b: 6 } } },
  ],
  MultiTurn: [{
    Id: "mt-1", Slice: "general",
    Turns: [{ Role: "User", Content: "My name is Layla." }, { Role: "Assistant", Content: "Hi Layla." }, { Role: "User", Content: "My name?" }],
    MustContain: ["Layla"],
  }],
};

const Text = "function add(a, b) {\n  return a + b;\n}\nconst total = [1, 2, 3].reduce((x, y) => x + y, 0);\n";
const HeldOut: HeldOutSet = { Origin: "file", Note: "in-test", Docs: [{ Kind: "code", Id: "h1", Content: Text.repeat(3) }, { Kind: "prose", Id: "h2", Content: "the quick brown fox jumps over the lazy dog. ".repeat(4) }] };
const Train: TrainCorpus = { Origin: "file", Note: "in-test", Docs: [Text.repeat(2), "let x = 1; let y = 2; console.log(x + y);"], Bytes: 300 };

const FastSettings = {
  CodeMaxTokens: 8, ReplyMaxTokens: 8, ToolMaxTokens: 8, PassAtKSamples: 2, BpbMaxWindows: 2, CaseTimeoutMs: 30_000,
  MemorizationPromptsPerSide: 2, MemorizationPromptTokens: 8, MemorizationNewTokens: 12, MemorizationNgram: 4,
};

function Metric(Report: SuiteReport, Name: string): MetricResult {
  const Found = Report.Metrics.find((M) => M.Name === Name);
  if (Found === undefined) throw new Error(`missing metric ${Name}`);
  return Found;
}

test("RunSuite runs end-to-end on a tiny chat model and returns every metric with N > 0", async () => {
  const Report = await RunSuite(TinyRunnable(true), CaseSet, { Checkpoint: "tiny", Settings: FastSettings, HeldOut, TrainCorpus: Train });
  const Names = ["bpb.code", "bpb.prose", "bpb.all", "code.pass@1", "code.pass@2", "format.compliance", "tool.accuracy", "tool.name_accuracy", "multiturn.coherence", "memorization.ngram", "memorization.heldout_vs_train_gap"];
  for (const Name of Names) {
    const M = Metric(Report, Name);
    expect(M.N).toBeGreaterThan(0);
    expect(Number.isFinite(M.Value)).toBe(true);
  }
  expect(Metric(Report, "bpb.all").Value).toBeGreaterThan(0); // an untrained model is far from 0 bits
  expect(Report.SuiteVersion).toBe("test");
  expect(Report.Slices.ts?.map((M) => M.Name)).toContain("code.pass@1");
  expect(Report.Slices.sql?.find((M) => M.Name === "code.pass@1")?.N).toBe(1);
  expect(ParseSuiteReport(JSON.parse(JSON.stringify(Report)))).toEqual(ParseSuiteReport(Report)); // JSON-safe
  expect(FormatReport(Report)).toContain("tool.accuracy");
}, 120_000);

test("a base model runs BPB + code + memorization; chat-only metrics are skipped with a reason", async () => {
  const Report = await RunSuite(TinyRunnable(false), CaseSet, { Checkpoint: "base", Settings: { ...FastSettings, PassAtKSamples: 1 }, HeldOut, TrainCorpus: null });
  expect(Metric(Report, "code.pass@1").N).toBe(2);
  for (const Name of ["format.compliance", "tool.accuracy", "multiturn.coherence", "memorization.ngram"]) {
    expect(Metric(Report, Name).N).toBe(0);
    expect(typeof Metric(Report, Name).Details?.["Skipped"]).toBe("string");
  }
}, 120_000);

test("a stub generator that always emits the expected tool call scores tool.accuracy = 1 on matching cases", async () => {
  const Runnable = TinyRunnable(true);
  const Tok = Runnable.Tokenizer as SpecialTokenizer;
  const Stub: GeneratorFactory = () => () => ({
    Ids: [Tok.Id(ToolTokens.CallStart), ...Tok.EncodeBase(JSON.stringify(CalcCall)), Tok.Id(ToolTokens.CallEnd), Tok.Id(ChatTokens.EndOfTurn)],
    Tok,
  });
  const OnlyFirst: EvalCaseSet = { ...CaseSet, Code: [], Tools: [CaseSet.Tools[0]] };
  const Report = await RunSuite(Runnable, OnlyFirst, { Checkpoint: "stub", Settings: FastSettings, HeldOut, Generators: Stub });
  expect(Metric(Report, "tool.accuracy")).toMatchObject({ Value: 1, N: 1 });
  expect(Metric(Report, "tool.name_accuracy")).toMatchObject({ Value: 1, N: 1 });
  // Against both cases the second expects b = 6: the name is right, the arguments are not.
  const Both = await RunSuite(Runnable, { ...OnlyFirst, Tools: CaseSet.Tools }, { Checkpoint: "stub", Settings: FastSettings, HeldOut, Generators: Stub });
  expect(Metric(Both, "tool.accuracy").Value).toBeCloseTo(0.5, 10);
  expect(Metric(Both, "tool.name_accuracy").Value).toBe(1);
  expect(Metric(Both, "tool.accuracy").Details?.["FailureReasons"]).toEqual({ "wrong-arguments": 1 });
}, 60_000);

test("code: a stub answering each case's reference in a fence passes pass@1 and pass@k (tests AND static checks)", async () => {
  const Runnable = TinyRunnable(true);
  const Tok = Runnable.Tokenizer as SpecialTokenizer;
  const Stub: GeneratorFactory = () => (Session) => {
    const Prompt = Session.Messages[Session.Messages.length - 1].Content;
    const Case = CaseSet.Code.find((C) => C.Prompt === Prompt);
    return { Ids: [...Tok.EncodeBase(`Sure:\n\`\`\`ts\n${Case?.Reference ?? ""}\n\`\`\`\n`), Tok.Id(ChatTokens.EndOfTurn)], Tok };
  };
  const Report = await RunSuite(Runnable, { ...CaseSet, Format: [], Tools: [], MultiTurn: [] }, { Checkpoint: "ref", Settings: FastSettings, HeldOut, Generators: Stub });
  expect(Metric(Report, "code.pass@1")).toMatchObject({ Value: 1, N: 2 });
  expect(Metric(Report, "code.pass@2")).toMatchObject({ Value: 1, N: 2 });
}, 60_000);

test("JudgeToolCall: subset argument match, extra keys allowed, multiple calls rejected", () => {
  const Expected = { Name: "calculator", Arguments: { a: 7, op: "+" } };
  expect(JudgeToolCall(Expected, { Name: "calculator", Arguments: { a: 7, op: "+", b: 5 } }, 1)).toEqual({ NameOk: true, ArgumentsOk: true, Reasons: [] });
  expect(JudgeToolCall(Expected, { Name: "calculator", Arguments: { a: "7", op: "+" } }, 1).ArgumentsOk).toBe(false); // strict types
  expect(JudgeToolCall(Expected, { Name: "stats", Arguments: {} }, 1).Reasons).toEqual(["wrong-tool-name"]);
  expect(JudgeToolCall(Expected, { Name: "calculator", Arguments: { a: 7, op: "+" } }, 2).Reasons).toEqual(["multiple-tool-calls"]);
  expect(JudgeToolCall(Expected, null, 0).Reasons).toEqual(["no-tool-call"]);
});

test("format: a reply with a leaked '<|tool_call|>' literal fails; a clean, closed turn passes", () => {
  const Tok = TinyTokenizer();
  const End = Tok.Id(ChatTokens.EndOfTurn);
  expect(FormatFailures(ReadTurnIds([...Tok.EncodeBase("Hello there."), End], Tok))).toEqual([]);
  const Leaked = ReadTurnIds([...Tok.EncodeBase(`ok ${ToolTokens.CallStart}{"name":"x"}`), End], Tok);
  expect(Leaked.ToolCallCount).toBe(0); // spelled in byte tokens: not a call
  expect(FormatFailures(Leaked)).toEqual(["leaked-control-token"]);
  expect(FormatFailures(ReadTurnIds(Tok.EncodeBase("never stops"), Tok))).toEqual(["no-end-of-turn"]);
  expect(FormatFailures(ReadTurnIds([Tok.Id(ChatTokens.Think), ...Tok.EncodeBase("hmm"), End], Tok))).toEqual(["empty-reply", "unclosed-thinking"]);
  expect(FormatFailures(ReadTurnIds([...Tok.EncodeBase("hi"), Tok.Id(ChatTokens.User)], Tok))).toEqual(["ended-by-role-marker"]);
  const Thought = ReadTurnIds([Tok.Id(ChatTokens.Think), ...Tok.EncodeBase("plan"), Tok.Id(ChatTokens.EndThink), ...Tok.EncodeBase("Answer"), End], Tok);
  expect(Thought.Answer).toBe("Answer");
  expect(FormatFailures(Thought)).toEqual([]);
});

test("multi-turn coherence is case-insensitive on MustContain / MustNotContain", () => {
  const Case = { Id: "x", Slice: "general" as const, Turns: [], MustContain: ["Layla"], MustNotContain: ["python"] };
  expect(CoherenceFailures(Case, "your name is LAYLA")).toEqual([]);
  expect(CoherenceFailures(Case, "I do not know")).toEqual(["missing-expected"]);
  expect(CoherenceFailures(Case, "Layla writes Python")).toEqual(["contains-forbidden"]);
});

test("CapCases stratifies a small cap across slices, keeps file order, and 0 means every case", () => {
  const Cases = [
    { Id: "ts-1", Slice: "ts" }, { Id: "ts-2", Slice: "ts" }, { Id: "ts-3", Slice: "ts" },
    { Id: "go-1", Slice: "go" }, { Id: "sql-1", Slice: "sql" }, { Id: "sql-2", Slice: "sql" },
  ];
  expect(CapCases(Cases, 3).map((C) => C.Id)).toEqual(["ts-1", "go-1", "sql-1"]);
  expect(CapCases(Cases, 5).map((C) => C.Id)).toEqual(["ts-1", "ts-2", "go-1", "sql-1", "sql-2"]);
  expect(CapCases(Cases, 0)).toBe(Cases);
  expect(CapCases(Cases, 99)).toBe(Cases);
});

test("ExtractCode takes the first fenced block, an unclosed fence to the end, else the whole reply", () => {
  expect(ExtractCode("Here:\n```ts\nconst a = 1;\n```\nmore")).toBe("const a = 1;\n");
  expect(ExtractCode("```\nlet b = 2;")).toBe("let b = 2;");
  expect(ExtractCode("function f() {}")).toBe("function f() {}");
});

function StubReport(Name: string, Metrics: MetricResult[]): SuiteReport {
  return { SuiteVersion: "v1", Checkpoint: Name, ConfigHash: "h", Params: 1, RanAt: "2026-11-02T00:00:00.000Z", DurationMs: 1, Metrics, Slices: { ts: [{ Name: "code.pass@1", Value: 0.5, Higher: "better", N: 2 }] } };
}

test("CompareReports produces direction-aware deltas and n/a for unmeasured metrics", () => {
  const A = StubReport("a", [
    { Name: "code.pass@1", Value: 0.2, Higher: "better", N: 10 },
    { Name: "bpb.all", Value: 2.0, Higher: "worse", N: 100 },
    { Name: "memorization.ngram", Value: 0, Higher: "worse", N: 0 },
  ]);
  const B = StubReport("b", [
    { Name: "code.pass@1", Value: 0.3, Higher: "better", N: 10 },
    { Name: "bpb.all", Value: 2.5, Higher: "worse", N: 100 },
    { Name: "memorization.ngram", Value: 0.1, Higher: "worse", N: 50 },
  ]);
  const { Rows, Table } = CompareReports(A, B);
  const Row = (Name: string): (typeof Rows)[number] | undefined => Rows.find((R) => R.Metric === Name);
  expect(Row("code.pass@1")?.Delta).toBeCloseTo(0.1, 10);
  expect(Row("code.pass@1")?.Direction).toBe("better");
  expect(Row("bpb.all")?.Delta).toBeCloseTo(0.5, 10);
  expect(Row("bpb.all")?.Direction).toBe("worse"); // higher bpb is a regression
  expect(Row("memorization.ngram")).toMatchObject({ Baseline: null, Delta: null, Direction: "n/a" });
  expect(Row("code.pass@1 [ts]")?.Direction).toBe("same");
  expect(Table).toContain("▲ better");
  expect(Table).toContain("▼ worse");
});

test("--WriteMeta: a report stored under Meta.Eval round-trips through the checkpoint writer/reader", () => {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-eval-"));
  try {
    const Runnable = TinyRunnable(true);
    const Path = join(Dir, "Tiny.ckpt");
    SaveCheckpoint(Path, Runnable.Model, CreateOptimizer(Runnable.Model.Parameters(), Runnable.Config), Runnable.Rng, { Format: "chat" }, { Kind: "Char", Chars: [" ", "a"], Specials });
    const Report = StubReport("Tiny", [{ Name: "bpb.all", Value: 1.25, Higher: "worse", N: 64, Details: { Origin: "seed" } }]);
    WriteEvalReportToFile(Path, Report);
    const Back = LoadCheckpoint(Path); // re-verifies the payload checksum
    expect(ReadEvalReport(Back)).toEqual(Report);
    expect(Back.Meta["Format"]).toBe("chat"); // existing Meta preserved
    expect(LoadRunnableModel(Path).Chat).toBe(true); // still a loadable model
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
});

// A fake per-kind store: ids in ascending order, with keyset paging and a page cap like Postgres.
function FakeStore(Ids: readonly string[], PageCap: number): SuiteDocumentSource {
  const Sorted = [...Ids].sort();
  return {
    ListSummaries: async (Opts) => ({ Rows: Sorted.filter((Id) => Id > (Opts.AfterId ?? "")).slice(0, Math.min(Opts.Limit ?? PageCap, PageCap)).map((Id) => ({ Id })) }),
    ContentByIds: async (Wanted) => new Map(Wanted.map((Id) => [Id, `content of ${Id}`])),
    ByTier: async (_Tier, Limit) => Sorted.slice(0, Limit).map((Id) => ({ Id, Content: `content of ${Id}` })),
  };
}

test("held-out selection uses the shared validation split, ascending ids, capped per kind; training excludes it", async () => {
  const Ids = Array.from({ length: 400 }, (_V, I) => `doc-${String(I).padStart(4, "0")}`);
  expect(IsValidationDocument(Ids[0])).toBe(DocumentSplitBucket(Ids[0]) < ValidationPercent);
  const Sources = new Map([["code", FakeStore(Ids, 7)]]); // a tiny page cap exercises multi-page scans
  const Held = await LoadHeldOutFromStore(Sources, 5, 10_000);
  const Expected = Ids.filter(IsValidationDocument).slice(0, 5);
  expect(Held.Docs.map((D) => D.Id)).toEqual(Expected);
  expect(Held.Docs.every((D) => D.Kind === "code")).toBe(true);
  const TrainSide = await LoadTrainCorpusFromStore(Sources, 1_000_000);
  expect(TrainSide.Docs.length).toBe(Ids.filter((Id) => !IsValidationDocument(Id)).length);
  expect(TrainSide.Docs.some((D) => Expected.some((Id) => D === `content of ${Id}`))).toBe(false);
  expect(LoadHeldOutFromSeed().Docs.length).toBeGreaterThan(0); // the last-resort source exists
});

test("MarkFound (rolling hash) finds exactly the n-grams a naive scan finds", () => {
  const Rng = CreateRngStreams(7).DataRng;
  const N = 4;
  const Corpus = Array.from({ length: 500 }, () => Math.floor(Rng.NextFloat() * 6));
  const Generated = Array.from({ length: 60 }, () => Math.floor(Rng.NextFloat() * 6));
  const Grams = Array.from({ length: Generated.length - N + 1 }, (_V, From) => ({ Ids: Generated, From, Found: false }));
  const ByHash = new Map<number, typeof Grams>();
  const Hash = (Ids: readonly number[], From: number): number => {
    let H = 0;
    for (let J = From; J < From + N; J++) H = (Math.imul(H, 0x01000193) + Ids[J]) >>> 0;
    return H;
  };
  for (const G of Grams) ByHash.set(Hash(G.Ids, G.From), [...(ByHash.get(Hash(G.Ids, G.From)) ?? []), G]);
  MarkFound(Corpus, ByHash, N);
  const Key = (Ids: readonly number[], From: number): string => Ids.slice(From, From + N).join(",");
  const CorpusKeys = new Set(Array.from({ length: Corpus.length - N + 1 }, (_V, I) => Key(Corpus, I)));
  for (const G of Grams) expect(G.Found).toBe(CorpusKeys.has(Key(G.Ids, G.From)));
});

test("SuiteEvalDocs drive decontamination: a training doc copying a case is dropped, an unrelated one kept", () => {
  const Copy = `// helper\n${CaseSet.Code[0].Reference}\n${CaseSet.Code[0].Tests ?? ""}\n// end`;
  const Unrelated = "export function multiplyMatrices(left: number[][], right: number[][]): number[][] { return left.map((row) => row); }";
  const Result = Decontaminate([Copy, Unrelated], SuiteEvalDocs(CaseSet), 8);
  expect(Result.Removed).toEqual([0]);
  expect(Result.Kept).toEqual([1]);
});
