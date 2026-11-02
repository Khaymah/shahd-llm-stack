import { test, expect } from "bun:test";
import type { ChatDocument } from "../Brain/Sft/ChatDocument.ts";
import { ConversationHash, SerializeChatDocument } from "../Brain/Sft/ChatDocument.ts";
import { AnswerTurn, BuildSynthDocument, CallTurn } from "../Brain/Sft/Synthetic/SynthConversation.ts";
import type { SynthRecord } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { DedupSynthRecords, ParseSynthRecords, SerializeSynthRecord, SyntheticProvenance } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { SynthToolReplay } from "../Brain/Sft/Synthetic/SynthTools.ts";
import { UngroundedSourceLicense } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import type { TeacherRequest } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { InMemoryDocumentStore } from "../Foundry/InMemoryDocumentStore.ts";
import { ImportSyntheticRecords } from "../Foundry/SyntheticImport.ts";
import { ParseSyntheticLicense, SyntheticLicense } from "../Foundry/SyntheticLicense.ts";
import { ClassifyDocument } from "../Foundry/Tiering.ts";
import { CreateOpenAiTeacher, ResolveTeacherModel } from "../Foundry/TeacherClient.ts";
import type { TeacherFetch } from "../Foundry/TeacherClient.ts";

const Teacher = "fake-teacher";
const NoSleep = async (): Promise<void> => {};

function MakeRecord(Document: ChatDocument, SourceId: string, SourceLicense: string, Generator: SynthRecord["Generator"]): SynthRecord {
  return {
    Generator, Teacher, Slice: String(Document.Meta.Slice), Source: Generator, SourceId, SourceLicense,
    Provenance: SyntheticProvenance(Generator, Teacher, SourceId), Tokens: 100, Hash: ConversationHash(Document), Document,
  };
}

function ToolRecord(Result: Record<string, unknown> = { result: 391 }): SynthRecord {
  const Doc = BuildSynthDocument({ Generator: "tools", Teacher, SourceId: "tools-0", Slice: "general", ClockMs: 1_735_689_600_000 }, [
    { Role: "User", Content: "what's 17 times 23? I need it for a grid layout." },
    CallTurn("Multiply with the calculator.", { name: "calculator", arguments: { a: 17, op: "*", b: 23 } }),
    { Role: "Tool", Name: "calculator", Result },
    AnswerTurn("It returned 391.", "17 × 23 = 391, so the grid has 391 cells in total."),
  ]);
  return MakeRecord(Doc, "tools-0", UngroundedSourceLicense, "tools");
}

function GroundedRecord(SourceLicense: string): SynthRecord {
  const Doc = BuildSynthDocument({ Generator: "grounded", Teacher, SourceId: "abc123", Slice: "ts", Task: "Explain" }, [
    { Role: "User", Content: "What does this helper do?\n\n```ts\nexport function clamp(n: number, lo: number, hi: number): number {\n  return Math.min(Math.max(n, lo), hi);\n}\n```" },
    AnswerTurn("Explain clamp briefly.", "It limits `n` to the range from `lo` to `hi`: values below `lo` become `lo`, values above `hi` become `hi`, and anything in between is returned unchanged."),
  ]);
  return MakeRecord(Doc, "abc123", SourceLicense, "grounded");
}

test("import is idempotent and stores synthetic conversations as Filtered with full provenance", async () => {
  const Store = new InMemoryDocumentStore();
  const Records = [ToolRecord(), GroundedRecord("MIT")];
  const First = await ImportSyntheticRecords(Records, Store, "2026-11-02T00:00:00Z", SynthToolReplay());
  expect(First.Stats.New).toBe(2);
  expect(First.Stats.ByTier.Filtered).toBe(2);
  const Second = await ImportSyntheticRecords(Records, Store, "2026-11-03T00:00:00Z", SynthToolReplay());
  expect(Second.Stats.New).toBe(0);
  expect(Second.Stats.Duplicate).toBe(2);
  const All = await Store.All();
  expect(All).toHaveLength(2);
  const Grounded = All.find((D) => D.Source === "synthetic-grounded")!;
  expect(Grounded.Origin).toBe("synthetic");
  expect(Grounded.License).toBe("MIT;teacher:fake-teacher");
  expect(Grounded.Provenance).toBe("synthetic:grounded:fake-teacher:abc123");
  expect(Grounded.Lang).toBe("chat-ts");
  expect(Grounded.Content).toBe(SerializeChatDocument(Records[1].Document));
});

test("import refuses a record whose stored tool result is not the real one", async () => {
  const Store = new InMemoryDocumentStore();
  const Result = await ImportSyntheticRecords([ToolRecord({ result: 400 })], Store, "2026-11-02T00:00:00Z", SynthToolReplay());
  expect(Result.Invalid).toEqual({ "result-mismatch": 1 });
  expect(await Store.Count()).toBe(0);
});

test("tiering: a synthetic doc needs a permissive source, a teacher-tagged license and a valid document", () => {
  const Content = SerializeChatDocument(GroundedRecord("MIT").Document);
  expect(ClassifyDocument(SyntheticLicense("MIT", Teacher), Content, "synthetic").Tier).toBe("Filtered");
  expect(ClassifyDocument(SyntheticLicense(UngroundedSourceLicense, Teacher), Content, "synthetic").Tier).toBe("Filtered");
  const Gpl = ClassifyDocument(SyntheticLicense("GPL-3.0-only", Teacher), Content, "synthetic");
  expect(Gpl.Tier).toBe("Rejected");
  expect(Gpl.RejectReason).toContain("GPL-3.0-only");
  expect(ClassifyDocument("MIT", Content, "synthetic").Tier).toBe("Rejected"); // no teacher recorded
  expect(ClassifyDocument(SyntheticLicense("MIT", Teacher), "User: hi\n\nAssistant: hello", "synthetic").RejectReason).toContain("malformed");
  expect(ParseSyntheticLicense("MIT;teacher:qwen2.5-coder")).toEqual({ Source: "MIT", Teacher: "qwen2.5-coder" });
});

test("JSONL records round-trip; malformed lines are reported; cross-shard dedup recomputes the hash", () => {
  const A = ToolRecord();
  const B = GroundedRecord("MIT");
  const Text = [SerializeSynthRecord(A), "{not json", SerializeSynthRecord(B), JSON.stringify({ Generator: "tools" }), SerializeSynthRecord({ ...A, Hash: "forged" })].join("\n");
  const Parsed = ParseSynthRecords(Text);
  expect(Parsed.Records).toHaveLength(3);
  expect(Parsed.Invalid.map((I) => I.Line)).toEqual([2, 4]);
  const Merged = DedupSynthRecords(Parsed.Records);
  expect(Merged.Records).toHaveLength(2);
  expect(Merged.Duplicates).toBe(1);
});

// ---- the HTTP teacher client (fake fetch) ------------------------------------------------------------

function Completion(Content: string): Response {
  return new Response(JSON.stringify({ choices: [{ message: { content: Content } }] }), { status: 200, headers: { "content-type": "application/json" } });
}

const Request: TeacherRequest = { Messages: [{ role: "user", content: "hi" }], Json: true, MaxTokens: 64, Temperature: 0.5, Seed: 7 };

test("teacher client: JSON mode falls back once on HTTP 400, transient errors retry with backoff", async () => {
  const Bodies: Record<string, unknown>[] = [];
  const Statuses = [400, 200, 503, 200];
  const Fetch: TeacherFetch = async (_Url, Init) => {
    Bodies.push(JSON.parse(String(Init.body)) as Record<string, unknown>);
    const Status = Statuses.shift() ?? 200;
    return Status === 200 ? Completion("```json\n{\"ok\": true}\n```") : new Response("nope", { status: Status });
  };
  const Client = CreateOpenAiTeacher({ Url: "http://127.0.0.1:8081/", Model: "m", Fetch, Backoff: { Sleep: NoSleep } });
  expect(await Client.Complete(Request)).toContain("\"ok\": true");
  expect(Bodies[0]["response_format"]).toEqual({ type: "json_object" });
  expect(Bodies[1]["response_format"]).toBeUndefined(); // the 400 retry drops JSON mode
  expect(await Client.Complete(Request)).toContain("ok"); // 503 then 200
  expect(Bodies).toHaveLength(4);
  expect(Bodies[3]["response_format"]).toBeUndefined(); // stays off for the run
  expect(Bodies[3]["seed"]).toBe(7);
});

test("teacher client: a malformed completion is a TeacherError, not retried; the model id comes from /v1/models", async () => {
  let Calls = 0;
  const Bad = CreateOpenAiTeacher({ Url: "http://x", Model: "m", Fetch: async () => { Calls++; return new Response("{}", { status: 200 }); }, Backoff: { Sleep: NoSleep } });
  await expect(Bad.Complete(Request)).rejects.toThrow("unexpected completion shape");
  expect(Calls).toBe(1);
  const Models: TeacherFetch = async () => new Response(JSON.stringify({ data: [{ id: "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" }] }), { status: 200 });
  expect(await ResolveTeacherModel("http://x", undefined, Models)).toBe("qwen2.5-coder-1.5b-instruct-q4_k_m");
  expect(await ResolveTeacherModel("http://x", "explicit", Models)).toBe("explicit");
});
