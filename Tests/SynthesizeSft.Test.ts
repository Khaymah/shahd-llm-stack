// The synthetic-SFT pipeline end to end, with an IN-PROCESS fake teacher (no network, no model): it reads
// the generator prompts the way a real teacher would and answers in JSON, so every generator, the
// acceptance gate, the shard partition and the corpus grounding run for real.

import { test, expect, afterAll } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { RunAgainstTestsAsync } from "../Brain/Eval/EvalHarness.ts";
import { EvalCaseSetV1 } from "../Brain/Eval/Suite/Data/CaseSetV1.ts";
import { TrainingDecontaminationDocs } from "../Brain/Eval/Suite/SuiteEvalDocs.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import type { ChatDocument } from "../Brain/Sft/ChatDocument.ts";
import { ReplayToolCalls } from "../Brain/Sft/ChatDocumentValidation.ts";
import { SynthAcceptance } from "../Brain/Sft/Synthetic/SynthAcceptance.ts";
import { CreateDocumentMeasure, EstimateTokens } from "../Brain/Sft/Synthetic/SynthBudget.ts";
import { AnswerTurn, BuildSynthDocument } from "../Brain/Sft/Synthetic/SynthConversation.ts";
import { ParseMix, PlanQuotas } from "../Brain/Sft/Synthetic/SynthMix.ts";
import type { SynthRecord } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { InShard, ParseShard, ShardItemIndex, ShardOf } from "../Brain/Sft/Synthetic/SynthSampling.ts";
import { GroupGrounding, RunSynthesis } from "../Brain/Sft/Synthetic/SynthesisRunner.ts";
import { SynthToolReplay } from "../Brain/Sft/Synthetic/SynthTools.ts";
import type { GroundingDoc, SftTeacher, SynthContext, SynthItem } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { CodeSlices } from "../Brain/Sft/Synthetic/SynthTypes.ts";
import { TextbookGenerator } from "../Brain/Sft/Synthetic/TextbookGenerator.ts";
import { ToolUseGenerator } from "../Brain/Sft/Synthetic/ToolUseGenerator.ts";
import { GroundedCodeGenerator } from "../Brain/Sft/Synthetic/GroundedCodeGenerator.ts";
import { ExtractJsonObject } from "../Brain/Sft/Synthetic/TeacherJson.ts";
import { LoadDataSpec } from "../Foundry/DataSpec.ts";
import { LoadCorpusGrounding } from "../Foundry/SftGrounding.ts";

const TestBudget = 2048;
const Spec = LoadDataSpec("Data/Spec.ci.json");
const TempDirs: string[] = [];
afterAll(() => {
  for (const Dir of TempDirs) rmSync(Dir, { recursive: true, force: true });
});

// ---- the fake teacher -------------------------------------------------------------------------------

const GoodSolution = "function addAll(xs: number[]): number {\n  return xs.reduce((a, b) => a + b, 0);\n}";
const WrongSolution = "function addAll(xs: number[]): number {\n  return xs.reduce((a, b) => a - b, 0);\n}";
const UnitTests = "assert(addAll([1, 2, 3]) === 6, \"sums\");\nassert(addAll([]) === 0, \"empty\");";

/** The user text a fake teacher writes for a planned call: states every value the call carries. */
function FakeUserText(Call: { name: string; arguments: Record<string, unknown> }): string {
  const Args = Call.arguments;
  if (Call.name === "calculator") return `quick one: ${String(Args["a"])} ${String(Args["op"])} ${String(Args["b"])}?`;
  if (Call.name === "stats") return `I have ${(Args["values"] as number[]).join(", ")} — give me the ${String(Args["op"])}`;
  if (Call.name === "memory_store") return `please remember ${String(Args["key"])} is ${String(Args["value"])}`;
  if (Call.name === "memory_recall") return "what did I ask you to remember earlier?";
  return `can you check ${JSON.stringify(Args)} for me`;
}

type FakeOptions = { Solution?: string; ClaimedResult?: boolean; CallOverride?: (Call: Record<string, unknown>) => Record<string, unknown> };

/** Answers each generator's prompt the way a cooperative teacher would. */
function FakeTeacher(Options: FakeOptions = {}): SftTeacher {
  const Reply = (Prompt: string): unknown => {
    if (Prompt.includes("Write ONE training example for this task:")) {
      return { question: "What does this code do, and is anything wrong with it?", thinking: "Read the snippet, then answer briefly.", answer: "It exports helpers.\n\n```ts\nexport const fixed = true;\n```" };
    }
    if (Prompt.includes("textbook unit about:")) {
      return {
        thinking_lesson: "Introduce reduce with a sum.", explanation: "reduce folds an array into one value by applying a callback to an accumulator and each element.",
        example: "const total = [1, 2].reduce((a, b) => a + b, 0);", exercise: "Implement addAll(xs) that returns the sum of xs.",
        thinking_solution: "Use reduce starting at 0.", solution: Options.Solution ?? GoodSolution, tests: UnitTests,
      };
    }
    if (Prompt.includes("Write the user side of a tool-use training conversation")) {
      const Calls = [...Prompt.matchAll(/Shahd's call must be exactly (\{.*\})\.$/gm)].map((M) => JSON.parse(M[1]) as { name: string; arguments: Record<string, unknown> });
      return {
        steps: Calls.map((Call) => ({
          user: FakeUserText(Call),
          thinking: `I should use ${Call.name}.`,
          call: Options.CallOverride === undefined ? Call : { ...Call, arguments: Options.CallOverride(Call.arguments) },
          ...(Options.ClaimedResult === true ? { result: { result: 999 } } : {}), // a claim the pipeline must ignore
        })),
      };
    }
    const Returned = /it returned exactly: (.*)$/m.exec(Prompt);
    if (Returned !== null) return { thinking: "The tool answered.", answer: `Done: ${Returned[1]}` };
    return { nothing: true };
  };
  // Wrapped in a fence + prose half the time: the extractor must cope with both.
  let Calls = 0;
  return {
    Model: "fake-teacher",
    Complete: async (Request) => {
      Calls++;
      const Json = JSON.stringify(Reply(Request.Messages[Request.Messages.length - 1].content));
      return Calls % 2 === 0 ? `Sure! Here it is:\n\`\`\`json\n${Json}\n\`\`\`` : Json;
    },
  };
}

function Context(Teacher: SftTeacher, Grounding: GroundingDoc[] = [], Shard = ParseShard("0/1")): SynthContext {
  return {
    Teacher,
    Budget: TestBudget,
    MeasureText: EstimateTokens,
    Slices: CodeSlices,
    Grounding: GroupGrounding(Grounding, Shard, CodeSlices),
    Tools: SynthToolReplay(),
    Verify: (Problem, Candidate) => RunAgainstTestsAsync(Problem, Candidate),
  };
}

function Item(Seed: number): SynthItem {
  return { Key: `test-${Seed}`, Index: Seed, Seed, Rng: new SeededRng(Seed) };
}

function Gate(EvalDocs: readonly string[] = TrainingDecontaminationDocs()): SynthAcceptance {
  return new SynthAcceptance({ EvalDocs, Measure: CreateDocumentMeasure(null), Budget: TestBudget, Replay: SynthToolReplay() });
}

// ---- tests ------------------------------------------------------------------------------------------

test("ExtractJsonObject finds the object in plain, fenced and chatty replies", () => {
  expect(ExtractJsonObject("{\"a\":1}")).toEqual({ a: 1 });
  expect(ExtractJsonObject("Here:\n```json\n{\"a\": \"}\"}\n```\nbye")).toEqual({ a: "}" });
  expect(ExtractJsonObject("I think {\"a\": {\"b\": [1, 2]}} is right")).toEqual({ a: { b: [1, 2] } });
  expect(ExtractJsonObject("no json here")).toBeNull();
  expect(ExtractJsonObject("[1, 2]")).toBeNull(); // an array is not an object
});

test("textbook unit is kept only when its solution passes its own tests", async () => {
  const Kept = await TextbookGenerator.Generate(Item(3), Context(FakeTeacher()));
  expect(Kept.Ok).toBe(true);
  if (Kept.Ok) {
    expect(Kept.Document.Messages.filter((M) => M.Role === "User")).toHaveLength(2); // the two-turn shape
    expect(JSON.stringify(Kept.Document)).toContain("addAll");
    expect((await Gate().Judge(Kept.Document)).Ok).toBe(true);
  }
  const Wrong = await TextbookGenerator.Generate(Item(3), Context(FakeTeacher({ Solution: WrongSolution })));
  expect(Wrong.Ok).toBe(false);
  if (!Wrong.Ok) expect(Wrong.Reason).toBe("tests-failed");
}, 30_000);

test("tool generator stores the REAL tool output, never the teacher's claim", async () => {
  const Teacher = FakeTeacher({ ClaimedResult: true });
  const Docs: ChatDocument[] = [];
  for (let Seed = 1; Seed <= 40; Seed++) {
    const Outcome = await ToolUseGenerator.Generate(Item(Seed), Context(Teacher));
    if (Outcome.Ok) Docs.push(Outcome.Document);
  }
  expect(Docs.length).toBeGreaterThan(10);
  let Calculators = 0;
  for (const Doc of Docs) {
    expect(await ReplayToolCalls(Doc, SynthToolReplay())).toEqual([]); // every stored result re-derives
    for (let I = 0; I < Doc.Messages.length; I++) {
      const Message = Doc.Messages[I];
      if (Message.Role !== "Tool") continue;
      expect(Message.Result).not.toEqual({ result: 999 });
      const Call = Doc.Messages[I - 1];
      if (Message.Name === "calculator" && Call.Role === "Assistant" && Call.ToolCall !== undefined) {
        Calculators++;
        const { a: A, op: Op, b: B } = Call.ToolCall.arguments as { a: number; op: string; b: number };
        const Expected = Op === "+" ? A + B : Op === "-" ? A - B : Op === "*" ? A * B : Op === "/" ? A / B : Op === "^" ? A ** B : A % B;
        expect(Message.Result).toEqual({ result: Expected });
      }
    }
  }
  expect(Calculators).toBeGreaterThan(0);
});

test("tool generator discards calls whose arguments differ from the intent", async () => {
  const Teacher = FakeTeacher({ CallOverride: (Args) => Object.fromEntries(Object.entries(Args).map(([K, V]) => [K, typeof V === "number" ? String(V) : V])) });
  const Reasons = new Set<string>();
  for (let Seed = 1; Seed <= 20; Seed++) {
    const Outcome = await ToolUseGenerator.Generate(Item(Seed), Context(Teacher));
    if (!Outcome.Ok) Reasons.add(Outcome.Reason);
  }
  expect(Reasons.has("call-mismatch")).toBe(true); // numbers sent as strings: the eval-measured failure
});

test("shard partition is disjoint and complete (documents and item indices)", () => {
  const Total = 4;
  const Ids = Array.from({ length: 300 }, (_V, I) => `doc-${I}`);
  const Owners = Ids.map((Id) => Array.from({ length: Total }, (_V, K) => K).filter((K) => InShard(Id, { Index: K, Total })));
  expect(Owners.every((List) => List.length === 1)).toBe(true); // each id in exactly one shard
  expect(new Set(Ids.map((Id) => ShardOf(Id, Total))).size).toBe(Total); // every shard gets work
  const Docs: GroundingDoc[] = Ids.map((Id) => ({ Id, Slice: "ts", Lang: "typescript", License: "MIT", Source: "s", Path: Id, Content: "x" }));
  const Union = Array.from({ length: Total }, (_V, K) => GroupGrounding(Docs, { Index: K, Total }, ["ts"]).get("ts") ?? []).flat().map((D) => D.Id);
  expect(Union.sort()).toEqual([...Ids].sort());
  const Indices = Array.from({ length: Total }, (_V, K) => Array.from({ length: 10 }, (_W, L) => ShardItemIndex({ Index: K, Total }, L))).flat();
  expect(new Set(Indices).size).toBe(Indices.length);
  expect([...Indices].sort((A, B) => A - B)).toEqual(Array.from({ length: 40 }, (_V, I) => I));
  expect(() => ParseShard("8/8")).toThrow();
  expect(ParseShard("3/8")).toEqual({ Index: 3, Total: 8 });
});

test("decontamination drops a conversation that contains an eval reference", async () => {
  const Reference = [...EvalCaseSetV1.Code].sort((A, B) => B.Reference.length - A.Reference.length)[0].Reference;
  const Envelope = { Generator: "grounded" as const, Teacher: "fake-teacher", SourceId: "x", Slice: "ts" as const };
  const Leaky = BuildSynthDocument(Envelope, [{ Role: "User", Content: "How do I compare two values deeply?" }, AnswerTurn("Recursive compare.", `\`\`\`ts\n${Reference}\n\`\`\``)]);
  const Verdict = await Gate().Judge(Leaky);
  expect(Verdict.Ok).toBe(false);
  if (!Verdict.Ok) expect(Verdict.Reason).toBe("contaminated");
  const Clean = BuildSynthDocument(Envelope, [{ Role: "User", Content: "How do I compare two numbers?" }, AnswerTurn("Simple.", "Use `a === b`.")]);
  expect((await Gate().Judge(Clean)).Ok).toBe(true);
});

test("mix parsing and quotas: unavailable generators' shares are redistributed", () => {
  const Mix = ParseMix("grounded:0.4,textbook:0.2,tools:0.25,multiturn:0.15");
  const All = PlanQuotas(100, Mix, new Set(["grounded", "textbook", "tools", "multiturn"]));
  expect(All).toEqual({ grounded: 40, textbook: 20, tools: 25, multiturn: 15 });
  const NoGrounding = PlanQuotas(10, Mix, new Set(["textbook", "tools", "multiturn"]));
  expect(Object.values(NoGrounding).reduce((S, V) => S + V, 0)).toBe(10);
  expect(NoGrounding.grounded).toBe(0);
  expect(() => ParseMix("bogus:1")).toThrow();
  expect(() => ParseMix("tools:-1")).toThrow();
});

test("--Corpus grounding works without a database: verbatim snippets from training-side files", async () => {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-sft-corpus-"));
  TempDirs.push(Dir);
  mkdirSync(join(Dir, "src"));
  for (let I = 0; I < 12; I++) {
    writeFileSync(join(Dir, "src", `Module${I}.ts`), [
      `// Module ${I}: small pricing helpers used by the checkout flow.`,
      `export function priceWithTax${I}(cents: number, rate: number): number {`,
      "  if (!Number.isFinite(cents) || cents < 0) throw new Error(\"invalid price\");",
      `  return Math.round(cents * (1 + rate) + ${I});`,
      "}",
      "",
      `export function formatPrice${I}(cents: number): string {`,
      "  const whole = Math.floor(cents / 100);",
      "  const rest = String(cents % 100).padStart(2, \"0\");",
      "  return `$${whole}.${rest}`;",
      "}",
      "",
    ].join("\n"));
  }
  const Docs = await LoadCorpusGrounding(Dir, Spec, "MIT");
  expect(Docs.length).toBeGreaterThan(0);
  expect(Docs.every((D) => D.Slice === "ts" && D.License === "MIT")).toBe(true);
  const Records: SynthRecord[] = [];
  const Stats = await RunSynthesis({
    Count: 3, Mix: ParseMix("grounded:1"), Shard: ParseShard("0/1"), Seed: 7,
    Context: Context(FakeTeacher(), Docs), Gate: Gate(), Generators: [GroundedCodeGenerator],
    Sink: (Record) => { Records.push(Record); },
  });
  expect(Stats.StopReason).toBe("count-reached");
  expect(Records).toHaveLength(3);
  const Ids = new Set(Docs.map((D) => D.Id));
  for (const Record of Records) {
    expect(Ids.has(Record.SourceId)).toBe(true);
    expect(Record.Provenance).toBe(`synthetic:grounded:fake-teacher:${Record.SourceId}`);
    const Source = Docs.find((D) => D.Id === Record.SourceId)!;
    const User = Record.Document.Messages.find((M) => M.Role === "User")!;
    const Snippet = /```ts\n([\s\S]*?)\n```/.exec(User.Role === "User" ? User.Content : "")![1];
    expect(Source.Content).toContain(Snippet); // grounded VERBATIM in the real file
  }
});

test("runner stops cleanly at its deadline and counts rejections by reason", async () => {
  let Clock = 0;
  const Stats = await RunSynthesis({
    Count: 50, Mix: ParseMix("tools:1"), Shard: ParseShard("1/2"), Seed: 1,
    Context: Context(FakeTeacher({ CallOverride: () => ({}) })), Gate: Gate(), Generators: [ToolUseGenerator],
    Sink: () => {}, DeadlineMs: 5, Now: () => Clock++,
  });
  expect(Stats.StopReason).toBe("deadline");
  expect(Stats.Shard).toBe("1/2");
  expect(Stats.Generators["tools"].Accepted).toBe(0);
  expect(Object.keys(Stats.Generators["tools"].Rejected).length).toBeGreaterThan(0);
});
