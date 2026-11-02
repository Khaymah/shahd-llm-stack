// Verified textbook units (the phi-1 / CodeTextbook recipe, Plan §2.3 item 1) for the TS/JS slices: the
// teacher writes a concept explanation + example + exercise + solution + test statements, and the unit is
// KEPT ONLY when its solution passes its own tests through the nonce harness (RunAgainstTestsAsync — the
// same reward-hack-resistant verifier the eval and STaR use). A unit whose tests are empty or contain no
// assertion is rejected before running: a test that asserts nothing passes any solution.
//
// Stored as a two-turn conversation: (1) "explain X with an example, then give me an exercise" -> the
// lesson; (2) "show me the solution" -> the verified solution and its tests.

import { z } from "zod";
import { AskTeacherJson } from "./TeacherJson.ts";
import { AnswerTurn, BuildSynthDocument, Fenced, SliceLanguageName, ThinkingTooLong } from "./SynthConversation.ts";
import { PickOne } from "./SynthSampling.ts";
import type { CodeSlice, SynthContext, SynthGeneratorImpl, SynthItem, SynthOutcome } from "./SynthTypes.ts";
import { Rejected, UngroundedSourceLicense } from "./SynthTypes.ts";

/** The slices whose solutions the harness can execute (Bun runs TS and JS). */
const ExecutableSlices: readonly CodeSlice[] = ["ts", "js"];

/** The concepts units are written about — small, testable, core TS/JS (the M-A curriculum stage). */
export const TextbookTopics: readonly string[] = [
  "optional chaining", "nullish coalescing", "Array.prototype.reduce", "closures", "generic functions",
  "type narrowing with typeof", "discriminated unions", "Map versus plain objects", "Set for de-duplication",
  "destructuring with default values", "rest parameters", "template literals", "Promise.all",
  "async/await error handling", "String.prototype.padStart", "building ranges with Array.from",
  "sorting with a comparator", "immutable updates with spread", "keyof and indexed access types",
  "readonly arrays", "union literal types instead of enums", "binary search", "memoization", "recursion",
  "regular expression basics", "JSON.parse and JSON.stringify", "class getters and setters",
  "custom error subclasses", "generator functions", "Object.entries and Object.fromEntries",
  "flat and flatMap", "chunking an array", "counting word frequencies", "shallow equality of objects",
  "clamping numbers", "parsing query strings", "palindromes and string reversal", "user-defined type guards",
  "Array.prototype.every and some", "grouping items by a key",
];

const LessonAsks: readonly string[] = [
  "Explain {topic} in {lang} with a short example, then give me an exercise.",
  "Can you teach me {topic} in {lang}? A small example and one practice exercise please.",
  "i keep getting confused by {topic} in {lang}. explain it and give me something to practice",
  "Give me a quick lesson on {topic} ({lang}) with an example and an exercise.",
];

const SolutionAsks: readonly string[] = [
  "Show me the solution.",
  "Can you show a solution with tests?",
  "ok I tried it, what's the answer?",
  "Please give the reference solution and how to test it.",
];

const TextbookReply = z.object({
  thinking_lesson: z.string().min(1),
  explanation: z.string().min(20),
  example: z.string().min(10),
  exercise: z.string().min(10),
  thinking_solution: z.string().min(1),
  solution: z.string().min(10),
  tests: z.string().min(10),
});

const AssertCall = /\bassert\s*\(/;
/** The teacher may use this multiple of the budget for its JSON reply. */
const TeacherTokenFactor = 2;

function BuildPrompt(Topic: string, Slice: CodeSlice): string {
  const Lang = SliceLanguageName[Slice];
  const Typing = Slice === "ts" ? "TypeScript with type annotations" : "plain modern JavaScript (no type annotations)";
  return [
    `Write a short ${Lang} textbook unit about: ${Topic}.`,
    "Fields:",
    "- \"thinking_lesson\": Shahd's private plan for the lesson, one short sentence.",
    "- \"explanation\": the concept in at most 60 words.",
    `- "example": a ${Typing} example, at most 8 lines, no imports.`,
    "- \"exercise\": one sentence asking to implement ONE named function (name it explicitly).",
    "- \"thinking_solution\": Shahd's private plan for the solution, one short sentence.",
    `- "solution": ${Typing} defining that function, at most 15 lines, no imports, no exports, no console output.`,
    "- \"tests\": 2 to 4 statements of the form assert(condition, \"message\") that call the function; no imports, no test framework.",
    "Reply as one JSON object with exactly those keys.",
  ].join("\n");
}

async function Generate(Item: SynthItem, Context: SynthContext): Promise<SynthOutcome> {
  const Slices = ExecutableSlices.filter((Slice) => Context.Slices.includes(Slice));
  if (Slices.length === 0) return Rejected("no-executable-slice", "textbook units need the ts or js slice");
  const Slice = PickOne(Item.Rng, Slices);
  const Topic = PickOne(Item.Rng, TextbookTopics);
  const Reply = await AskTeacherJson(Context.Teacher, { Prompt: BuildPrompt(Topic, Slice), MaxTokens: Context.Budget * TeacherTokenFactor, Seed: Item.Seed }, TextbookReply);
  if (!Reply.Ok) return Rejected(Reply.Reason, Reply.Detail);
  const Unit = Reply.Value;
  const Long = ThinkingTooLong(Unit.thinking_lesson, Unit.thinking_solution);
  if (Long !== null) return Rejected("thinking-too-long", Long);
  if (!AssertCall.test(Unit.tests)) return Rejected("tests-assert-nothing", "the tests contain no assert(...) call");
  const Passed = await Context.Verify({ Name: Item.Key, Tests: Unit.tests }, Unit.solution);
  if (!Passed) return Rejected("tests-failed", `${Topic}: the solution does not pass its tests`);

  const Lang = SliceLanguageName[Slice];
  const Lesson = `${Unit.explanation.trim()}\n\n${Fenced(Slice, Unit.example)}\n\n**Exercise:** ${Unit.exercise.trim()}`;
  const Solution = `${Fenced(Slice, Unit.solution)}\n\nTests:\n\n${Fenced(Slice, Unit.tests)}`;
  const Document = BuildSynthDocument(
    { Generator: "textbook", Teacher: Context.Teacher.Model, SourceId: Item.Key, Slice, Task: Topic },
    [
      { Role: "User", Content: PickOne(Item.Rng, LessonAsks).replace("{topic}", Topic).replace("{lang}", Lang) },
      AnswerTurn(Unit.thinking_lesson, Lesson),
      { Role: "User", Content: PickOne(Item.Rng, SolutionAsks) },
      AnswerTurn(Unit.thinking_solution, Solution),
    ],
  );
  return { Ok: true, Document, Source: { Id: Item.Key, License: UngroundedSourceLicense, Name: "textbook" }, Slice };
}

export const TextbookGenerator: SynthGeneratorImpl = {
  Name: "textbook",
  Available: (Context) => ExecutableSlices.some((Slice) => Context.Slices.includes(Slice)),
  Generate,
};
