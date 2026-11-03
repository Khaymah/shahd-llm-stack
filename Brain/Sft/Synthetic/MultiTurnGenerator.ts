// Multi-turn conversations where a LATER turn depends on an EARLIER one — the behavior the eval's
// multiturn.coherence measures and a single-exchange corpus never teaches. The pipeline picks a pattern
// and its facts (old/new names, the numbers given, the person's name …); the teacher writes the turns;
// a pattern-specific check verifies the dependency actually holds (the renamed function no longer uses
// the old name, the recalled number is the right one and the question did not restate it, …).

import { z } from "zod";
import type { EvalSlice } from "../../Eval/Suite/SuiteTypes.ts";
import type { SeededRng } from "../../Random/SeededRng.ts";
import { AskTeacherJson } from "./TeacherJson.ts";
import { AnswerTurn, BuildSynthDocument, SliceLanguageName, ThinkingTooLong } from "./SynthConversation.ts";
import { PickInt, PickOne, UserStyles } from "./SynthSampling.ts";
import { MentionsNumber } from "./ToolIntents.ts";
import type { CodeSlice, SynthContext, SynthGeneratorImpl, SynthItem, SynthOutcome } from "./SynthTypes.ts";
import { Rejected, UngroundedSourceLicense } from "./SynthTypes.ts";

type TurnReply = { user: string; thinking: string; answer: string };
type PatternPlan = { Turns: number; Instruction: string; Check: (Turns: readonly TurnReply[]) => string | null };
type Pattern = { Name: string; Slices: readonly EvalSlice[]; Plan: (Rng: SeededRng, Slice: EvalSlice) => PatternPlan };

const MultiTurnReply = z.object({
  turns: z.array(z.object({ user: z.string().min(1), thinking: z.string().min(1), answer: z.string().min(1) })).min(2),
});

/** Tokens the teacher may spend on the whole JSON reply (a multiple of the budget). */
const TeacherTokenFactor = 2;
const Fence = "```";

// Small functions with a natural first name and the name a user might ask for instead.
const Functions: readonly [string, string, string][] = [
  ["adds up an array of prices", "sumPrices", "total"],
  ["checks whether a string is a palindrome", "isPalindrome", "checkPalindrome"],
  ["returns the largest number in a list", "maxOf", "largest"],
  ["formats a price in cents as a dollar string", "formatCents", "toDollars"],
  ["counts the vowels in a string", "countVowels", "vowelCount"],
  ["removes duplicate values from an array", "unique", "dedupe"],
  ["capitalizes the first letter of each word", "titleCase", "capitalizeWords"],
  ["computes the average of a list of scores", "averageScore", "meanScore"],
];

const Fetchers: readonly [string, string][] = [
  ["fetches a user by id from /api/users/:id", "fetchUser"],
  ["loads the product list from /api/products", "loadProducts"],
  ["posts a comment to /api/comments", "postComment"],
  ["fetches the weather for a city from /api/weather?city=", "getWeather"],
];

const Params: readonly string[] = ["locale", "precision", "limit", "separator", "caseSensitive", "fallback"];
const People: readonly string[] = ["Layla", "Omar", "Sofia", "Kenji", "Amara", "Diego", "Noor", "Elena"];
const Projects: readonly string[] = ["a recipe app", "a budgeting tool", "a chess engine", "a weather dashboard", "a habit tracker", "an online bookstore"];
const Asides: readonly string[] = [
  "what const means in JavaScript", "the difference between == and ===", "what a Promise is", "what JSON stands for", "why people use TypeScript",
];

const NumberKinds: readonly [string, (Values: readonly number[]) => number][] = [
  ["first", (Values) => Values[0]],
  ["last", (Values) => Values[Values.length - 1]],
  ["largest", (Values) => Math.max(...Values)],
  ["smallest", (Values) => Math.min(...Values)],
  ["sum of the", (Values) => Values.reduce((Sum, Value) => Sum + Value, 0)],
];

function Lang(Slice: EvalSlice): string {
  return Slice === "general" ? "TypeScript" : SliceLanguageName[Slice as CodeSlice];
}

function Last(Turns: readonly TurnReply[]): TurnReply {
  return Turns[Turns.length - 1];
}

/** Null when every check holds, else the first failure. */
function FirstFailure(Checks: readonly [boolean, string][]): string | null {
  return Checks.find(([Holds]) => !Holds)?.[1] ?? null;
}

const Rename: Pattern = {
  Name: "rename",
  Slices: ["ts", "js", "go"],
  Plan: (Rng, Slice) => {
    const [Purpose, Old, New] = PickOne(Rng, Functions);
    return {
      Turns: 2,
      Instruction: `Turn 1: the user asks for a ${Lang(Slice)} function that ${Purpose}; Shahd answers with a function named \`${Old}\` in a fenced code block. ` +
        `Turn 2: the user asks to rename it to \`${New}\`; Shahd answers with the full renamed function in a fenced code block and no trace of the old name.`,
      Check: (Turns) => FirstFailure([
        [Turns[0].answer.includes(Old) && Turns[0].answer.includes(Fence), `turn 1 does not define ${Old} in code`],
        [Turns[1].user.includes(New), `the rename request does not name ${New}`],
        [Last(Turns).answer.includes(`${New}(`) && !Last(Turns).answer.includes(Old), `the final answer is not renamed to ${New}`],
      ]),
    };
  },
};

const MakeAsync: Pattern = {
  Name: "make-async",
  Slices: ["ts", "js"],
  Plan: (Rng, Slice) => {
    const [Purpose, Name] = PickOne(Rng, Fetchers);
    return {
      Turns: 2,
      Instruction: `Turn 1: the user asks for a ${Lang(Slice)} function \`${Name}\` that ${Purpose}; Shahd writes it with fetch and .then() chains (no async/await) in a fenced code block. ` +
        "Turn 2: the user asks to make it use async/await; Shahd rewrites the same function with async/await and no .then().",
      Check: (Turns) => FirstFailure([
        [Turns[0].answer.includes(".then(") && Turns[0].answer.includes(Name), "turn 1 is not a .then() version"],
        [Last(Turns).answer.includes("async") && Last(Turns).answer.includes("await") && Last(Turns).answer.includes(Name), "the final answer is not async/await"],
        [!Last(Turns).answer.includes(".then("), "the final answer still uses .then()"],
      ]),
    };
  },
};

const AddParameter: Pattern = {
  Name: "add-parameter",
  Slices: ["ts", "js", "go"],
  Plan: (Rng, Slice) => {
    const [Purpose, Name] = PickOne(Rng, Functions);
    const Param = PickOne(Rng, Params);
    return {
      Turns: 2,
      Instruction: `Turn 1: the user asks for a ${Lang(Slice)} function \`${Name}\` that ${Purpose}; Shahd answers in a fenced code block. ` +
        `Turn 2: the user asks to add a parameter \`${Param}\` (pick a sensible meaning); Shahd answers with the updated function.`,
      Check: (Turns) => FirstFailure([
        [Turns[0].answer.includes(Name), `turn 1 does not define ${Name}`],
        [Last(Turns).answer.includes(Name) && Last(Turns).answer.includes(Param), `the final answer lacks ${Param}`],
      ]),
    };
  },
};

const ConvertArrow: Pattern = {
  Name: "convert-arrow",
  Slices: ["ts", "js"],
  Plan: (Rng, Slice) => {
    const [Purpose, Name] = PickOne(Rng, Functions);
    return {
      Turns: 2,
      Instruction: `Turn 1: the user asks for a ${Lang(Slice)} function declaration \`function ${Name}\` that ${Purpose}. ` +
        `Turn 2: the user asks to turn it into an arrow function; Shahd answers with \`const ${Name} = (...) => ...\`.`,
      Check: (Turns) => FirstFailure([
        [Turns[0].answer.includes(`function ${Name}`), "turn 1 is not a function declaration"],
        [Last(Turns).answer.includes(`const ${Name}`) && Last(Turns).answer.includes("=>"), "the final answer is not an arrow function"],
      ]),
    };
  },
};

const RecallNumber: Pattern = {
  Name: "recall-number",
  Slices: ["general"],
  Plan: (Rng) => {
    const Values = [...new Set(Array.from({ length: PickInt(Rng, 3, 5) }, () => PickInt(Rng, 2, 999)))];
    const [Kind, Pick] = PickOne(Rng, NumberKinds);
    const Expected = Pick(Values);
    const Aside = Rng.NextFloat() < 0.5 ? PickOne(Rng, Asides) : null;
    return {
      Turns: Aside === null ? 2 : 3,
      Instruction: `Turn 1: the user gives Shahd these numbers in this order: ${Values.join(", ")}; Shahd acknowledges briefly and repeats them. ` +
        (Aside === null ? "" : `Turn 2: the user asks a short unrelated question about ${Aside}; Shahd answers briefly. `) +
        `Final turn: the user asks for the ${Kind} number(s) they gave, WITHOUT repeating any of the numbers; Shahd answers ${Expected}.`,
      Check: (Turns) => FirstFailure([
        [Values.every((Value) => MentionsNumber(Turns[0].user, Value)), "turn 1 does not give every number"],
        [!Values.some((Value) => MentionsNumber(Last(Turns).user, Value)), "the final question restates a number"],
        [MentionsNumber(Last(Turns).answer, Expected), `the final answer does not say ${Expected}`],
      ]),
    };
  },
};

const RecallFact: Pattern = {
  Name: "recall-fact",
  Slices: ["general"],
  Plan: (Rng) => {
    const Person = PickOne(Rng, People);
    const Project = PickOne(Rng, Projects);
    const AskName = Rng.NextFloat() < 0.5;
    const Target = AskName ? Person : Project.replace(/^an? /, "");
    return {
      Turns: 2,
      Instruction: `Turn 1: the user introduces themselves as ${Person} and says they are building ${Project}; Shahd greets them by name and offers help. ` +
        `Turn 2: the user asks what ${AskName ? "their name is" : "they said they are building"}, without repeating it; Shahd answers with "${Target}".`,
      Check: (Turns) => FirstFailure([
        [Turns[0].user.includes(Person) && Turns[0].user.includes(Project.replace(/^an? /, "")), "turn 1 does not state the facts"],
        [!Last(Turns).user.includes(Target), "the final question restates the answer"],
        [Last(Turns).answer.toLowerCase().includes(Target.toLowerCase()), `the final answer does not say ${Target}`],
      ]),
    };
  },
};

const Patterns: readonly Pattern[] = [Rename, MakeAsync, AddParameter, ConvertArrow, RecallNumber, RecallFact];

function Applicable(Context: SynthContext): { Pattern: Pattern; Slices: EvalSlice[] }[] {
  return Patterns
    .map((Pattern) => ({ Pattern, Slices: Pattern.Slices.filter((Slice) => Slice === "general" || Context.Slices.includes(Slice as CodeSlice)) }))
    .filter((Entry) => Entry.Slices.length > 0);
}

function BuildPrompt(Plan: PatternPlan, Style: string): string {
  return [
    `Write a ${Plan.Turns}-turn conversation between a user and Shahd. The user writes ${Style}.`,
    Plan.Instruction,
    "Every Shahd answer is short (at most 80 words) and consistent with the earlier turns. Code goes in Markdown fences with a language tag.",
    "For each turn give \"user\", \"thinking\" (Shahd's one-sentence private plan) and \"answer\".",
    `Reply as {"turns": [${Array.from({ length: Plan.Turns }, () => "{\"user\": \"...\", \"thinking\": \"...\", \"answer\": \"...\"}").join(", ")}]}`,
  ].join("\n");
}

async function Generate(Item: SynthItem, Context: SynthContext): Promise<SynthOutcome> {
  const { Pattern, Slices } = PickOne(Item.Rng, Applicable(Context));
  const Slice = PickOne(Item.Rng, Slices);
  const Plan = Pattern.Plan(Item.Rng, Slice);
  const Style = PickOne(Item.Rng, UserStyles);
  const Reply = await AskTeacherJson(Context.Teacher, { Prompt: BuildPrompt(Plan, Style), MaxTokens: Context.Budget * TeacherTokenFactor, Seed: Item.Seed }, MultiTurnReply);
  if (!Reply.Ok) return Rejected(Reply.Reason, Reply.Detail);
  const Turns = Reply.Value.turns;
  if (Turns.length !== Plan.Turns) return Rejected("turn-count", `${Turns.length} turns for a ${Plan.Turns}-turn plan`);
  const Failure = Plan.Check(Turns);
  if (Failure !== null) return Rejected("dependency-check", `${Pattern.Name}: ${Failure}`);
  const Long = ThinkingTooLong(...Turns.map((Turn) => Turn.thinking));
  if (Long !== null) return Rejected("thinking-too-long", Long);
  const Document = BuildSynthDocument(
    { Generator: "multiturn", Teacher: Context.Teacher.Model, SourceId: Item.Key, Slice, Task: Pattern.Name },
    Turns.flatMap((Turn) => [{ Role: "User" as const, Content: Turn.user.trim() }, AnswerTurn(Turn.thinking, Turn.answer)]),
  );
  return { Ok: true, Document, Source: { Id: Item.Key, License: UngroundedSourceLicense, Name: "multiturn" }, Slice };
}

export const MultiTurnGenerator: SynthGeneratorImpl = {
  Name: "multiturn",
  Available: (Context) => Applicable(Context).length > 0,
  Generate,
};
