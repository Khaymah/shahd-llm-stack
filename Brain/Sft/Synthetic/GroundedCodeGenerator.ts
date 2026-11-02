// Grounded code conversations: a REAL permissive document from the corpus (training side of the split
// only — the caller loads it that way) is the subject; the teacher writes a realistic developer question
// about it for one task from TaskTaxonomy, plus Shahd's short scratchpad and answer. The snippet in the
// user turn is attached by the pipeline VERBATIM from the corpus, never retyped by the teacher, so the
// conversation is anchored to code that exists (the grounding that keeps a teacher from hallucinating).
// The snippet is bounded so the rendered conversation fits the token budget.

import { z } from "zod";
import type { CodeTask } from "../TaskTaxonomy.ts";
import { CodeTasks, TaskInstructions } from "../TaskTaxonomy.ts";
import { AskTeacherJson } from "./TeacherJson.ts";
import { BoundSnippet } from "./SynthBudget.ts";
import { AnswerTurn, BuildSynthDocument, Fenced, SliceLanguageName, ThinkingTooLong } from "./SynthConversation.ts";
import { PickInt, PickOne, UserStyles } from "./SynthSampling.ts";
import type { CodeSlice, GroundingDoc, SynthContext, SynthGeneratorImpl, SynthItem, SynthOutcome } from "./SynthTypes.ts";
import { Rejected } from "./SynthTypes.ts";

/** Share of the budget the grounding snippet may use (the rest is question, thinking and answer). */
const SnippetShare = 0.4;
/** Tokens reserved for role markers, the system prompt and the question around the snippet. */
const OverheadTokens = 60;
/** Words per answer token handed to the teacher as a length target (prose ~0.75, code less). */
const WordsPerToken = 0.6;
const MinAnswerWords = 30;
const MaxAnswerWords = 200;
/** A snippet smaller than this is not worth a conversation. */
const MinSnippetChars = 80;
const MinSnippetLines = 3;
/** Chance the snippet starts at the top of the file (imports + first declarations); else anywhere in
 *  the first SnippetStartSpan of the file. */
const TopStartChance = 0.4;
const SnippetStartSpan = 0.7;
/** The teacher may use this multiple of the budget for its JSON reply (keys, escaping). */
const TeacherTokenFactor = 2;

// Tasks that make sense per slice (markup / stylesheets / SQL have no unit tests or translation target).
const TasksBySlice: Record<CodeSlice, readonly CodeTask[]> = {
  ts: CodeTasks,
  js: CodeTasks,
  go: CodeTasks,
  html: ["BugFix", "DocGeneration", "CodeReview", "Refactor", "Explain", "CommitMessage"],
  css: ["BugFix", "DocGeneration", "CodeReview", "Refactor", "Explain", "CommitMessage"],
  sql: ["BugFix", "DocGeneration", "CodeReview", "Refactor", "Explain", "CommitMessage"],
};

// Tasks whose answer must contain code.
const CodeAnswerTasks: ReadonlySet<CodeTask> = new Set<CodeTask>(["BugFix", "TestGeneration", "DocGeneration", "Refactor", "Translate"]);

const TranslateTarget: Partial<Record<CodeSlice, CodeSlice>> = { ts: "js", js: "ts", go: "ts" };

const GroundedReply = z.object({
  question: z.string().min(8),
  thinking: z.string().min(1),
  answer: z.string().min(1),
});

/** The first top-level (unindented, non-blank) line at or after Start, so a snippet opens on a whole
 *  declaration instead of mid-function; Start itself when there is none. */
function TopLevelLineFrom(Lines: readonly string[], Start: number): number {
  for (let I = Start; I < Lines.length; I++) {
    if (Lines[I].trim() !== "" && !/^\s/.test(Lines[I])) return I;
  }
  return Start;
}

/** The slices that have grounding documents in this shard. */
function GroundedSlices(Context: SynthContext): CodeSlice[] {
  return Context.Slices.filter((Slice) => (Context.Grounding.get(Slice)?.length ?? 0) > 0);
}

function TaskLine(Task: CodeTask, Slice: CodeSlice): string {
  const Target = TranslateTarget[Slice];
  const Suffix = Task === "Translate" && Target !== undefined ? ` Target language: ${SliceLanguageName[Target]}.` : "";
  const Diff = Task === "CommitMessage" ? " Treat the snippet as a change that was just added." : "";
  return `${Task} — ${TaskInstructions[Task]}${Suffix}${Diff}`;
}

function BuildPrompt(Doc: GroundingDoc, Snippet: string, Task: CodeTask, Style: string, AnswerWords: number): string {
  return [
    `Below is a real ${SliceLanguageName[Doc.Slice]} snippet from ${Doc.Source} (${Doc.Path}).`,
    "",
    Fenced(Doc.Slice, Snippet),
    "",
    `Write ONE training example for this task: ${TaskLine(Task, Doc.Slice)}`,
    "Fields:",
    `- "question": what a developer would ask Shahd about this snippet for that task, written ${Style}. One to three sentences. Do NOT include the code; it is attached after the question automatically.`,
    "- \"thinking\": Shahd's private plan before answering: one or two short sentences, under 300 characters.",
    `- "answer": Shahd's reply, at most ${AnswerWords} words. Put any code in Markdown fences with a language tag. Only refer to what is actually in the snippet.`,
    "Reply as {\"question\": \"...\", \"thinking\": \"...\", \"answer\": \"...\"}",
  ].join("\n");
}

async function Generate(Item: SynthItem, Context: SynthContext): Promise<SynthOutcome> {
  const Slices = GroundedSlices(Context);
  if (Slices.length === 0) return Rejected("no-grounding", "no grounding documents in this shard for the requested slices");
  const Slice = PickOne(Item.Rng, Slices);
  const Doc = PickOne(Item.Rng, Context.Grounding.get(Slice) ?? []);
  const Task = PickOne(Item.Rng, TasksBySlice[Slice]);
  const Style = PickOne(Item.Rng, UserStyles);

  const Lines = Doc.Content.split("\n");
  const Drawn = Item.Rng.NextFloat() < TopStartChance ? 0 : PickInt(Item.Rng, 0, Math.floor(Lines.length * SnippetStartSpan));
  const StartLine = TopLevelLineFrom(Lines, Drawn);
  const SnippetTokens = Math.floor(Context.Budget * SnippetShare);
  const Snippet = BoundSnippet(Doc.Content, StartLine, SnippetTokens, Context.MeasureText);
  if (Snippet.length < MinSnippetChars || Snippet.split("\n").length < MinSnippetLines) return Rejected("snippet-too-small", `${Snippet.length} chars`);

  const AnswerTokens = Context.Budget - Context.MeasureText(Snippet) - OverheadTokens;
  const AnswerWords = Math.max(MinAnswerWords, Math.min(MaxAnswerWords, Math.floor(AnswerTokens * WordsPerToken)));
  const Reply = await AskTeacherJson(Context.Teacher, { Prompt: BuildPrompt(Doc, Snippet, Task, Style, AnswerWords), MaxTokens: Context.Budget * TeacherTokenFactor, Seed: Item.Seed }, GroundedReply);
  if (!Reply.Ok) return Rejected(Reply.Reason, Reply.Detail);
  const { question: Question, thinking: Thinking, answer: Answer } = Reply.Value;
  if (Question.includes("```")) return Rejected("question-has-code", "the question pasted code instead of leaving it to the attachment");
  if (CodeAnswerTasks.has(Task) && !Answer.includes("```")) return Rejected("no-code-fence", `${Task} answer has no fenced code`);
  const Long = ThinkingTooLong(Thinking);
  if (Long !== null) return Rejected("thinking-too-long", Long);

  const Document = BuildSynthDocument(
    { Generator: "grounded", Teacher: Context.Teacher.Model, SourceId: Doc.Id, Slice, Task },
    [
      { Role: "User", Content: `${Question.trim()}\n\n${Fenced(Slice, Snippet)}` },
      AnswerTurn(Thinking, Answer),
    ],
  );
  return { Ok: true, Document, Source: { Id: Doc.Id, License: Doc.License, Name: Doc.Source }, Slice };
}

export const GroundedCodeGenerator: SynthGeneratorImpl = {
  Name: "grounded",
  Available: (Context) => GroundedSlices(Context).length > 0,
  Generate,
};
