// Grounded tool use. The pipeline samples the intent (ToolIntents.ts: tool + exact arguments), the teacher
// phrases the user's request(s) and Shahd's call scratchpad, the pipeline EXECUTES the real tool from the
// synthesis registry (fixture workspace, fresh memory, a per-conversation clock), and the teacher writes
// the final answer from the REAL result it is shown. The stored Tool message is always the executed
// output — whatever the teacher may claim — and the replay in the acceptance gate re-derives it.
//
// Rejected: a call whose name/arguments differ from the intent (the argument-accuracy target), a request
// that does not state the numbers it asks about, a recall request that restates the value, a tool that
// errors (failed argument validation), and an answer that does not echo the real result.

import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
import type { ChatDocumentMessage } from "../ChatDocument.ts";
import { AskTeacherJson } from "./TeacherJson.ts";
import { AnswerTurn, BuildSynthDocument, CallTurn, ThinkingTooLong } from "./SynthConversation.ts";
import { PickInt, PickOne, UserStyles } from "./SynthSampling.ts";
import type { ToolPlan, ToolStepPlan, WorkspaceView } from "./ToolIntents.ts";
import { AnswerEchoesResult, MentionsNumber, PlanToolConversation, ReadWorkspaceView } from "./ToolIntents.ts";
import type { SynthContext, SynthGeneratorImpl, SynthItem, SynthOutcome } from "./SynthTypes.ts";
import { Rejected, UngroundedSourceLicense } from "./SynthTypes.ts";

// The clock range a conversation's current_time may report (2024-01-01 .. 2027-01-01, whole seconds).
const ClockFromMs = 1_704_067_200_000;
const ClockToMs = 1_798_761_600_000;
const MsPerSecond = 1000;
/** Max words of a final answer (tool answers are short). */
const AnswerWords = 60;
/** Tokens the teacher may spend per reply. */
const TeacherMaxTokens = 600;

const StepsReply = z.object({
  steps: z.array(z.object({
    user: z.string().min(2),
    thinking: z.string().min(1),
    call: z.object({ name: z.string().min(1), arguments: z.record(z.string(), z.unknown()) }),
  })).min(1),
});

const AnswerReply = z.object({ thinking: z.string().min(1), answer: z.string().min(1) });

const ViewCache = new Map<string, WorkspaceView>();

function WorkspaceViewFor(Root: string): WorkspaceView {
  let View = ViewCache.get(Root);
  if (View === undefined) {
    View = ReadWorkspaceView(Root);
    ViewCache.set(Root, View);
  }
  return View;
}

function Manifest(Context: SynthContext): string {
  return Context.Tools.Registry.List().map((Tool) => `- ${Tool.Name} ${Tool.Args ?? "{}"}: ${Tool.Description}`).join("\n");
}

function StepsPrompt(Plan: ToolPlan, Style: string, Context: SynthContext): string {
  const Requests = Plan.Steps.map((Step, Index) =>
    `Request ${Index + 1}: the user wants ${Step.Want}. Shahd's call must be exactly ${JSON.stringify({ name: Step.Tool, arguments: Step.Arguments })}.`);
  return [
    "Write the user side of a tool-use training conversation for Shahd, who has these tools:",
    Manifest(Context),
    "",
    ...Requests,
    "",
    `For each request write "user" (the user's message, ${Style}; state every value the user is giving, digits as digits; never mention tool names or JSON), ` +
      "\"thinking\" (Shahd's one-sentence plan to use the tool) and \"call\" (copy the exact call above).",
    Plan.Steps.length > 1 ? "Later requests happen later in the same chat and may refer back to earlier ones." : "",
    "Reply as {\"steps\": [{\"user\": \"...\", \"thinking\": \"...\", \"call\": {\"name\": \"...\", \"arguments\": {}}}]}",
  ].join("\n");
}

function AnswerPrompt(User: string, Step: ToolStepPlan, Result: Record<string, unknown>): string {
  return [
    `The user wrote: ${JSON.stringify(User)}`,
    `Shahd called the tool ${Step.Tool} with ${JSON.stringify(Step.Arguments)} and it returned exactly: ${JSON.stringify(Result)}`,
    "Write Shahd's reply based ONLY on that returned result.",
    `- "thinking": one short sentence reading the result.`,
    `- "answer": the reply to the user, at most ${AnswerWords} words, using the returned values exactly.`,
    "Reply as {\"thinking\": \"...\", \"answer\": \"...\"}",
  ].join("\n");
}

/** Why the teacher's request/call for a planned step is unusable, or null when it matches the intent. */
function StepMismatch(Step: ToolStepPlan, User: string, Call: { name: string; arguments: Record<string, unknown> }): { Reason: string; Detail: string } | null {
  if (Call.name !== Step.Tool) return { Reason: "wrong-tool", Detail: `${Call.name} instead of ${Step.Tool}` };
  if (!isDeepStrictEqual(Call.arguments, Step.Arguments)) return { Reason: "call-mismatch", Detail: `${JSON.stringify(Call.arguments)} instead of ${JSON.stringify(Step.Arguments)}` };
  const Missing = Step.Numbers.filter((Value) => !MentionsNumber(User, Value));
  if (Missing.length > 0) return { Reason: "arguments-not-grounded", Detail: `the request never states ${Missing.join(", ")}` };
  const Leaked = Step.Hidden.filter((Text) => User.includes(Text));
  if (Leaked.length > 0) return { Reason: "recall-restates-value", Detail: Leaked.join(", ") };
  return null;
}

async function Generate(Item: SynthItem, Context: SynthContext): Promise<SynthOutcome> {
  const ClockMs = PickInt(Item.Rng, ClockFromMs / MsPerSecond, ClockToMs / MsPerSecond) * MsPerSecond;
  const Envelope = { Generator: "tools" as const, Teacher: Context.Teacher.Model, SourceId: Item.Key, Slice: "general" as const, ClockMs };
  // One context for the whole conversation (memory persists across its steps), built by the SAME factory
  // the replay uses, from the document's own Meta.
  const ToolContext = Context.Tools.Context(BuildSynthDocument(Envelope, []));
  const Root = ToolContext.Workspace?.Root;
  if (Root === undefined) return Rejected("no-workspace", "the synthesis tool context has no workspace");
  const Plan = PlanToolConversation(Item.Rng, WorkspaceViewFor(Root));
  const Style = PickOne(Item.Rng, UserStyles);

  const Steps = await AskTeacherJson(Context.Teacher, { Prompt: StepsPrompt(Plan, Style, Context), MaxTokens: TeacherMaxTokens, Seed: Item.Seed }, StepsReply);
  if (!Steps.Ok) return Rejected(Steps.Reason, Steps.Detail);
  if (Steps.Value.steps.length !== Plan.Steps.length) return Rejected("step-count", `${Steps.Value.steps.length} steps for a ${Plan.Steps.length}-step plan`);

  const Turns: ChatDocumentMessage[] = [];
  for (let Index = 0; Index < Plan.Steps.length; Index++) {
    const Step = Plan.Steps[Index];
    const Written = Steps.Value.steps[Index];
    const Mismatch = StepMismatch(Step, Written.user, Written.call);
    if (Mismatch !== null) return Rejected(Mismatch.Reason, Mismatch.Detail);
    // The REAL execution — the registry's own argument parsing; an error result discards the call.
    const Result = await Context.Tools.Registry.Run({ Name: Step.Tool, Arguments: Step.Arguments }, ToolContext);
    if ("error" in Result) return Rejected("tool-error", `${Step.Tool}: ${String(Result["error"])}`);
    const Answer = await AskTeacherJson(Context.Teacher, { Prompt: AnswerPrompt(Written.user, Step, Result), MaxTokens: TeacherMaxTokens, Seed: Item.Seed + Index + 1 }, AnswerReply);
    if (!Answer.Ok) return Rejected(Answer.Reason, Answer.Detail);
    if (!AnswerEchoesResult(Answer.Value.answer, Step.Tool, Result)) return Rejected("answer-ignores-result", `${Step.Tool} returned ${JSON.stringify(Result)}`);
    const Long = ThinkingTooLong(Written.thinking, Answer.Value.thinking);
    if (Long !== null) return Rejected("thinking-too-long", Long);
    Turns.push(
      { Role: "User", Content: Written.user.trim() },
      CallTurn(Written.thinking, { name: Step.Tool, arguments: Step.Arguments }),
      { Role: "Tool", Name: Step.Tool, Result: JSON.parse(JSON.stringify(Result)) as Record<string, unknown> },
      AnswerTurn(Answer.Value.thinking, Answer.Value.answer),
    );
  }
  const Document = BuildSynthDocument({ ...Envelope, Task: Plan.Kind }, Turns);
  return { Ok: true, Document, Source: { Id: Item.Key, License: UngroundedSourceLicense, Name: "tools" }, Slice: "general" };
}

export const ToolUseGenerator: SynthGeneratorImpl = {
  Name: "tools",
  Available: () => true,
  Generate,
};
