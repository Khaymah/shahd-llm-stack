// How the suite talks to a model — one home for: the generator seam (the REAL guarded serving
// generator by default, an injected stub in tests), building a chat session from case messages,
// reading ONE generated turn into the facts every metric judges (visible answer, how the turn ended,
// thinking closed, leaked control text, tool calls), base-model plain completion, per-case deadlines
// and the per-sample seeding that makes every number reproducible.
//
// Every judgement on a special-token model is made on SPECIAL IDS, exactly as the agent loop serves
// it (TruncateAtTurnEnd / SplitThinkingIds / ParseToolCallIds), never by re-parsing decoded text: a
// model that SPELLS "<|tool_call|>" in ordinary byte tokens is leaking control text, not calling a tool.

import type { RunnableModel } from "../../Checkpoint/LoadRunnableModel.ts";
import type { SamplingOptions } from "../../Sampling/Sampler.ts";
import { DefaultSampling } from "../../Sampling/Sampler.ts";
import { CreateRngStreams } from "../../Random/SeededRng.ts";
import type { SeededRng } from "../../Random/SeededRng.ts";
import type { Tokenizer } from "../../Tokenizer/TokenizerTypes.ts";
import { SpecialTokenizer } from "../../Tokenizer/SpecialTokenizer.ts";
import { SpecialTokens } from "../../Tokenizer/SpecialTokens.ts";
import { GuardedGenerateIdsStream } from "../../Safety/GuardedGenerate.ts";
import type { GuardedStreamOptions } from "../../Safety/GuardedGenerate.ts";
import { SafetyBlockedError } from "../../Safety/SafetyPolicy.ts";
import { ChatSession } from "../../Serving/ChatSession.ts";
import type { Generator, GeneratedTurn } from "../../Serving/AgentLoop.ts";
import { CreateModelGenerator } from "../../Serving/ModelGenerator.ts";
import { ParseToolCall, ParseToolCallIds, ToolTokens, ToolTokenList } from "../../Serving/ToolProtocol.ts";
import type { ToolCall } from "../../Serving/ToolProtocol.ts";
import { ChatTokens, ChatTokenList, TruncateAtTurnEnd, SplitThinkingIds } from "../../Sft/ChatTemplate.ts";
import { OwnedSystemPrompt } from "../../Sft/OwnedSftData.ts";
import { SplitThinking } from "../../Reasoning/ThinkingMode.ts";
import type { SuiteSettings } from "./SuiteSettings.ts";

/** What one generation needs; a ModelGeneratorOptions superset, so the real factory takes it as-is. */
export type GenerationSettings = {
  MaxTokens: number;
  Sampling: SamplingOptions;
  Rng: SeededRng;
  ShouldStop?: () => boolean;
  OnPassEnd?: (CompletionIds: readonly number[]) => void; // raw ids incl. a natural stop id
};

/** The generator seam: settings in, an agent-loop Generator out. */
export type GeneratorFactory = (Settings: GenerationSettings) => Generator;

/** Everything a metric module needs to run its cases. */
export type SuiteContext = {
  Runnable: RunnableModel;
  Factory: GeneratorFactory; // chat generations (the serving generator unless a test injects one)
  Settings: SuiteSettings;
  Log: (Line: string) => void;
};

/** The default factory: the guarded serving generator over the loaded model. */
export function ModelGeneratorFactory(Runnable: RunnableModel): GeneratorFactory {
  return (Settings) => CreateModelGenerator(Runnable, Settings);
}

export type SuiteMessage = { Role: "System" | "User" | "Assistant"; Content: string };

/** A chat session holding Messages; the owned system prompt is prepended when the case brings none
 *  (train/serve prompt parity — the SFT data always carries it). Assistant history is base-encoded at
 *  render time, exactly like a client-replayed history at serving time. */
export function BuildSession(Messages: readonly SuiteMessage[]): ChatSession {
  const Session = new ChatSession(Messages[0]?.Role === "System" ? undefined : OwnedSystemPrompt);
  for (const M of Messages) {
    if (M.Role === "System") Session.Messages.push({ Role: "System", Content: M.Content });
    else if (M.Role === "User") Session.AddUser(M.Content);
    else Session.AddAssistant(M.Content);
  }
  return Session;
}

/** How the generated turn ended. */
export type TurnEnd = "end-of-turn" | "role-marker" | "no-end-of-turn";

/** The facts every chat metric judges, read from ONE generated turn. */
export type ReadTurn = {
  Answer: string; // the visible answer: thinking removed, stops at a genuine tool call; trimmed
  End: TurnEnd;
  ThinkingClosed: boolean; // every <|think|> span was closed by <|endthink|>
  LeakedLiterals: string[]; // chat/tool control-token text appearing in the visible answer
  ToolCallCount: number; // genuine tool-call starts (special ids; literal sentinels for trusted text)
  Call: ToolCall | null; // the first parsed tool call (the one the agent loop would run)
};

const ControlLiterals: readonly string[] = [...ChatTokenList, ...ToolTokenList];

function LeaksIn(Text: string): string[] {
  return ControlLiterals.filter((Literal) => Text.includes(Literal));
}

/** Read a special-token model's completion ids (as returned by the guarded generator). */
export function ReadTurnIds(Ids: readonly number[], Tok: SpecialTokenizer): ReadTurn {
  const TurnIds = TruncateAtTurnEnd(Ids, Tok);
  const EndId = TurnIds.length < Ids.length ? Ids[TurnIds.length] : null;
  const End: TurnEnd = EndId === null ? "no-end-of-turn" : EndId === Tok.TryId(ChatTokens.EndOfTurn) ? "end-of-turn" : "role-marker";
  const ThinkId = Tok.TryId(ChatTokens.Think);
  const EndThinkId = Tok.TryId(ChatTokens.EndThink);
  let Open = false;
  for (const Id of TurnIds) {
    if (Id === ThinkId) Open = true;
    else if (Id === EndThinkId) Open = false;
  }
  const AnswerIds = SplitThinkingIds(TurnIds, Tok).AnswerIds;
  const CallStartId = Tok.TryId(ToolTokens.CallStart);
  const CallAt = CallStartId === null ? -1 : AnswerIds.indexOf(CallStartId);
  // Decoding maps a STRAY special id to its literal too — shown to a user, that is a leak as well.
  const Visible = Tok.Decode(CallAt === -1 ? AnswerIds : AnswerIds.slice(0, CallAt));
  return {
    Answer: Visible.trim(),
    End,
    ThinkingClosed: !Open,
    LeakedLiterals: LeaksIn(Visible),
    // Counted over the whole turn, like the agent loop's own parse (ParseToolCallIds(TurnIds)).
    ToolCallCount: CallStartId === null ? 0 : TurnIds.filter((Id) => Id === CallStartId).length,
    Call: ParseToolCallIds(TurnIds, Tok),
  };
}

function CountOccurrences(Text: string, Needle: string): number {
  return Text.split(Needle).length - 1;
}

/** Read a TRUSTED text turn (a scripted stub, or a chat checkpoint with a plain tokenizer). Raw, when
 *  known, is the undecoded completion; without it a text turn is taken as properly ended. */
export function ReadTurnText(Text: string, Raw: string | null): ReadTurn {
  const EndAt = Text.indexOf(ChatTokens.EndOfTurn);
  const Turn = EndAt >= 0 ? Text.slice(0, EndAt) : Text;
  const Ended = Raw === null ? true : Raw.includes(ChatTokens.EndOfTurn);
  const Split = SplitThinking(Turn);
  const CallAt = Split.Answer.indexOf(ToolTokens.CallStart);
  const Visible = CallAt === -1 ? Split.Answer : Split.Answer.slice(0, CallAt);
  return {
    Answer: Visible.trim(),
    End: Ended ? "end-of-turn" : "no-end-of-turn",
    ThinkingClosed: CountOccurrences(Turn, ChatTokens.Think) <= CountOccurrences(Turn, ChatTokens.EndThink),
    LeakedLiterals: LeaksIn(Visible).filter((Literal) => !(ToolTokenList as readonly string[]).includes(Literal)),
    ToolCallCount: CountOccurrences(Turn, ToolTokens.CallStart),
    Call: ParseToolCall(Turn),
  };
}

/** Read whatever a Generator returned (Raw = the captured raw completion ids, if any). */
export function ReadGenerated(Output: string | GeneratedTurn, Raw: readonly number[] | null, Tok: Tokenizer): ReadTurn {
  if (typeof Output !== "string") return ReadTurnIds(Output.Ids, Output.Tok);
  return ReadTurnText(Output, Raw === null ? null : Tok.Decode([...Raw]));
}

/** Generate ONE assistant turn for Messages and read it. Throws what the generator throws. */
export async function ChatReply(Factory: GeneratorFactory, Tok: Tokenizer, Messages: readonly SuiteMessage[], Settings: GenerationSettings): Promise<ReadTurn> {
  let Raw: readonly number[] | null = null;
  const Generate = Factory({ ...Settings, OnPassEnd: (Ids) => { Raw = Ids; } });
  return ReadGenerated(await Generate(BuildSession(Messages)), Raw, Tok);
}

/** Untrusted text to ids: a SpecialTokenizer's base vocab only, so data can never forge a control id. */
export function EncodeUntrusted(Tok: Tokenizer, Text: string): number[] {
  return Tok instanceof SpecialTokenizer ? Tok.EncodeBase(Text) : Tok.Encode(Text);
}

/** Plain (base-model) continuation of PromptIds through the guarded generator; stops at EOS when the
 *  tokenizer reserves one. Returns the completion ids (EOS excluded). */
export async function CompleteIds(Runnable: RunnableModel, PromptIds: readonly number[], Settings: GenerationSettings): Promise<number[]> {
  const { Model, Tokenizer: Tok, Config } = Runnable;
  if (PromptIds.length === 0) throw new Error("CompleteIds: empty prompt — nothing to condition on");
  const Guard: GuardedStreamOptions = {};
  const EosId = Tok instanceof SpecialTokenizer ? Tok.TryId(SpecialTokens.Eos) : null;
  if (EosId !== null) Guard.StopIds = new Set([EosId]);
  const Ids = await GuardedGenerateIdsStream(Model, Tok, PromptIds, Settings.MaxTokens, Settings.Sampling, Settings.Rng, Config, () => undefined, Settings.ShouldStop, Guard);
  return EosId !== null && Ids[Ids.length - 1] === EosId ? Ids.slice(0, -1) : Ids;
}

/** Plain (base-model) completion of a prompt string; the decoded continuation. */
export async function CompletePlain(Runnable: RunnableModel, Prompt: string, Settings: GenerationSettings): Promise<string> {
  return Runnable.Tokenizer.Decode(await CompleteIds(Runnable, EncodeUntrusted(Runnable.Tokenizer, Prompt), Settings));
}

/** Sampling options at a temperature (0 = greedy argmax). */
export function SamplingAt(Temperature: number): SamplingOptions {
  return { ...DefaultSampling, Temperature };
}

// Distinct seeds per (case, sample) — the stride leaves room for any realistic sample count.
const CaseSeedStride = 1000;

/** The sampling RNG of one (case, sample): reproducible, decorrelated across cases and samples. */
export function CaseRng(Seed: number, CaseIndex: number, Sample: number): SeededRng {
  return CreateRngStreams(Seed + CaseIndex * CaseSeedStride + Sample).SamplingRng;
}

/** A per-case wall-clock budget: ShouldStop for generation, RemainingMs for test runs. */
export type CaseDeadline = { ShouldStop: () => boolean; RemainingMs: () => number; Expired: () => boolean };

export function StartDeadline(TimeoutMs: number, Now: () => number = Date.now): CaseDeadline {
  const Until = Now() + TimeoutMs;
  return {
    ShouldStop: () => Now() >= Until,
    RemainingMs: () => Math.max(0, Until - Now()),
    Expired: () => Now() >= Until,
  };
}

/** A failure reason for a thrown generation: the safety guard, or any other error. */
export function ErrorReason(Caught: unknown): string {
  return Caught instanceof SafetyBlockedError ? "safety-blocked" : "generation-error";
}
