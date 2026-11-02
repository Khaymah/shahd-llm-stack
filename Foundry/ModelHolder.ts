// The live chat model (STAGE 3 of the Foundry pipeline): which checkpoint is loaded, how it is
// (re)loaded, and how a chat turn is served from it. Extracted from the dashboard script so the load
// rules are unit-testable with injected loaders:
//   - Postgres first (durable/synced), then a FILE fallback — for the DEFAULT checkpoint name only
//     (another name must never silently serve the default file's weights under its own label).
//   - Name/Runnable/Info change together and only after a SUCCESSFUL load; a failed load keeps the
//     current model serving.
//   - Loads are serialized (promise-chain mutex): a user's load-model and a train-done reload can
//     never interleave and leave Name pointing at one checkpoint while Runnable holds another.

import { existsSync } from "node:fs";
import type { Shahd } from "../Brain/Nn/Shahd.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { LoadRunnableModel, LoadRunnableModelFrom } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import type { RunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { GuardedGenerateStream, IncrementalDecoder } from "../Brain/Safety/GuardedGenerate.ts";
import { SafetyPolicy } from "../Brain/Safety/SafetyPolicy.ts";
import { RenderMessages } from "../Brain/Serving/RenderChat.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import type { AgentStep } from "../Brain/Serving/AgentLoop.ts";
import { CreateModelGenerator } from "../Brain/Serving/ModelGenerator.ts";
import { FormatTrace, BuildTrace } from "../Brain/Serving/ReasoningTrace.ts";
import { BuildAgentTooling } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { OwnedSystemPrompt } from "../Brain/Sft/OwnedSftData.ts";
import { SplitThinkingIds, TruncateAtTurnEnd } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";
import { ExtractAnswer, NormalizeAnswer, MajorityVote } from "../Brain/Reasoning/ReasoningBarrel.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { DescribeModel } from "./ModelInfo.ts";
import type { ModelInfo } from "./ModelInfo.ts";
import type { ChatMessage } from "./ChatStore.ts";
import type { ChatStreamFn, ChatOpts } from "./ChatService.ts";

/** Where a named checkpoint is loaded from (PostgresCheckpointStore in production). */
export type CheckpointSource = { Load(Name: string): Promise<Checkpoint | null> };

export type ModelHolderDeps = {
  Store: CheckpointSource | null; // null = no database configured
  DefaultName: string; // the startup checkpoint name — the only one allowed the file fallback
  FallbackFile: string; // checkpoint FILE for the default name (FoundryEnv.CheckpointFile)
  FromCheckpoint?: (Ckpt: Checkpoint) => RunnableModel;
  FromFile?: (Path: string) => RunnableModel;
  FileExists?: (Path: string) => boolean;
  Describe?: (Model: Shahd) => ModelInfo;
  Warn?: (Message: string) => void;
};

export class ModelHolder {
  Runnable: RunnableModel | null = null;
  Info: ModelInfo | null = null;
  Source = "none";
  Name: string;
  private Deps: ModelHolderDeps;
  private Queue: Promise<void> = Promise.resolve(); // the load mutex (tail of the chain)

  constructor(Deps: ModelHolderDeps) {
    this.Deps = Deps;
    this.Name = Deps.DefaultName;
  }

  /** Load a checkpoint (by name; default = the current one) as the live model. Serialized with every
   *  other load. Resolves true when it loaded, false when nothing was found (current model kept). */
  Reload(Name: string = this.Name): Promise<boolean> {
    const Next = this.Queue.then(() => this.LoadNow(Name));
    this.Queue = Next.then(
      () => undefined,
      () => undefined,
    );
    return Next;
  }

  private async LoadNow(Name: string): Promise<boolean> {
    const Warn = this.Deps.Warn ?? ((Message: string): void => console.warn(Message));
    const FromCheckpoint = this.Deps.FromCheckpoint ?? LoadRunnableModelFrom;
    let Loaded: { Runnable: RunnableModel; Source: string } | null = null;
    if (this.Deps.Store !== null) {
      try {
        const Ckpt = await this.Deps.Store.Load(Name);
        if (Ckpt !== null) Loaded = { Runnable: FromCheckpoint(Ckpt), Source: `postgres:${Name}` };
      } catch (Caught) {
        Warn(`[model] checkpoint load failed for "${Name}": ${(Caught as Error).message}`);
      }
    }
    const File = this.Deps.FallbackFile;
    if (Loaded === null && Name === this.Deps.DefaultName && (this.Deps.FileExists ?? existsSync)(File)) {
      try {
        Loaded = { Runnable: (this.Deps.FromFile ?? LoadRunnableModel)(File), Source: `file:${File}` };
      } catch (Caught) {
        Warn(`[model] checkpoint file load failed for "${File}": ${(Caught as Error).message}`);
      }
    }
    if (Loaded === null) {
      Warn(`[model] no loadable checkpoint "${Name}" — keeping ${this.Runnable === null ? "no model" : `"${this.Name}"`}`);
      return false;
    }
    this.Runnable = Loaded.Runnable;
    this.Info = (this.Deps.Describe ?? DescribeModel)(Loaded.Runnable.Model);
    this.Source = Loaded.Source;
    this.Name = Name;
    return true;
  }
}

// The streaming detokenizer now lives with the guarded generator that uses it (Brain/Safety/);
// re-exported so existing Foundry-side imports keep working.
export { IncrementalDecoder };

/** Live think/answer split of ONE generation pass, decided on SPECIAL ids exactly as AgentLoop decides
 *  the final turn (TruncateAtTurnEnd + SplitThinkingIds), never by parsing decoded text: ids inside a
 *  <|think|>…<|endthink|> span stream to OnThink, answer ids to OnAnswer. Nothing streams past the end
 *  of the turn, and the answer stops at a <|tool_call|> SPECIAL id (the call surfaces as a structured
 *  tool line when it executes). Each id is routed once and decoded incrementally; the id split itself
 *  is a cheap integer scan per token. A new think span after the first is separated by a blank line,
 *  matching how the final trace joins spans. */
export class LiveTurnSplit {
  private readonly Tok: SpecialTokenizer;
  private readonly OnThink: (Delta: string) => void;
  private readonly OnAnswer: (Delta: string) => void;
  private readonly CallStartId: number | null;
  private readonly Ids: number[] = [];
  private readonly Answer: IncrementalDecoder;
  private Think: IncrementalDecoder;
  private AnswerIdCount = 0;
  private ThinkIdCount = 0;
  private SpanCount = 0;
  private SentAnswer = 0;
  private SentThink = 0;
  private AnswerOpen = true; // false once a tool call starts: the rest of the answer is call payload
  private TurnOpen = true; // false once an end-of-turn / role-marker id ends the turn

  constructor(Tok: SpecialTokenizer, OnThink: (Delta: string) => void, OnAnswer: (Delta: string) => void) {
    this.Tok = Tok;
    this.OnThink = OnThink;
    this.OnAnswer = OnAnswer;
    this.CallStartId = Tok.TryId(ToolTokens.CallStart);
    this.Answer = new IncrementalDecoder(Tok);
    this.Think = new IncrementalDecoder(Tok);
  }

  Push(Id: number): void {
    if (!this.TurnOpen) return;
    if (TruncateAtTurnEnd([Id], this.Tok).length === 0) {
      this.TurnOpen = false; // turn-end detection is per id, so one id answers it
      return;
    }
    this.Ids.push(Id);
    const Split = SplitThinkingIds(this.Ids, this.Tok);
    if (Split.AnswerIds.length > this.AnswerIdCount) {
      this.AnswerIdCount = Split.AnswerIds.length;
      if (Id === this.CallStartId) this.AnswerOpen = false;
      if (this.AnswerOpen) this.SentAnswer = LiveTurnSplit.Emit(this.Answer, Id, this.SentAnswer, this.OnAnswer);
      return;
    }
    if (Split.ThinkingIds.length > this.SpanCount) {
      this.SpanCount = Split.ThinkingIds.length; // a <|think|> id opened a new span
      if (this.SpanCount > 1) {
        this.Think = new IncrementalDecoder(this.Tok);
        this.SentThink = 0;
        this.OnThink("\n\n");
      }
      return;
    }
    let ThinkIds = 0;
    for (const Span of Split.ThinkingIds) ThinkIds += Span.length;
    if (ThinkIds > this.ThinkIdCount) {
      this.ThinkIdCount = ThinkIds;
      this.SentThink = LiveTurnSplit.Emit(this.Think, Id, this.SentThink, this.OnThink);
    }
    // Otherwise the id was an <|endthink|> marker: nothing to show.
  }

  // Decode Id into Decoder and send the newly stable text; returns the new sent length.
  private static Emit(Decoder: IncrementalDecoder, Id: number, Sent: number, Send: (Delta: string) => void): number {
    const Text = Decoder.Push(Id);
    if (Text.length <= Sent) return Sent;
    Send(Text.slice(Sent));
    return Text.length;
  }
}

export type ChatServingOptions = {
  SelfConsistencySamples: number; // N>1 = majority vote over N runs (buffered, not live-streamed)
  Log?: (Message: string) => void;
};

/** The chat stream served from the holder's CURRENT model, so a post-training reload takes effect
 *  immediately. Chat-format (SFT) models go through the agent loop; base models the guarded plain path. */
export function CreateChatStream(Holder: ModelHolder, Options: ChatServingOptions): ChatStreamFn {
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));
  const Counter = { Value: 0 }; // distinct sampling seed per run => diverse samples
  return async (Messages, Opts, OnDelta) => {
    const Loaded = Holder.Runnable;
    if (Loaded === null) throw new Error("no trained model yet — collect data, then press Train Model");
    if (Loaded.Chat) return ServeChatAgent(Loaded, Messages, Opts, OnDelta, Options.SelfConsistencySamples, Counter, Log);
    const { Model, Tokenizer, Config } = Loaded;
    const Prompt = RenderMessages(Messages.map((M) => ({ role: M.Role, content: M.Content })));
    const LastMsg = (Messages[Messages.length - 1]?.Content ?? "").slice(0, 70).replace(/\s+/g, " ");
    const T0 = Date.now();
    let Chars = 0;
    let Chunks = 0;
    Log(`[chat] ▶ turns=${Messages.length} temp=${Opts.Temperature} maxTok=${Opts.MaxTokens} · "${LastMsg}"`);
    const Rng = new SeededRng(Config.Training.Seed + Counter.Value++);
    const Reply = await GuardedGenerateStream(
      Model, Tokenizer, Prompt, Opts.MaxTokens, { ...DefaultSampling, Temperature: Opts.Temperature }, Rng, Config,
      (Delta) => {
        Chars += Delta.length;
        Chunks++;
        OnDelta(Delta);
      },
      Opts.ShouldStop,
    );
    const Ms = Date.now() - T0;
    Log(`[chat] ✓ ${Chars} chars / ${Chunks} tokens in ${Ms}ms (${Ms > 0 ? Math.round(Chars / (Ms / 1000)) : 0} ch/s) · "${Reply.slice(0, 70).replace(/\s+/g, " ")}"`);
    return Reply;
  };
}

// Chat-format (SFT) models serve through the AGENT LOOP: the model can emit tool calls and thinking,
// tools run against the config-gated registry, and the reasoning trace (thinking -> tool+result ->
// answer) is logged — the development lens. Every generation pass goes through
// GuardedGenerateIdsStream, so safety + resource limits apply exactly as on the guarded plain path: the
// pass's prompt (the whole conversation, tool results included) is vetted before any compute, the pass
// is capped at Config.Limits.MaxNewTokens / MaxContextTokens, and its output is vetted before any of it
// streams. The final answer is vetted once more before it is returned/persisted (see below).
async function ServeChatAgent(
  Loaded: RunnableModel,
  Messages: ChatMessage[],
  Opts: ChatOpts,
  OnDelta: (Delta: string) => void,
  SelfConsistencySamples: number,
  Counter: { Value: number },
  Log: (Message: string) => void,
): Promise<string> {
  const { Tokenizer, Config } = Loaded;
  const Policy = new SafetyPolicy(Config);
  const Tooling = BuildAgentTooling(Config);
  // TRAIN/SERVE PROMPT PARITY: the SFT conversations were trained with the short OwnedSystemPrompt and
  // NO tool manifest — the tool-call FORMAT lives in the weights. Tiny models do not survive
  // prompt-distribution drift; serve them EXACTLY what they were trained on.
  const SystemPrompt = OwnedSystemPrompt;
  // A chat model uses a SpecialTokenizer: the prompt is rendered directly to ids (untrusted content
  // base-encoded) so a user/tool string can't smuggle a real control token, and every control decision
  // is made on special ids. (A chat checkpoint with a plain tokenizer has no special ids: it falls back
  // to text and is not token-streamed.)
  const Special = Tokenizer instanceof SpecialTokenizer ? Tokenizer : null;
  // With one run (the default) each tool call surfaces the moment it executes and — for a special-token
  // model — think text streams into the trace and the answer into the bubble. With self-consistency
  // N>1 the runs are candidates (losers are discarded), so nothing can stream.
  const SingleRun = SelfConsistencySamples <= 1;
  const StreamTokens = SingleRun && Special !== null;
  // ASYNC + EARLY-STOP: the guarded generator (Brain/Serving/ModelGenerator.ts — the same factory the
  // eval suite drives) yields per token (the dashboard never freezes), honors ShouldStop, and the
  // <|endofturn|> id ends the pass the moment the model closes its turn.
  const Sampling = { ...DefaultSampling, Temperature: Opts.Temperature };
  let ThinkSeg = 0; // one segment per generation pass => the UI groups think spans per agent step

  // One full agent run over a FRESH session (so N self-consistency runs don't contaminate each other).
  const RunOnce = async (): Promise<{ Text: string; Steps: AgentStep[] }> => {
    const Session = new ChatSession(SystemPrompt);
    for (const M of Messages) {
      if (M.Role === "assistant") Session.AddAssistant(M.Content);
      else Session.AddUser(M.Content);
    }
    Tooling.Context.Session = Session;
    const Rng = new SeededRng(Config.Training.Seed + Counter.Value++);
    const Gen = CreateModelGenerator(Loaded, {
      MaxTokens: Opts.MaxTokens,
      Sampling,
      Rng,
      ShouldStop: Opts.ShouldStop,
      OnPassStart: () => {
        const Seg = ThinkSeg++;
        if (!StreamTokens || Special === null) return null;
        const Live = new LiveTurnSplit(Special, (Delta) => Opts.OnThink?.(Delta, Seg), OnDelta);
        return (Id) => Live.Push(Id);
      },
    });
    const Steps: AgentStep[] = [];
    const Result = await RunAgent(Session, Gen, Tooling.Registry, Tooling.MaxSteps, Tooling.Context, (Step) => {
      Steps.push(Step);
      // Surface each tool call/result the moment it happens; the final BuildTrace stays authoritative.
      if (SingleRun) for (const L of BuildTrace([Step])) if (L.Kind === "tool") Opts.OnToolStep?.(L);
    });
    return { Text: Result.FinalText, Steps };
  };

  let Best: { Text: string; Steps: AgentStep[] };
  if (SingleRun) {
    Best = await RunOnce();
  } else {
    // Draw N runs, then majority-vote over their NORMALIZED final answers (so "42"/"42." count
    // together); the winning run's own text + trace is what we surface.
    const Runs: { Text: string; Steps: AgentStep[] }[] = [];
    for (let I = 0; I < SelfConsistencySamples; I++) Runs.push(await RunOnce());
    const Vote = MajorityVote(Runs.map((R) => ExtractAnswer(R.Text)), NormalizeAnswer);
    const WinnerKey = NormalizeAnswer(Vote.Winner);
    Best = Runs.find((R) => NormalizeAnswer(ExtractAnswer(R.Text)) === WinnerKey) ?? Runs[0]!;
    Log(`[chat] self-consistency: ${SelfConsistencySamples} runs, winner "${Vote.Winner}" (${Vote.Count}/${Vote.Total}); tally ${JSON.stringify(Vote.Tally)}`);
  }

  // Not redundant with the per-pass output guard: a `finish` tool's answer is a JSON-DECODED argument
  // (escapes like \n become real chars), so the returned text can differ from anything the guard scanned.
  Policy.EnforceOutput(Best.Text);
  Log(`[chat] reasoning trace:\n${FormatTrace(Best.Steps)}`);
  Opts.OnTrace?.(BuildTrace(Best.Steps)); // surface the trace to the dashboard UI
  if (!StreamTokens) OnDelta(Best.Text); // token-streamed runs already delivered the answer
  return Best.Text;
}
