// The REAL-model Generator for RunAgent (AgentLoop.ts), in one place: every generation pass renders the
// live session to ids (special-token models) and runs through GuardedGenerateIdsStream, so safety +
// resource limits apply exactly as on the guarded plain path — the pass's prompt is vetted before any
// compute, generation is capped at Config.Limits.MaxNewTokens / MaxContextTokens, and the output is
// vetted before any of it is delivered. Extracted from the dashboard's chat serving (Foundry/
// ModelHolder.ts) so the evaluation suite (Brain/Eval/Suite/) drives the model through the IDENTICAL
// path it is served on — an eval number measured on a different render/stop path would be meaningless.
//
// A special-token (chat) model hands the loop its completion ids (end of turn, thinking and tool calls
// are then decided on SPECIAL ids, never on decoded text). The <|endofturn|> id ends the pass the
// moment the model closes its turn and is kept as the LAST completion id, so a caller can tell a
// natural stop from a MaxTokens cut. A plain-tokenizer checkpoint has no special ids: it falls back to
// text, truncated at the literal end-of-turn marker.

import type { RunnableModel } from "../Checkpoint/LoadRunnableModel.ts";
import type { SamplingOptions } from "../Sampling/Sampler.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import { GuardedGenerateIdsStream } from "../Safety/GuardedGenerate.ts";
import type { GuardedStreamOptions, TokenGenerator } from "../Safety/GuardedGenerate.ts";
import { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";
import { ChatTokens } from "../Sft/ChatTemplate.ts";
import type { Generator } from "./AgentLoop.ts";

export type ModelGeneratorOptions = {
  MaxTokens: number; // per generation pass (further capped by Config.Limits.MaxNewTokens)
  Sampling: SamplingOptions;
  Rng: SeededRng; // shared across the passes of one agent run
  ShouldStop?: () => boolean; // polled per token (client disconnect, eval deadline)
  /** Called at the start of EACH generation pass; may return a per-token sink for that pass (live
   *  streaming of the vetted ids). Return null when nothing streams. */
  OnPassStart?: () => ((Id: number) => void) | null;
  /** Called with each pass's raw completion ids (a natural end-of-turn stop id kept as the LAST id). */
  OnPassEnd?: (CompletionIds: readonly number[]) => void;
  /** Test seam: the token generator under the guards (CachedGenerateAsync by default). */
  TokenGenerator?: TokenGenerator;
};

/** The model pieces a generator needs (a RunnableModel satisfies it). */
export type GeneratorModel = Pick<RunnableModel, "Model" | "Tokenizer" | "Config">;

/** Build the guarded agent-loop Generator for a loaded model. */
export function CreateModelGenerator(Runnable: GeneratorModel, Options: ModelGeneratorOptions): Generator {
  const { Model, Tokenizer, Config } = Runnable;
  // A chat model uses a SpecialTokenizer: the prompt is rendered directly to ids (untrusted content
  // base-encoded) so a user/tool string can't smuggle a real control token.
  const Special = Tokenizer instanceof SpecialTokenizer ? Tokenizer : null;
  const EndOfTurnId = Special === null ? null : Special.TryId(ChatTokens.EndOfTurn);
  const Guard: GuardedStreamOptions = {};
  if (EndOfTurnId !== null) Guard.StopIds = new Set([EndOfTurnId]);
  if (Options.TokenGenerator !== undefined) Guard.Generator = Options.TokenGenerator;

  return async (Turn) => {
    const PromptIds = Special !== null ? Turn.RenderPromptIds(Special) : Tokenizer.Encode(Turn.RenderPrompt());
    const Sink = Options.OnPassStart?.() ?? null;
    const Ids = await GuardedGenerateIdsStream(
      Model, Tokenizer, PromptIds, Options.MaxTokens, Options.Sampling, Options.Rng, Config,
      (Id) => Sink?.(Id),
      Options.ShouldStop,
      Guard,
    );
    Options.OnPassEnd?.(Ids);
    if (Special !== null) return { Ids, Tok: Special };
    const Text = Tokenizer.Decode(Ids);
    const End = Text.indexOf(ChatTokens.EndOfTurn);
    return End >= 0 ? Text.slice(0, End) : Text;
  };
}
