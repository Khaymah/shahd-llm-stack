// Bits-per-BYTE on the frozen held-out set (SuiteCorpora.ts), per data kind and overall. "Does the
// model learn the language/code?" as one number that is comparable ACROSS tokenizers: EvalLoss reports
// mean nats per TOKEN (its BitsPerByte is bits-per-token — see Brain/Training/EvalLoop.ts), so here the
// total nats are divided by the UTF-8 bytes the scored target tokens decode to.
//
// Deterministic by construction: each kind's documents (in their frozen order) are joined with a blank
// line, encoded with the base vocabulary only (data is never allowed to forge a control id), and scored
// as consecutive NON-overlapping windows from the start — no sampling, so two runs on one checkpoint
// produce the same number, and a baseline means something.

import type { RunnableModel } from "../../Checkpoint/LoadRunnableModel.ts";
import type { DataLoader, Sequence } from "../../Data/DataLoader.ts";
import { EvalLoss } from "../../Training/EvalLoop.ts";
import type { Tokenizer } from "../../Tokenizer/TokenizerTypes.ts";
import type { MetricResult } from "./SuiteTypes.ts";
import type { HeldOutSet } from "./SuiteCorpora.ts";
import { EncodeUntrusted } from "./SuiteGeneration.ts";

const DocumentJoiner = "\n\n";

/** Consecutive non-overlapping windows from the start of a token stream; counts the target bytes. */
export class SequentialWindowLoader implements DataLoader {
  TargetBytes = 0;
  private Next = 0;
  private readonly Tokens: readonly number[];
  private readonly Window: number;
  private readonly Tok: Tokenizer;

  constructor(Tokens: readonly number[], Window: number, Tok: Tokenizer) {
    if (Window < 1 || Tokens.length < Window + 1) throw new Error(`SequentialWindowLoader: ${Tokens.length} tokens cannot fill a ${Window}-token window`);
    this.Tokens = Tokens;
    this.Window = Window;
    this.Tok = Tok;
  }

  /** How many full windows the stream holds. */
  get Capacity(): number {
    return Math.floor((this.Tokens.length - 1) / this.Window);
  }

  GetSequence(): Sequence {
    if (this.Next >= this.Capacity) throw new Error("SequentialWindowLoader: past the last full window");
    const Start = this.Next * this.Window;
    this.Next++;
    const Ids = this.Tokens.slice(Start, Start + this.Window);
    const Targets = this.Tokens.slice(Start + 1, Start + 1 + this.Window);
    this.TargetBytes += Buffer.byteLength(this.Tok.Decode(Targets), "utf8");
    return { Ids, Targets };
  }
}

type KindScore = { Kind: string; Docs: number; Tokens: number; Bytes: number; Nats: number };

function ScoreKind(Runnable: RunnableModel, Kind: string, Texts: readonly string[], MaxWindows: number): KindScore | null {
  const Tokens = EncodeUntrusted(Runnable.Tokenizer, Texts.join(DocumentJoiner));
  const Window = Math.min(Runnable.Config.Model.BlockSize, Tokens.length - 1);
  if (Window < 1) return null;
  const Loader = new SequentialWindowLoader(Tokens, Window, Runnable.Tokenizer);
  const Windows = Math.min(MaxWindows, Loader.Capacity);
  const { Loss } = EvalLoss(Runnable.Model, Loader, Windows); // mean nats per scored token
  const Scored = Window * Windows;
  return { Kind, Docs: Texts.length, Tokens: Scored, Bytes: Loader.TargetBytes, Nats: Loss * Scored };
}

function Bpb(Nats: number, Bytes: number): number {
  return Bytes > 0 ? Nats / Math.LN2 / Bytes : 0;
}

/** `bpb.<kind>` per kind (sorted by kind) and `bpb.all` (total nats / total bytes); lower is better. */
export function HeldOutBpbMetrics(Runnable: RunnableModel, HeldOut: HeldOutSet, MaxWindows: number): MetricResult[] {
  const ByKind = new Map<string, string[]>();
  for (const Doc of HeldOut.Docs) ByKind.set(Doc.Kind, [...(ByKind.get(Doc.Kind) ?? []), Doc.Content]);
  const Scores: KindScore[] = [];
  const TooShort: string[] = [];
  for (const Kind of [...ByKind.keys()].sort()) {
    const Score = ScoreKind(Runnable, Kind, ByKind.get(Kind) ?? [], MaxWindows);
    if (Score === null) TooShort.push(Kind);
    else Scores.push(Score);
  }
  const Common = { Origin: HeldOut.Origin, Source: HeldOut.Note, Unit: "bits per UTF-8 byte of the scored targets" };
  const Out: MetricResult[] = Scores.map((S) => ({
    Name: `bpb.${S.Kind}`,
    Value: Bpb(S.Nats, S.Bytes),
    Higher: "worse",
    N: S.Tokens,
    Details: { ...Common, Docs: S.Docs, Bytes: S.Bytes, BitsPerToken: S.Nats / Math.LN2 / S.Tokens },
  }));
  let Nats = 0;
  let Bytes = 0;
  let Tokens = 0;
  for (const S of Scores) {
    Nats += S.Nats;
    Bytes += S.Bytes;
    Tokens += S.Tokens;
  }
  Out.push({
    Name: "bpb.all",
    Value: Bpb(Nats, Bytes),
    Higher: "worse",
    N: Tokens,
    Details: { ...Common, Kinds: Scores.map((S) => S.Kind), Bytes, ...(TooShort.length > 0 ? { SkippedTooShort: TooShort } : {}) },
  });
  return Out;
}
