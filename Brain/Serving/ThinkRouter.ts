// Routes a decoded completion into the visible answer and the private <|think|>…<|endthink|> scratchpad
// as the text streams in. The OpenAI-compatible surface must not leak the scratchpad into `content`:
// the thinking goes out as `reasoning` deltas (the field reasoning-capable OpenAI-style servers use)
// and the answer as `content`. Works on text deltas of any size: a delta can end in the middle of a
// marker, so the longest possible marker prefix is held back until the next delta decides it.

import { ChatTokens } from "../Sft/ChatTemplate.ts";

export type RoutedText = { Content: string; Reasoning: string };

const Markers = [ChatTokens.Think, ChatTokens.EndThink] as const;
const LongestMarker = Math.max(...Markers.map((M) => M.length));

/** True when Text ends with a strict prefix of some marker (so the marker might complete later). */
function EndsWithMarkerPrefix(Text: string): number {
  for (let Len = Math.min(LongestMarker - 1, Text.length); Len > 0; Len--) {
    const Tail = Text.slice(Text.length - Len);
    if (Markers.some((M) => M.startsWith(Tail))) return Len;
  }
  return 0;
}

export class ThinkRouter {
  private Pending = "";
  private Thinking = false;

  /** Feed a delta; returns the text that is now safely attributable to each channel. */
  Push(Delta: string): RoutedText {
    this.Pending += Delta;
    const Out: RoutedText = { Content: "", Reasoning: "" };
    for (;;) {
      const Marker = this.Thinking ? ChatTokens.EndThink : ChatTokens.Think;
      const At = this.Pending.indexOf(Marker);
      if (At === -1) break;
      this.Emit(Out, this.Pending.slice(0, At));
      this.Pending = this.Pending.slice(At + Marker.length);
      this.Thinking = !this.Thinking;
    }
    const HoldBack = EndsWithMarkerPrefix(this.Pending);
    const Safe = this.Pending.slice(0, this.Pending.length - HoldBack);
    this.Pending = this.Pending.slice(this.Pending.length - HoldBack);
    this.Emit(Out, Safe);
    return Out;
  }

  /** The stream ended: whatever was held back is ordinary text of the current channel. */
  Flush(): RoutedText {
    const Out: RoutedText = { Content: "", Reasoning: "" };
    this.Emit(Out, this.Pending);
    this.Pending = "";
    return Out;
  }

  private Emit(Out: RoutedText, Text: string): void {
    if (Text.length === 0) return;
    if (this.Thinking) Out.Reasoning += Text;
    else Out.Content += Text;
  }
}

/** One-shot split of a complete completion. */
export function SplitThinkingText(Text: string): RoutedText {
  const Router = new ThinkRouter();
  const First = Router.Push(Text);
  const Rest = Router.Flush();
  return { Content: First.Content + Rest.Content, Reasoning: First.Reasoning + Rest.Reasoning };
}
