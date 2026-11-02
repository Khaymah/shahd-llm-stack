// Deterministic sampling + sharding for the synthesizer. Parallel CI jobs must produce DISJOINT work that
// together is COMPLETE, with no coordination: a shard k/N owns
//   • the grounding documents whose id hashes to k (ShardOf), and
//   • the item indices k, k+N, k+2N, … of every generator (ShardItemIndex),
// so two shards never ask the teacher the same question about the same document, and the union over
// k = 0..N-1 covers every document and index exactly once. Each item's randomness comes from its own
// key (SeedForKey), so a re-run of one shard reproduces it regardless of what other shards did.

import { createHash } from "node:crypto";
import type { SeededRng } from "../../Random/SeededRng.ts";

export type ShardSpec = { Index: number; Total: number };

const ShardPattern = /^(\d+)\/(\d+)$/;
// 8 hex digits = 32 bits of the sha256: exact as a JS integer, ample for any shard count.
const HashHexDigits = 8;
const HexRadix = 16;

/** Parse "--Shard=k/N" (0-based k, 0 <= k < N). */
export function ParseShard(Text: string): ShardSpec {
  const Match = ShardPattern.exec(Text.trim());
  if (Match === null) throw new Error(`--Shard expects k/N (0-based, e.g. 0/8), got '${Text}'`);
  const Index = Number(Match[1]);
  const Total = Number(Match[2]);
  if (Total < 1 || Index >= Total) throw new Error(`--Shard=${Text}: need N >= 1 and 0 <= k < N`);
  return { Index, Total };
}

function HashU32(Text: string): number {
  return Number.parseInt(createHash("sha256").update(Text).digest("hex").slice(0, HashHexDigits), HexRadix);
}

/** The shard (0..Total-1) a key belongs to. */
export function ShardOf(Key: string, Total: number): number {
  return HashU32(Key) % Total;
}

/** True when Key belongs to this shard. */
export function InShard(Key: string, Shard: ShardSpec): boolean {
  return ShardOf(Key, Shard.Total) === Shard.Index;
}

/** The global item index of this shard's Local-th item (disjoint across shards, complete over all). */
export function ShardItemIndex(Shard: ShardSpec, Local: number): number {
  return Shard.Index + Shard.Total * Local;
}

/** A 32-bit seed for one item, derived from the run seed and the item's key. */
export function SeedForKey(Seed: number, Key: string): number {
  return HashU32(`${Seed}:${Key}`);
}

/** One element drawn uniformly. Items must be non-empty. */
export function PickOne<T>(Rng: SeededRng, Items: readonly T[]): T {
  if (Items.length === 0) throw new Error("PickOne: empty list");
  return Items[Math.floor(Rng.NextFloat() * Items.length)];
}

/** A uniform integer in [Min, Max]. */
export function PickInt(Rng: SeededRng, Min: number, Max: number): number {
  return Min + Math.floor(Rng.NextFloat() * (Max - Min + 1));
}

/** How the user of a synthetic conversation writes — varied so the model learns the INTENT, not one
 *  sentence template (the tool-argument failure the eval measured is exactly that over-fit). */
export const UserStyles: readonly string[] = [
  "terse and lowercase, like a quick chat message",
  "polite and complete, one or two sentences",
  "casual, with a small typo",
  "in the middle of a work task, giving a bit of context first",
  "as a direct imperative instruction",
  "as a question",
  "slightly impatient, very short",
  "a beginner who is unsure of the terminology",
];
