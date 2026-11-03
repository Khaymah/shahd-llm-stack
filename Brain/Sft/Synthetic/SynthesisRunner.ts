// The synthesis loop of one shard. Each step picks the generator furthest behind its accepted quota
// (deterministic), derives the item (global index k + N·i, seeded by its key), asks the generator, runs
// the acceptance gate, and hands every accepted record to the Sink IMMEDIATELY — so a run stopped by its
// deadline, a SIGTERM or a dead teacher has already persisted everything it accepted. Stops on: the
// count reached, the deadline, an external stop, a dead teacher (K consecutive endpoint failures), or
// every generator out of attempts (quota × MaxAttemptFactor).

import { SeededRng } from "../../Random/SeededRng.ts";
import type { SynthAcceptance } from "./SynthAcceptance.ts";
import type { SynthMix } from "./SynthMix.ts";
import { PlanQuotas } from "./SynthMix.ts";
import type { GeneratorStats, SynthRecord, SynthStats } from "./SynthRecord.ts";
import { EmptyGeneratorStats, SyntheticProvenance } from "./SynthRecord.ts";
import type { ShardSpec } from "./SynthSampling.ts";
import { InShard, SeedForKey, ShardItemIndex } from "./SynthSampling.ts";
import type { CodeSlice, GroundingDoc, SynthContext, SynthGenerator, SynthGeneratorImpl } from "./SynthTypes.ts";
import { SynthGenerators } from "./SynthTypes.ts";
import { GroundedCodeGenerator } from "./GroundedCodeGenerator.ts";
import { TextbookGenerator } from "./TextbookGenerator.ts";
import { ToolUseGenerator } from "./ToolUseGenerator.ts";
import { MultiTurnGenerator } from "./MultiTurnGenerator.ts";

export const DefaultGenerators: readonly SynthGeneratorImpl[] = [GroundedCodeGenerator, TextbookGenerator, ToolUseGenerator, MultiTurnGenerator];

export type StopReason = "count-reached" | "deadline" | "stopped" | "teacher-down" | "attempts-exhausted" | "nothing-available";

/** Attempts allowed per generator, as a multiple of its accepted quota. */
export const DefaultMaxAttemptFactor = 4;
/** Consecutive endpoint failures (each already retried with backoff) that mean the teacher is down. */
export const DefaultMaxConsecutiveTeacherErrors = 5;
/** Log a progress line every this many accepted conversations. */
const ProgressEvery = 25;
const MsPerSecond = 1000;

export type SynthesisOptions = {
  Count: number; // accepted conversations wanted from THIS shard
  Mix: SynthMix;
  Shard: ShardSpec;
  Seed: number;
  Context: SynthContext;
  Gate: SynthAcceptance;
  Sink: (Record: SynthRecord) => void | Promise<void>;
  Generators?: readonly SynthGeneratorImpl[];
  DeadlineMs?: number | null; // absolute epoch ms; null = none
  Now?: () => number;
  ShouldStop?: () => boolean;
  MaxAttemptFactor?: number;
  MaxConsecutiveTeacherErrors?: number;
  Log?: (Line: string) => void;
};

/** This shard's grounding documents per requested slice (the shard owns the ids that hash to it). */
export function GroupGrounding(Docs: readonly GroundingDoc[], Shard: ShardSpec, Slices: readonly CodeSlice[]): Map<CodeSlice, GroundingDoc[]> {
  const Out = new Map<CodeSlice, GroundingDoc[]>();
  for (const Doc of Docs) {
    if (!Slices.includes(Doc.Slice) || !InShard(Doc.Id, Shard)) continue;
    const List = Out.get(Doc.Slice) ?? [];
    List.push(Doc);
    Out.set(Doc.Slice, List);
  }
  return Out;
}

function EmptyStats(): Record<SynthGenerator, GeneratorStats> {
  return { grounded: EmptyGeneratorStats(), textbook: EmptyGeneratorStats(), tools: EmptyGeneratorStats(), multiturn: EmptyGeneratorStats() };
}

// The generator furthest behind its quota (relative), among those with quota and attempts left.
function NextGenerator(Quotas: SynthMix, Stats: Record<SynthGenerator, GeneratorStats>, Factor: number): SynthGenerator | null {
  let Best: SynthGenerator | null = null;
  let BestDeficit = 0;
  for (const Name of SynthGenerators) {
    const Quota = Quotas[Name];
    const G = Stats[Name];
    if (Quota === 0 || G.Accepted >= Quota || G.Attempted >= Quota * Factor) continue;
    const Deficit = (Quota - G.Accepted) / Quota;
    if (Best === null || Deficit > BestDeficit) {
      Best = Name;
      BestDeficit = Deficit;
    }
  }
  return Best;
}

function CountRejection(G: GeneratorStats, Reason: string): void {
  G.Rejected[Reason] = (G.Rejected[Reason] ?? 0) + 1;
}

/** Run one shard to completion (see the header for the stop conditions). */
export async function RunSynthesis(Options: SynthesisOptions): Promise<SynthStats> {
  const Now = Options.Now ?? Date.now;
  const Log = Options.Log ?? ((): void => {});
  const Factor = Options.MaxAttemptFactor ?? DefaultMaxAttemptFactor;
  const MaxTeacherErrors = Options.MaxConsecutiveTeacherErrors ?? DefaultMaxConsecutiveTeacherErrors;
  const Impls = new Map((Options.Generators ?? DefaultGenerators).map((Impl) => [Impl.Name, Impl]));
  const Available = new Set([...Impls.values()].filter((Impl) => Impl.Available(Options.Context)).map((Impl) => Impl.Name));
  for (const Name of SynthGenerators) {
    if (Options.Mix[Name] > 0 && !Available.has(Name)) Log(`[synth] ${Name}: unavailable in this run (share redistributed)`);
  }
  const Quotas = PlanQuotas(Options.Count, Options.Mix, Available);
  Log(`[synth] shard ${Options.Shard.Index}/${Options.Shard.Total}: quotas ${SynthGenerators.map((Name) => `${Name}=${Quotas[Name]}`).join(" ")}`);
  const Stats = EmptyStats();
  const Started = Now();
  let Accepted = 0;
  let TeacherErrors = 0;
  let Stop: StopReason;

  for (;;) {
    if (Accepted >= Options.Count) { Stop = "count-reached"; break; }
    if (Options.DeadlineMs !== undefined && Options.DeadlineMs !== null && Now() >= Options.DeadlineMs) { Stop = "deadline"; break; }
    if (Options.ShouldStop?.() === true) { Stop = "stopped"; break; }
    const Name = NextGenerator(Quotas, Stats, Factor);
    if (Name === null) { Stop = Accepted === 0 && Available.size === 0 ? "nothing-available" : "attempts-exhausted"; break; }
    const Impl = Impls.get(Name)!;
    const G = Stats[Name];
    const Index = ShardItemIndex(Options.Shard, G.Attempted);
    const Key = `${Name}-${Index}`;
    const Seed = SeedForKey(Options.Seed, Key);
    G.Attempted++;
    const Outcome = await Impl.Generate({ Key, Index, Seed, Rng: new SeededRng(Seed) }, Options.Context);
    if (!Outcome.Ok) {
      CountRejection(G, Outcome.Reason);
      TeacherErrors = Outcome.Reason === "teacher-error" ? TeacherErrors + 1 : 0;
      if (TeacherErrors >= MaxTeacherErrors) { Log(`[synth] teacher down: ${Outcome.Detail}`); Stop = "teacher-down"; break; }
      continue;
    }
    TeacherErrors = 0;
    const Verdict = await Options.Gate.Judge(Outcome.Document);
    if (!Verdict.Ok) {
      CountRejection(G, Verdict.Reason);
      continue;
    }
    Options.Gate.Remember(Verdict.Hash);
    await Options.Sink({
      Generator: Name,
      Teacher: Options.Context.Teacher.Model,
      Slice: Outcome.Slice,
      Source: Outcome.Source.Name,
      SourceId: Outcome.Source.Id,
      SourceLicense: Outcome.Source.License,
      Provenance: SyntheticProvenance(Name, Options.Context.Teacher.Model, Outcome.Source.Id),
      Tokens: Verdict.Tokens,
      Hash: Verdict.Hash,
      Document: Outcome.Document,
    });
    G.Accepted++;
    G.Tokens += Verdict.Tokens;
    Accepted++;
    if (Accepted % ProgressEvery === 0) Log(`[synth] ${Accepted}/${Options.Count} accepted (${((Now() - Started) / MsPerSecond).toFixed(0)} s)`);
  }
  return {
    Teacher: Options.Context.Teacher.Model,
    Shard: `${Options.Shard.Index}/${Options.Shard.Total}`,
    StopReason: Stop,
    ElapsedMs: Now() - Started,
    Generators: Stats,
  };
}
