// The generator mix (--Mix=grounded:0.4,textbook:0.2,tools:0.25,multiturn:0.15) and the slice list
// (--Slices=ts,js,…), parsed strictly, and the per-generator ACCEPTED quotas a run aims for. A generator
// that cannot produce anything in this run (no grounding documents, no executable slice) gets no quota
// and its share is redistributed over the rest, so --Count is still met.

import type { CodeSlice, SynthGenerator } from "./SynthTypes.ts";
import { CodeSlices, SynthGenerators } from "./SynthTypes.ts";

export type SynthMix = Record<SynthGenerator, number>;

export const DefaultSynthMix = "grounded:0.4,textbook:0.2,tools:0.25,multiturn:0.15";

function IsGenerator(Name: string): Name is SynthGenerator {
  return (SynthGenerators as readonly string[]).includes(Name);
}

/** Parse "name:weight,…" (unlisted generators weigh 0). */
export function ParseMix(Text: string): SynthMix {
  const Mix: SynthMix = { grounded: 0, textbook: 0, tools: 0, multiturn: 0 };
  for (const Part of Text.split(",").map((P) => P.trim()).filter((P) => P !== "")) {
    const [Name, Raw] = Part.split(":").map((S) => S.trim());
    if (Name === undefined || !IsGenerator(Name)) throw new Error(`--Mix: unknown generator '${Name ?? Part}' (known: ${SynthGenerators.join(", ")})`);
    const Weight = Number(Raw);
    if (Raw === undefined || Raw === "" || !Number.isFinite(Weight) || Weight < 0) throw new Error(`--Mix: '${Part}' needs a non-negative weight (name:weight)`);
    Mix[Name] = Weight;
  }
  if (Object.values(Mix).every((Weight) => Weight === 0)) throw new Error("--Mix: every weight is 0");
  return Mix;
}

/** Parse "ts,js,…" into known code slices (deduplicated, in the given order). */
export function ParseSlices(Text: string): CodeSlice[] {
  const Out: CodeSlice[] = [];
  for (const Part of Text.split(",").map((P) => P.trim()).filter((P) => P !== "")) {
    if (!(CodeSlices as readonly string[]).includes(Part)) throw new Error(`--Slices: unknown slice '${Part}' (known: ${CodeSlices.join(", ")})`);
    if (!Out.includes(Part as CodeSlice)) Out.push(Part as CodeSlice);
  }
  if (Out.length === 0) throw new Error("--Slices: name at least one slice");
  return Out;
}

/** Accepted-item quotas summing to Count over the AVAILABLE generators (largest-remainder rounding,
 *  ties broken by generator order so the plan is deterministic). */
export function PlanQuotas(Count: number, Mix: SynthMix, Available: ReadonlySet<SynthGenerator>): SynthMix {
  const Quotas: SynthMix = { grounded: 0, textbook: 0, tools: 0, multiturn: 0 };
  const Live = SynthGenerators.filter((Name) => Available.has(Name) && Mix[Name] > 0);
  const Total = Live.reduce((Sum, Name) => Sum + Mix[Name], 0);
  if (Total === 0 || Count <= 0) return Quotas;
  const Exact = Live.map((Name) => ({ Name, Value: (Count * Mix[Name]) / Total }));
  for (const Entry of Exact) Quotas[Entry.Name] = Math.floor(Entry.Value);
  let Left = Count - Exact.reduce((Sum, Entry) => Sum + Quotas[Entry.Name], 0);
  const ByRemainder = [...Exact].sort((A, B) => (B.Value - Math.floor(B.Value)) - (A.Value - Math.floor(A.Value)));
  for (const Entry of ByRemainder) {
    if (Left <= 0) break;
    Quotas[Entry.Name]++;
    Left--;
  }
  return Quotas;
}
