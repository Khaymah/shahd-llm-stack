// Contracts of the synthetic-SFT pipeline (Plan: Brain-Strengthening §2.3). A local open-weights TEACHER
// (served over an OpenAI-compatible endpoint — the HTTP client lives in Foundry/TeacherClient.ts so the
// Brain never touches the network) writes conversations in Shahd's exact format; every generator
// grounds or verifies what the teacher says (real corpus code, executed tests, executed tools, checked
// recall) and the acceptance gate (SynthAcceptance.ts) decides what may be stored.

import type { z } from "zod";
import type { SeededRng } from "../../Random/SeededRng.ts";
import type { EvalSlice } from "../../Eval/Suite/SuiteTypes.ts";
import type { EvalProblem } from "../../Eval/EvalHarness.ts";
import type { ChatDocument } from "../ChatDocument.ts";
import type { ToolReplay } from "../ChatDocumentValidation.ts";

// ---- teacher ---------------------------------------------------------------------------------------

/** One chat message sent to the teacher (OpenAI wire shape, hence the lowercase keys). */
export type TeacherMessage = { role: "system" | "user" | "assistant"; content: string };

export type TeacherRequest = {
  Messages: readonly TeacherMessage[];
  Json: boolean; // ask for a JSON object (response_format) where the server supports it
  MaxTokens: number;
  Temperature: number;
  Seed: number; // per-item seed: the same item asks the same question
};

/** The teacher: anything that completes a chat. Injected everywhere (tests use an in-process fake). */
export type SftTeacher = {
  Model: string; // recorded in provenance as teacher:<Model>
  Complete: (Request: TeacherRequest) => Promise<string>;
};

/** Thrown by a teacher when the endpoint failed after its retries (counted as "teacher-error"). */
export class TeacherError extends Error {}

// ---- generators ------------------------------------------------------------------------------------

export const SynthGenerators = ["grounded", "textbook", "tools", "multiturn"] as const;
export type SynthGenerator = (typeof SynthGenerators)[number];

/** The code slices a run can target (the eval suite's slices minus "general"). */
export type CodeSlice = Exclude<EvalSlice, "general">;
export const CodeSlices: readonly CodeSlice[] = ["ts", "js", "html", "css", "go", "sql"];

/** A real corpus document a grounded conversation is built around (TRAINING side of the split only). */
export type GroundingDoc = {
  Id: string; // store id (or a content hash for --Corpus files): the shard + split key
  Slice: CodeSlice;
  Lang: string; // LangForPath tag, e.g. "typescript"
  License: string; // the document's SPDX id (or OWNED / NOASSERTION)
  Source: string; // repo / corpus name
  Path: string; // provenance path (shown to the teacher for context)
  Content: string;
};

/** The source-license value for conversations not built on any corpus document. */
export const UngroundedSourceLicense = "UNGROUNDED";

/** Where a conversation came from: the grounding document, or the item itself when ungrounded. */
export type SynthSource = { Id: string; License: string; Name: string };

/** One attempt: a stable key (also the ungrounded source id), its global index and a seeded stream. */
export type SynthItem = { Key: string; Index: number; Seed: number; Rng: SeededRng };

/** Runs a candidate solution against test statements (the nonce harness by default; injectable). */
export type CandidateVerifier = (Problem: EvalProblem, Candidate: string) => Promise<boolean>;

/** Everything a generator may use. */
export type SynthContext = {
  Teacher: SftTeacher;
  Budget: number; // token budget of the whole rendered conversation
  MeasureText: (Text: string) => number; // tokens of a plain text (real tokenizer or estimate)
  Slices: readonly CodeSlice[]; // the requested slices
  Grounding: ReadonlyMap<CodeSlice, readonly GroundingDoc[]>; // this shard's grounding docs per slice
  Tools: ToolReplay; // the synthesis tool registry + per-document context (shared with the validator)
  Verify: CandidateVerifier;
};

export type SynthOutcome =
  | { Ok: true; Document: ChatDocument; Source: SynthSource; Slice: EvalSlice }
  | { Ok: false; Reason: string; Detail: string };

export type SynthGeneratorImpl = {
  Name: SynthGenerator;
  /** Whether this generator can produce anything under the context (e.g. grounded needs documents). */
  Available: (Context: SynthContext) => boolean;
  Generate: (Item: SynthItem, Context: SynthContext) => Promise<SynthOutcome>;
};

/** A rejected attempt (Reason is the statistics bucket). */
export function Rejected(Reason: string, Detail = ""): SynthOutcome {
  return { Ok: false, Reason, Detail };
}

/** A Zod schema whose OUTPUT type is T, whatever its input (teacher replies are unknown JSON). */
export type ReplySchema<T> = z.ZodType<T, z.ZodTypeDef, unknown>;
