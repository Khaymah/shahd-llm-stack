// Robust JSON out of a teacher reply. Even in JSON mode a small local model sometimes wraps its object in
// a ```json fence, adds a sentence before it, or (without JSON mode) answers in prose around it. The
// extractor tries, in order: the whole reply, every fenced block, then every balanced {…} span found by
// a string-aware brace scan — and returns the first that parses. The schema check is the caller's
// contract (Zod), so a parseable object of the wrong shape is a distinct rejection from no JSON at all.

import type { ReplySchema, SftTeacher, TeacherMessage } from "./SynthTypes.ts";
import { TeacherError } from "./SynthTypes.ts";

const Fence = /```[A-Za-z0-9_-]*[ \t]*\r?\n([\s\S]*?)```/g;

function TryParse(Text: string): unknown {
  try {
    return JSON.parse(Text.trim()) as unknown;
  } catch {
    return undefined;
  }
}

function IsObject(Value: unknown): boolean {
  return typeof Value === "object" && Value !== null && !Array.isArray(Value);
}

// The end index (inclusive) of the balanced object opening at Start, honoring string literals and
// escapes so a "}" inside a string never closes the object; -1 when it never closes.
function MatchingBrace(Text: string, Start: number): number {
  let Depth = 0;
  let InString = false;
  for (let I = Start; I < Text.length; I++) {
    const Char = Text[I];
    if (InString) {
      if (Char === "\\") I++;
      else if (Char === "\"") InString = false;
      continue;
    }
    if (Char === "\"") InString = true;
    else if (Char === "{") Depth++;
    else if (Char === "}") {
      Depth--;
      if (Depth === 0) return I;
    }
  }
  return -1;
}

/** The first JSON OBJECT found in a teacher reply (whole text, fenced blocks, balanced spans), or null. */
export function ExtractJsonObject(Text: string): Record<string, unknown> | null {
  const Candidates: string[] = [Text];
  for (const Match of Text.matchAll(Fence)) Candidates.push(Match[1]);
  for (const Candidate of Candidates) {
    const Parsed = TryParse(Candidate);
    if (IsObject(Parsed)) return Parsed as Record<string, unknown>;
  }
  for (let Start = Text.indexOf("{"); Start !== -1; Start = Text.indexOf("{", Start + 1)) {
    const End = MatchingBrace(Text, Start);
    if (End === -1) continue;
    const Parsed = TryParse(Text.slice(Start, End + 1));
    if (IsObject(Parsed)) return Parsed as Record<string, unknown>;
  }
  return null;
}

export type TeacherJsonResult<T> = { Ok: true; Value: T } | { Ok: false; Reason: "teacher-error" | "no-json" | "bad-json-shape"; Detail: string };

/** The system message every generator sends: JSON only, and never the reserved token spellings. */
export const TeacherSystemPrompt =
  "You write training conversations for Shahd, a small, precise coding assistant. Reply with exactly ONE JSON object and nothing else. " +
  "Never write special tokens such as <|think|>, <|user|> or <|tool_call|>; plain text and Markdown code fences only.";

export type TeacherAsk = { Prompt: string; MaxTokens: number; Seed: number; Temperature?: number };

const DefaultTemperature = 0.7;

/** Ask the teacher for one JSON object and validate it against Schema. */
export async function AskTeacherJson<T>(Teacher: SftTeacher, Ask: TeacherAsk, Schema: ReplySchema<T>): Promise<TeacherJsonResult<T>> {
  const Messages: TeacherMessage[] = [
    { role: "system", content: TeacherSystemPrompt },
    { role: "user", content: Ask.Prompt },
  ];
  let Reply: string;
  try {
    Reply = await Teacher.Complete({ Messages, Json: true, MaxTokens: Ask.MaxTokens, Temperature: Ask.Temperature ?? DefaultTemperature, Seed: Ask.Seed });
  } catch (Caught) {
    const Message = Caught instanceof Error ? Caught.message : String(Caught);
    return { Ok: false, Reason: "teacher-error", Detail: Caught instanceof TeacherError ? Message : `unexpected: ${Message}` };
  }
  const Json = ExtractJsonObject(Reply);
  if (Json === null) return { Ok: false, Reason: "no-json", Detail: Reply.slice(0, 160) };
  const Parsed = Schema.safeParse(Json);
  if (!Parsed.success) return { Ok: false, Reason: "bad-json-shape", Detail: Parsed.error.issues.map((I) => `${I.path.join(".")}: ${I.message}`).join("; ") };
  return { Ok: true, Value: Parsed.data };
}
