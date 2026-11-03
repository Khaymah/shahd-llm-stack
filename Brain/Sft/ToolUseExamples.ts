// Tool-use SFT exemplars — this is what makes tools "مدمج في المودل نفسه": the model is TRAINED to
// decide when to call a tool, to format the call with the exact ToolTokens the loop parses, and to
// answer after reading the tool result. RenderForTraining puts the loss ONLY on the assistant turns
// (the tool-call turn and the final answer), so tool-calling behavior is learned, not hard-coded.
// (At toy scale this is mechanism, not competence — CAPABILITIES.md's mechanism-vs-quality split.)

import type { ChatMessage } from "./ChatTemplate.ts";
import { FormatToolResult, ToolTokens } from "../Serving/ToolProtocol.ts";
import { WrapThinking } from "../Reasoning/ThinkingMode.ts";

export type ToolExemplar = {
  User: string;
  Call: { name: string; arguments: Record<string, unknown> };
  Result: Record<string, unknown>;
  Answer: string;
  /** Short scratchpads emitted before the tool call / the final answer, wrapped in
   *  <|think|>…<|endthink|> by BuildToolConversation. Optional so a bare exemplar stays valid,
   *  but the owned mix always provides both (the unified thinking-everywhere recipe). */
  CallThinking?: string;
  AnswerThinking?: string;
};

// A spread of tools and intents so the model sees the general shape, not one memorized call.
export const ToolUseExemplars: ToolExemplar[] = [
  { User: "What is 128 * 47?", CallThinking: "The user wants 128 * 47. I should use the calculator tool.", Call: { name: "calculator", arguments: { a: 128, op: "*", b: 47 } }, Result: { result: 6016 }, AnswerThinking: "The calculator returned 6016.", Answer: "128 * 47 = 6016." },
  { User: "Run this and tell me the output: console.log(2 ** 10)", CallThinking: "The user wants me to run this code and report its output. I should use the run_code tool.", Call: { name: "run_code", arguments: { code: "console.log(2 ** 10)" } }, Result: { passed: true, stdout: "1024\n" }, AnswerThinking: "The code ran and printed 1024.", Answer: "It prints 1024." },
  { User: "Show me the contents of README.md", CallThinking: "The user wants the contents of README.md. I should use the file_read tool.", Call: { name: "file_read", arguments: { path: "README.md" } }, Result: { path: "README.md", content: "# Shahd" }, AnswerThinking: "The file was read. It starts with a heading.", Answer: "The file starts with the heading \"# Shahd\"." },
  { User: "Look up the latest Bun release.", CallThinking: "The user asks about the latest Bun release. I should use the web_search tool.", Call: { name: "web_search", arguments: { query: "latest Bun release" } }, Result: { stub: true, results: [] }, AnswerThinking: "The search returned no results here.", Answer: "I couldn't reach the web here, so I can't confirm the latest release." },
  { User: "Remember that the API base URL is https://api.example.com", CallThinking: "The user wants me to remember the API base URL. I should use the memory_store tool.", Call: { name: "memory_store", arguments: { key: "api_base_url", value: "https://api.example.com" } }, Result: { stored: true, key: "api_base_url" }, AnswerThinking: "The value was stored under api_base_url.", Answer: "Saved the API base URL." },
];

/** The assistant-side text of one tool call: the JSON {name, arguments} between the call sentinels.
 *  The single home of this shape for authored SFT data (BuildToolConversation, ChatDocument). */
export function ToolCallText(Call: ToolExemplar["Call"]): string {
  return `${ToolTokens.CallStart}${JSON.stringify(Call)}${ToolTokens.CallEnd}`;
}

/** Turn an exemplar into a 5-message SFT conversation (system, user, tool-call, result, answer).
 *  When the exemplar carries thinking, each assistant turn starts with its <|think|> scratchpad. */
export function BuildToolConversation(Exemplar: ToolExemplar, SystemPrompt: string): ChatMessage[] {
  const CallContent = Exemplar.CallThinking !== undefined ? WrapThinking(Exemplar.CallThinking, ToolCallText(Exemplar.Call)) : ToolCallText(Exemplar.Call);
  const AnswerContent = Exemplar.AnswerThinking !== undefined ? WrapThinking(Exemplar.AnswerThinking, Exemplar.Answer) : Exemplar.Answer;
  return [
    { Role: "System", Content: SystemPrompt },
    { Role: "User", Content: Exemplar.User },
    { Role: "Assistant", Content: CallContent }, // trainable: think, then emit the tool call
    { Role: "User", Content: FormatToolResult(Exemplar.Result) }, // tool result fed back (same text the agent loop feeds)
    { Role: "Assistant", Content: AnswerContent }, // trainable: think, then answer from the result
  ];
}
