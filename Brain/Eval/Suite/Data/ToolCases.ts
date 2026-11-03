// Frozen v1 tool cases (40). Each user message should produce exactly ONE call to Expected.Name;
// Expected.Arguments is a SUBSET that must deep-equal the call's arguments (numbers as numbers).
// Tool names and argument shapes are the real ones from Brain/Serving/Tools (the test checks every
// name against a full-capability registry). Phrasings are natural and deliberately NOT the template
// sentences of ToolUseExemplars (Brain/Sft/ToolUseExamples.ts), so the metric measures generalisation, not recall.
//
// Free-text arguments are only pinned when the prompt pins them: web_search queries are quoted in the
// prompt; memory_store pins the value but not the key (any sensible key is fine); memory_recall and
// current_time pin only the tool name.

import type { ToolCase } from "../SuiteTypes.ts";

/** A tool case in the "general" slice (tool use is not domain-specific). */
function Call(Id: string, User: string, Name: string, Arguments: Record<string, unknown>): ToolCase {
  return { Id, Slice: "general", User, Expected: { Name, Arguments } };
}

export const ToolCases: readonly ToolCase[] = [
  // calculator — { a: number, op: '+'|'-'|'*'|'/'|'^'|'%', b: number }
  Call("tool-001", "what's 7 times 86?", "calculator", { a: 7, op: "*", b: 86 }),
  Call("tool-002", "multiply 12.5 by 4", "calculator", { a: 12.5, op: "*", b: 4 }),
  Call("tool-003", "how much is 1000 minus 377", "calculator", { a: 1000, op: "-", b: 377 }),
  Call("tool-004", "Add 2048 and 512 for me.", "calculator", { a: 2048, op: "+", b: 512 }),
  Call("tool-005", "I need 945 divided by 15.", "calculator", { a: 945, op: "/", b: 15 }),
  Call("tool-006", "What do I get if I raise 2 to the power of 16?", "calculator", { a: 2, op: "^", b: 16 }),
  Call("tool-007", "What's the remainder when 100 is divided by 7?", "calculator", { a: 100, op: "%", b: 7 }),
  Call("tool-008", "Can you compute 3.75 + 18.4?", "calculator", { a: 3.75, op: "+", b: 18.4 }),
  Call("tool-009", "We're splitting a 1250 bill evenly between 5 people. How much does each person pay?", "calculator", { a: 1250, op: "/", b: 5 }),
  Call("tool-010", "My disk has 64 GB and I've used 23 GB. How much is free?", "calculator", { a: 64, op: "-", b: 23 }),
  Call("tool-011", "quick: 19 * 19", "calculator", { a: 19, op: "*", b: 19 }),
  Call("tool-012", "Subtract 4.5 from 20.", "calculator", { a: 20, op: "-", b: 4.5 }),
  // stats — { values: number[], op: 'sum'|'mean'|'min'|'max'|'count' }
  Call("tool-013", "What's the average of 4, 8, 15, 16, 23 and 42?", "stats", { values: [4, 8, 15, 16, 23, 42], op: "mean" }),
  Call("tool-014", "Total up these response times in ms: 120, 340, 95, 210", "stats", { values: [120, 340, 95, 210], op: "sum" }),
  Call("tool-015", "Which is the biggest of 17, 3, 99 and 42?", "stats", { values: [17, 3, 99, 42], op: "max" }),
  // run_code — { code: string }
  Call("tool-016", "run this: console.log([1,2,3].map(x=>x*2))", "run_code", { code: "console.log([1,2,3].map(x=>x*2))" }),
  Call("tool-017", "Execute console.log(typeof null) and tell me what it prints.", "run_code", { code: "console.log(typeof null)" }),
  Call("tool-018", "What does console.log(0.1 + 0.2) actually output? Run it.", "run_code", { code: "console.log(0.1 + 0.2)" }),
  Call("tool-019", "Can you run console.log('abc'.padStart(6, '*')) for me?", "run_code", { code: "console.log('abc'.padStart(6, '*'))" }),
  Call("tool-020", "Try executing: console.log(JSON.stringify({ a: 1, b: [2, 3] }))", "run_code", { code: "console.log(JSON.stringify({ a: 1, b: [2, 3] }))" }),
  Call("tool-021", "Run console.log(new Set([1, 1, 2, 3]).size) and report the result.", "run_code", { code: "console.log(new Set([1, 1, 2, 3]).size)" }),
  // file_read — { path: string }
  Call("tool-022", "open package.json and show me what's in it", "file_read", { path: "package.json" }),
  Call("tool-023", "What does tsconfig.json say?", "file_read", { path: "tsconfig.json" }),
  Call("tool-024", "Read src/index.ts please.", "file_read", { path: "src/index.ts" }),
  Call("tool-025", "Can I see the contents of Docs/CONVENTIONS.md?", "file_read", { path: "Docs/CONVENTIONS.md" }),
  Call("tool-026", "Print the .env.example file.", "file_read", { path: ".env.example" }),
  // file_list — { path?: string }
  Call("tool-027", "What files are in the Brain folder?", "file_list", { path: "Brain" }),
  Call("tool-028", "List everything under Scripts", "file_list", { path: "Scripts" }),
  // file_search — { query: string }
  Call("tool-029", "Find any file with Tokenizer in its name.", "file_search", { query: "Tokenizer" }),
  // web_search — { query: string } (query quoted in the prompt so the expected text is unambiguous)
  Call("tool-030", "Search the web for \"Hono middleware docs\".", "web_search", { query: "Hono middleware docs" }),
  Call("tool-031", "Can you look up \"bun test coverage flag\" online?", "web_search", { query: "bun test coverage flag" }),
  Call("tool-032", "Google \"CSS container queries browser support\" for me.", "web_search", { query: "CSS container queries browser support" }),
  Call("tool-033", "Do a web search for \"Go 1.23 release notes\".", "web_search", { query: "Go 1.23 release notes" }),
  // memory_store — { key: string, value: string } (value pinned, key free)
  Call("tool-034", "remember my timezone is Africa/Cairo", "memory_store", { value: "Africa/Cairo" }),
  Call("tool-035", "Please keep in mind that our staging URL is https://staging.example.com", "memory_store", { value: "https://staging.example.com" }),
  Call("tool-036", "Note for later: my preferred package manager is bun.", "memory_store", { value: "bun" }),
  Call("tool-037", "Save this for later: we deploy from the release/v2 branch.", "memory_store", { value: "release/v2" }),
  Call("tool-038", "Don't forget that my GitHub username is sabry-dev.", "memory_store", { value: "sabry-dev" }),
  // memory_recall — { key?: string } (name only)
  Call("tool-039", "Check your memory: what did I save earlier?", "memory_recall", {}),
  // current_time — {} (name only)
  Call("tool-040", "What time is it right now?", "current_time", {}),
];
