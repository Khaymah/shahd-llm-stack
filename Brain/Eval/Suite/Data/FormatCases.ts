// Frozen v1 format cases (60). Judged on FORMAT only — the reply is non-empty, stops at end-of-turn
// and leaks no control tokens — so they carry no expected text. Mix: fmt-001..040 single user turns
// across slices (greetings, one-line questions, "explain X in one sentence", "reply in a code block"),
// fmt-041..050 system-prompt variations, fmt-051..060 conversations with prior Assistant turns.

import type { EvalSlice, FormatCase } from "../SuiteTypes.ts";

type FormatMessage = FormatCase["Messages"][number];

/** A single-user-turn case (optionally preceded by a system prompt). */
function Ask(Id: string, Slice: EvalSlice, Content: string, System?: string): FormatCase {
  const Messages: FormatMessage[] = System === undefined ? [] : [{ Role: "System", Content: System }];
  Messages.push({ Role: "User", Content });
  return { Id, Slice, Messages };
}

const SingleTurn: readonly FormatCase[] = [
  Ask("fmt-001", "general", "Hi!"),
  Ask("fmt-002", "general", "Good morning, how are you?"),
  Ask("fmt-003", "general", "Who are you?"),
  Ask("fmt-004", "general", "Thanks, that helped a lot."),
  Ask("fmt-005", "general", "What can you help me with?"),
  Ask("fmt-006", "general", "Say hello in Arabic."),
  Ask("fmt-007", "general", "Give me one tip for staying focused while coding."),
  Ask("fmt-008", "general", "What is 2 + 2?"),
  Ask("fmt-009", "general", "Explain what an API is in one sentence."),
  Ask("fmt-010", "general", "Reply with just the word OK."),
  Ask("fmt-011", "ts", "What is TypeScript, in one sentence?"),
  Ask("fmt-012", "ts", "Explain the difference between `interface` and `type` in one sentence."),
  Ask("fmt-013", "ts", "Write a TypeScript function that returns the square of a number. Reply with a single code block."),
  Ask("fmt-014", "ts", "What does the `readonly` modifier do?"),
  Ask("fmt-015", "ts", "Show me how to type an array of strings in TypeScript, in a code block."),
  Ask("fmt-016", "ts", "Is `unknown` safer than `any`? Answer in one sentence."),
  Ask("fmt-017", "ts", "What does `as const` do?"),
  Ask("fmt-018", "js", "What does `===` do in JavaScript?"),
  Ask("fmt-019", "js", "Explain closures in one sentence."),
  Ask("fmt-020", "js", "Write a one-line arrow function that adds two numbers. Use a code block."),
  Ask("fmt-021", "js", "What is the difference between `let` and `const`?"),
  Ask("fmt-022", "js", "What does `Array.prototype.map` return?"),
  Ask("fmt-023", "js", "Explain the event loop in two sentences."),
  Ask("fmt-024", "html", "What is the `alt` attribute for?"),
  Ask("fmt-025", "html", "Write the HTML for a link to https://example.com in a code block."),
  Ask("fmt-026", "html", "Why should I use semantic HTML elements? One sentence."),
  Ask("fmt-027", "html", "What does the `<main>` element mean?"),
  Ask("fmt-028", "css", "What does `box-sizing: border-box` do?"),
  Ask("fmt-029", "css", "Explain flexbox in one sentence."),
  Ask("fmt-030", "css", "Give me the CSS to center text, in a code block."),
  Ask("fmt-031", "css", "What is the difference between `em` and `rem`?"),
  Ask("fmt-032", "go", "What is a goroutine, in one sentence?"),
  Ask("fmt-033", "go", "Write a Go hello world program in a code block."),
  Ask("fmt-034", "go", "What does `defer` do in Go?"),
  Ask("fmt-035", "sql", "What does `GROUP BY` do in SQL?"),
  Ask("fmt-036", "sql", "Write a SQL query that selects every row from a table named `logs`. Code block only."),
  Ask("fmt-037", "sql", "What is a primary key, in one sentence?"),
  Ask("fmt-038", "general", "List three popular web frameworks as a bulleted list."),
  Ask("fmt-039", "general", "Can you keep your answers short?"),
  Ask("fmt-040", "general", "What's the difference between git merge and git rebase, briefly?"),
];

const SystemVariations: readonly FormatCase[] = [
  Ask("fmt-041", "general", "Introduce yourself in one sentence.", "You are Shahd, a concise coding assistant."),
  Ask("fmt-042", "ts", "A TypeScript type for a point with x and y numbers.", "Answer only with code, no prose."),
  Ask("fmt-043", "general", "What is a variable?", "Always answer in Arabic."),
  Ask("fmt-044", "js", "What is a function?", "You are a patient tutor for beginner JavaScript developers."),
  Ask("fmt-045", "general", "Why is testing important?", "Keep every reply under 20 words."),
  Ask("fmt-046", "css", "How do I make a two-column layout?", "You are a senior front-end engineer. Prefer modern CSS."),
  Ask("fmt-047", "go", "When should I use a pointer receiver?", "You are a Go expert. Be brief."),
  Ask("fmt-048", "sql", "How do I list all tables?", "You help with PostgreSQL. Use SQL code blocks for queries."),
  Ask("fmt-049", "html", "Is a `<div>` with an onclick handler a good button?", "You review HTML for accessibility."),
  Ask("fmt-050", "general", "Tell me a short joke about programmers.", "You are a helpful assistant."),
];

const WithHistory: readonly FormatCase[] = [
  { Id: "fmt-051", Slice: "general", Messages: [
    { Role: "User", Content: "Hi" },
    { Role: "Assistant", Content: "Hello! How can I help you today?" },
    { Role: "User", Content: "Can you help me with TypeScript?" },
  ] },
  { Id: "fmt-052", Slice: "ts", Messages: [
    { Role: "User", Content: "What is a union type?" },
    { Role: "Assistant", Content: "A union type lets a value be one of several types, like `string | number`." },
    { Role: "User", Content: "Show me an example." },
  ] },
  { Id: "fmt-053", Slice: "js", Messages: [
    { Role: "User", Content: "How do I copy an array?" },
    { Role: "Assistant", Content: "Use the spread syntax: `const copy = [...original];`" },
    { Role: "User", Content: "Is that a deep copy?" },
  ] },
  { Id: "fmt-054", Slice: "css", Messages: [
    { Role: "User", Content: "How do I add a shadow to a box?" },
    { Role: "Assistant", Content: "Use `box-shadow`, for example `box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);`." },
    { Role: "User", Content: "How do I make it bigger?" },
  ] },
  { Id: "fmt-055", Slice: "general", Messages: [
    { Role: "User", Content: "Thanks!" },
    { Role: "Assistant", Content: "You're welcome!" },
    { Role: "User", Content: "One more question: what is JSON?" },
  ] },
  { Id: "fmt-056", Slice: "go", Messages: [
    { Role: "User", Content: "What's a slice in Go?" },
    { Role: "Assistant", Content: "A slice is a resizable view over an array, written like `[]int`." },
    { Role: "User", Content: "How do I append to one?" },
  ] },
  { Id: "fmt-057", Slice: "sql", Messages: [
    { Role: "User", Content: "How do I sort query results?" },
    { Role: "Assistant", Content: "Add `ORDER BY column` to the query, with `DESC` for descending order." },
    { Role: "User", Content: "And to get only the first ten rows?" },
  ] },
  { Id: "fmt-058", Slice: "html", Messages: [
    { Role: "User", Content: "What's an HTML form?" },
    { Role: "Assistant", Content: "An HTML form collects user input and submits it, using the `<form>` element." },
    { Role: "User", Content: "Which attribute sets where it submits?" },
  ] },
  { Id: "fmt-059", Slice: "ts", Messages: [
    { Role: "System", Content: "You are Shahd, a coding assistant." },
    { Role: "User", Content: "What does `?.` do?" },
    { Role: "Assistant", Content: "It's optional chaining: it stops and returns `undefined` when the value before it is null or undefined." },
    { Role: "User", Content: "And `??`?" },
  ] },
  { Id: "fmt-060", Slice: "general", Messages: [
    { Role: "User", Content: "Hello" },
    { Role: "Assistant", Content: "Hi! What are you working on?" },
    { Role: "User", Content: "A small blog." },
    { Role: "Assistant", Content: "Nice! What stack are you using?" },
    { Role: "User", Content: "Bun and Hono." },
  ] },
];

export const FormatCases: readonly FormatCase[] = [...SingleTurn, ...SystemVariations, ...WithHistory];
