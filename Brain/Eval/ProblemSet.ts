// A REAL, executable coding-problem set — the missing piece that lets STaR / RLVR (RejectionSampling)
// and pass@k evaluation actually run against ground truth. Each problem carries:
//   • Prompt    — the instruction the model is sampled on (STaR),
//   • Reference — a known-good solution (proves the tests are correct; every reference MUST pass its
//                 own tests — asserted in Tests/Eval.Test.ts),
//   • Tests     — TypeScript that references the candidate's definition and THROWS on any failure, run
//                 by the resource-capped CodeExecutor (pass = the runner prints the per-run nonce after all tests).
// Kept intentionally small + self-contained (pure functions, no imports) so a candidate solution can be
// prepended and executed directly. This is genuine ground truth, not a placeholder — a toy model will
// pass few/none of these, which the STaR script reports honestly (capability is scale-bound; the loop
// and its verifier are real).

import type { EvalProblem } from "./EvalHarness.ts";

export type CodingProblem = EvalProblem & { Prompt: string; Reference: string };

export const CodingProblems: CodingProblem[] = [
  {
    Name: "add",
    Prompt: "Write a TypeScript function `add(a: number, b: number): number` that returns the sum of a and b.",
    Reference: "function add(a: number, b: number): number { return a + b; }",
    Tests: `for (const [a, b] of [[2, 3], [-4, 4], [0, 0], [17, -9], [1000, -1000]]) if (add(a, b) !== a + b) throw new Error(\`add(\${a},\${b})\`);`,
  },
  {
    Name: "subtract",
    Prompt: "Write a TypeScript function `subtract(a: number, b: number): number` that returns a minus b.",
    Reference: "function subtract(a: number, b: number): number { return a - b; }",
    Tests: `if (subtract(5, 3) !== 2) throw new Error("5-3"); if (subtract(0, 7) !== -7) throw new Error("0-7");`,
  },
  {
    Name: "multiply",
    Prompt: "Write a TypeScript function `multiply(a: number, b: number): number` that returns a times b.",
    Reference: "function multiply(a: number, b: number): number { return a * b; }",
    Tests: `if (multiply(3, 4) !== 12) throw new Error("3*4"); if (multiply(-2, 5) !== -10) throw new Error("-2*5"); if (multiply(9, 0) !== 0) throw new Error("9*0");`,
  },
  {
    Name: "maxOf",
    Prompt: "Write a TypeScript function `maxOf(a: number, b: number): number` that returns the larger of a and b.",
    Reference: "function maxOf(a: number, b: number): number { return a > b ? a : b; }",
    Tests: `if (maxOf(3, 7) !== 7) throw new Error("max(3,7)"); if (maxOf(9, 2) !== 9) throw new Error("max(9,2)"); if (maxOf(4, 4) !== 4) throw new Error("max(4,4)");`,
  },
  {
    Name: "absVal",
    Prompt: "Write a TypeScript function `absVal(n: number): number` that returns the absolute value of n.",
    Reference: "function absVal(n: number): number { return n < 0 ? -n : n; }",
    Tests: `if (absVal(-5) !== 5) throw new Error("abs(-5)"); if (absVal(6) !== 6) throw new Error("abs(6)"); if (absVal(0) !== 0) throw new Error("abs(0)");`,
  },
  {
    Name: "factorial",
    Prompt: "Write a TypeScript function `factorial(n: number): number` that returns n! (factorial). factorial(0) is 1.",
    Reference: "function factorial(n: number): number { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }",
    Tests: `if (factorial(0) !== 1) throw new Error("0!"); if (factorial(5) !== 120) throw new Error("5!"); if (factorial(1) !== 1) throw new Error("1!");`,
  },
  {
    Name: "fib",
    Prompt: "Write a TypeScript function `fib(n: number): number` returning the nth Fibonacci number, 0-indexed (fib(0)=0, fib(1)=1).",
    Reference: "function fib(n: number): number { let a = 0, b = 1; for (let i = 0; i < n; i++) { const t = a + b; a = b; b = t; } return a; }",
    Tests: `if (fib(0) !== 0) throw new Error("fib(0)"); if (fib(1) !== 1) throw new Error("fib(1)"); if (fib(7) !== 13) throw new Error("fib(7)");`,
  },
  {
    Name: "isEven",
    Prompt: "Write a TypeScript function `isEven(n: number): boolean` that returns true when n is even.",
    Reference: "function isEven(n: number): boolean { return n % 2 === 0; }",
    Tests: `if (isEven(4) !== true) throw new Error("even 4"); if (isEven(7) !== false) throw new Error("odd 7"); if (isEven(0) !== true) throw new Error("even 0");`,
  },
  {
    Name: "reverseString",
    Prompt: "Write a TypeScript function `reverseString(s: string): string` that returns s reversed.",
    Reference: 'function reverseString(s: string): string { return s.split("").reverse().join(""); }',
    Tests: `if (reverseString("abc") !== "cba") throw new Error("abc"); if (reverseString("") !== "") throw new Error("empty"); if (reverseString("x") !== "x") throw new Error("x");`,
  },
  {
    Name: "sumArray",
    Prompt: "Write a TypeScript function `sumArray(xs: number[]): number` that returns the sum of the array (0 for an empty array).",
    Reference: "function sumArray(xs: number[]): number { let s = 0; for (const x of xs) s += x; return s; }",
    Tests: `if (sumArray([1, 2, 3]) !== 6) throw new Error("1,2,3"); if (sumArray([]) !== 0) throw new Error("empty"); if (sumArray([-1, 1]) !== 0) throw new Error("-1,1");`,
  },
  {
    Name: "countVowels",
    Prompt: "Write a TypeScript function `countVowels(s: string): number` that counts the vowels (a, e, i, o, u) in the lowercase string s.",
    Reference: 'function countVowels(s: string): number { let c = 0; for (const ch of s) if ("aeiou".includes(ch)) c++; return c; }',
    Tests: `if (countVowels("hello") !== 2) throw new Error("hello"); if (countVowels("xyz") !== 0) throw new Error("xyz"); if (countVowels("aeiou") !== 5) throw new Error("aeiou");`,
  },
  {
    Name: "isPalindrome",
    Prompt: "Write a TypeScript function `isPalindrome(s: string): boolean` that returns true when s reads the same forwards and backwards.",
    Reference: 'function isPalindrome(s: string): boolean { return s === s.split("").reverse().join(""); }',
    Tests: `if (isPalindrome("racecar") !== true) throw new Error("racecar"); if (isPalindrome("abc") !== false) throw new Error("abc"); if (isPalindrome("") !== true) throw new Error("empty");`,
  },
];

/** The problem tests as an "eval set" for decontamination — training corpus documents that overlap
 *  these strings would contaminate the pass@k benchmark, so they can be removed via BuildCorpus. */
export function ProblemEvalDocs(): string[] {
  return CodingProblems.map((P) => `${P.Prompt}\n${P.Reference}\n${P.Tests}`);
}
