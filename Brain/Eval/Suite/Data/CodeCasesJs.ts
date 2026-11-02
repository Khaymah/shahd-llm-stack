// Frozen v1 code cases — JavaScript slice (15). Plain JS (no type annotations) in prompts and
// references; executable exactly like the TS slice. Ladder: js-001..006 trivial, js-007..012 easy,
// js-013..015 medium.

import type { CodeCase } from "../SuiteTypes.ts";
import { CallExpr, ExecutableCase } from "./CodeCaseBuilder.ts";

const JsTrivial: readonly CodeCase[] = [
  ExecutableCase("js-001", "js",
    "Write a JavaScript function `isOdd(n)` that returns true when the integer n is odd and false otherwise. It must work for negative numbers too.",
    "function isOdd(n) {\n  return Math.abs(n % 2) === 1;\n}",
    [[CallExpr("isOdd", 3), true], [CallExpr("isOdd", 4), false], [CallExpr("isOdd", -3), true], [CallExpr("isOdd", -4), false], [CallExpr("isOdd", 0), false]]),
  ExecutableCase("js-002", "js",
    "Write a JavaScript function `celsiusToFahrenheit(c)` that converts a temperature in Celsius to Fahrenheit using F = C * 9 / 5 + 32.",
    "function celsiusToFahrenheit(c) {\n  return (c * 9) / 5 + 32;\n}",
    [[CallExpr("celsiusToFahrenheit", 0), 32], [CallExpr("celsiusToFahrenheit", 100), 212], [CallExpr("celsiusToFahrenheit", -40), -40], [CallExpr("celsiusToFahrenheit", 25), 77], [CallExpr("celsiusToFahrenheit", -10), 14]]),
  ExecutableCase("js-003", "js",
    "Write a JavaScript function `largest(numbers)` that returns the largest number in the array numbers, or null when the array is empty.",
    "function largest(numbers) {\n  return numbers.length === 0 ? null : Math.max(...numbers);\n}",
    [[CallExpr("largest", [3, 9, 2]), 9], [CallExpr("largest", [-5, -2, -9]), -2], [CallExpr("largest", []), null], [CallExpr("largest", [7]), 7]]),
  ExecutableCase("js-004", "js",
    "Write a JavaScript function `charCount(s)` that returns the number of Unicode code points in the string s, so an emoji such as \"\u{1F44D}\" counts as 1 (unlike s.length, which counts UTF-16 units).",
    "function charCount(s) {\n  return [...s].length;\n}",
    [[CallExpr("charCount", "abc"), 3], [CallExpr("charCount", ""), 0], [CallExpr("charCount", "\u{1F44D}"), 1], [CallExpr("charCount", "naïve \u{1F44D}\u{1F44D}"), 8]]),
  ExecutableCase("js-005", "js",
    "Write a JavaScript function `removeWhitespace(s)` that returns s with every whitespace character (spaces, tabs, newlines) removed.",
    "function removeWhitespace(s) {\n  return s.replace(/\\s+/g, \"\");\n}",
    [[CallExpr("removeWhitespace", " a b\tc\nd "), "abcd"], [CallExpr("removeWhitespace", ""), ""], [CallExpr("removeWhitespace", "none"), "none"]]),
  ExecutableCase("js-006", "js",
    "Write a JavaScript function `startsWithVowel(word)` that returns true when the first character of word is a, e, i, o or u (case-insensitive), and false otherwise, including for an empty string.",
    "function startsWithVowel(word) {\n  return /^[aeiou]/i.test(word);\n}",
    [[CallExpr("startsWithVowel", "apple"), true], [CallExpr("startsWithVowel", "Orange"), true], [CallExpr("startsWithVowel", "banana"), false], [CallExpr("startsWithVowel", ""), false], [CallExpr("startsWithVowel", "umbrella"), true], [CallExpr("startsWithVowel", "yes"), false]]),
];

const JsEasy: readonly CodeCase[] = [
  ExecutableCase("js-007", "js",
    "Write a JavaScript function `zip(a, b)` that returns an array of [a[i], b[i]] pairs, stopping at the end of the shorter array.",
    "function zip(a, b) {\n  const out = [];\n  for (let i = 0; i < Math.min(a.length, b.length); i++) out.push([a[i], b[i]]);\n  return out;\n}",
    [[CallExpr("zip", [1, 2, 3], ["a", "b", "c"]), [[1, "a"], [2, "b"], [3, "c"]]], [CallExpr("zip", [1, 2, 3], ["x"]), [[1, "x"]]], [CallExpr("zip", [], [1]), []]]),
  ExecutableCase("js-008", "js",
    "Write a JavaScript function `wordFrequencies(text)` that splits text on whitespace, lower-cases every word, and returns an object mapping each word to the number of times it appears. Blank text returns {}.",
    "function wordFrequencies(text) {\n  const counts = {};\n  for (const word of text.trim().split(/\\s+/)) {\n    if (word === \"\") continue;\n    const w = word.toLowerCase();\n    counts[w] = (counts[w] || 0) + 1;\n  }\n  return counts;\n}",
    [[CallExpr("wordFrequencies", "the cat and The hat"), { the: 2, cat: 1, and: 1, hat: 1 }], [CallExpr("wordFrequencies", "  "), {}], [CallExpr("wordFrequencies", "a a\na"), { a: 3 }]]),
  ExecutableCase("js-009", "js",
    "Write a JavaScript function `titleCase(s)` that upper-cases the first letter of every space-separated word and lower-cases the rest of the word. The spacing of s must be preserved exactly (two spaces stay two spaces).",
    "function titleCase(s) {\n  return s\n    .split(\" \")\n    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())\n    .join(\" \");\n}",
    [[CallExpr("titleCase", "hello world"), "Hello World"], [CallExpr("titleCase", "hELLO  wORLD"), "Hello  World"], [CallExpr("titleCase", ""), ""], [CallExpr("titleCase", "a"), "A"]]),
  ExecutableCase("js-010", "js",
    "Write a JavaScript function `runLengthEncode(s)` that replaces each run of the same character with the character followed by the run length, e.g. \"aaabcc\" becomes \"a3b1c2\". An empty string returns \"\".",
    "function runLengthEncode(s) {\n  let out = \"\";\n  for (let i = 0; i < s.length; ) {\n    let j = i;\n    while (j < s.length && s[j] === s[i]) j++;\n    out += s[i] + (j - i);\n    i = j;\n  }\n  return out;\n}",
    [[CallExpr("runLengthEncode", "aaabcc"), "a3b1c2"], [CallExpr("runLengthEncode", ""), ""], [CallExpr("runLengthEncode", "x"), "x1"], [CallExpr("runLengthEncode", "aaaaaaaaaaaa"), "a12"], [CallExpr("runLengthEncode", "abab"), "a1b1a1b1"]]),
  ExecutableCase("js-011", "js",
    "Write a JavaScript function `intersection(a, b)` that returns the distinct values of array a that also appear in array b (compared with ===), in the order they first appear in a.",
    "function intersection(a, b) {\n  const inB = new Set(b);\n  const out = [];\n  for (const x of a) if (inB.has(x) && !out.includes(x)) out.push(x);\n  return out;\n}",
    [[CallExpr("intersection", [1, 2, 2, 3, 4], [2, 4, 6]), [2, 4]], [CallExpr("intersection", [1, 2], [3]), []], [CallExpr("intersection", ["a", "b", "a"], ["a"]), ["a"]], [CallExpr("intersection", [], [1]), []], [CallExpr("intersection", [4, 1, 3], [3, 4]), [4, 3]]]),
  ExecutableCase("js-012", "js",
    "Write a JavaScript function `sortBy(items, key)` that returns a NEW array of the objects in items sorted ascending by item[key] (numbers or strings, compared with < and >). Items with equal keys keep their original relative order, and the input array must not be modified.",
    "function sortBy(items, key) {\n  return [...items].sort((x, y) => (x[key] < y[key] ? -1 : x[key] > y[key] ? 1 : 0));\n}",
    [
      [CallExpr("sortBy", [{ n: "b", a: 2 }, { n: "a", a: 1 }, { n: "c", a: 2 }], "a"), [{ n: "a", a: 1 }, { n: "b", a: 2 }, { n: "c", a: 2 }]],
      [CallExpr("sortBy", [{ s: "pear" }, { s: "apple" }, { s: "fig" }], "s"), [{ s: "apple" }, { s: "fig" }, { s: "pear" }]],
      [CallExpr("sortBy", [], "a"), []],
      "const SortInput = [{ v: 3 }, { v: 1 }]; const Sorted = sortBy(SortInput, \"v\"); if (Sorted === SortInput || SortInput[0].v !== 3) throw new Error(\"sortBy must not modify its input\");",
    ]),
];

const JsMedium: readonly CodeCase[] = [
  ExecutableCase("js-013", "js",
    "Write a JavaScript function `twoSumSorted(nums, target)` where nums is sorted ascending. Return the indices [i, j] with i < j such that nums[i] + nums[j] === target, or null if no such pair exists. Each input has at most one valid pair.",
    "function twoSumSorted(nums, target) {\n  let i = 0;\n  let j = nums.length - 1;\n  while (i < j) {\n    const sum = nums[i] + nums[j];\n    if (sum === target) return [i, j];\n    if (sum < target) i++;\n    else j--;\n  }\n  return null;\n}",
    [[CallExpr("twoSumSorted", [1, 2, 4, 7, 11], 15), [2, 4]], [CallExpr("twoSumSorted", [-3, 0, 3, 8], 0), [0, 2]], [CallExpr("twoSumSorted", [1, 2, 3], 10), null], [CallExpr("twoSumSorted", [], 1), null], [CallExpr("twoSumSorted", [5, 5], 10), [0, 1]]]),
  ExecutableCase("js-014", "js",
    "Write a JavaScript function `parseCsvLine(line)` that splits one CSV line into an array of field strings. Fields are separated by commas. A field may be wrapped in double quotes; inside quotes a comma is part of the value and a doubled quote (\"\") stands for one literal quote. The wrapping quotes are not part of the value. An empty line returns [\"\"].",
    "function parseCsvLine(line) {\n  const fields = [];\n  let field = \"\";\n  let inQuotes = false;\n  for (let i = 0; i < line.length; i++) {\n    const ch = line[i];\n    if (inQuotes) {\n      if (ch === '\"' && line[i + 1] === '\"') {\n        field += '\"';\n        i++;\n      } else if (ch === '\"') inQuotes = false;\n      else field += ch;\n    } else if (ch === '\"') inQuotes = true;\n    else if (ch === \",\") {\n      fields.push(field);\n      field = \"\";\n    } else field += ch;\n  }\n  fields.push(field);\n  return fields;\n}",
    [[CallExpr("parseCsvLine", "a,b,c"), ["a", "b", "c"]], [CallExpr("parseCsvLine", "name,\"Cairo, Egypt\",42"), ["name", "Cairo, Egypt", "42"]], [CallExpr("parseCsvLine", "\"say \"\"hi\"\"\",x"), ["say \"hi\"", "x"]], [CallExpr("parseCsvLine", "a,,b"), ["a", "", "b"]], [CallExpr("parseCsvLine", ""), [""]], [CallExpr("parseCsvLine", "a,"), ["a", ""]]]),
  ExecutableCase("js-015", "js",
    "Write a JavaScript class `EventBus` with methods `on(event, handler)`, `off(event, handler)` and `emit(event, ...args)`. emit calls every handler registered for that event, in registration order, with the given arguments, and returns how many handlers it called (0 when there are none). off removes that handler from that event and does nothing if it was not registered.",
    "class EventBus {\n  constructor() {\n    this.handlers = new Map();\n  }\n  on(event, handler) {\n    if (!this.handlers.has(event)) this.handlers.set(event, []);\n    this.handlers.get(event).push(handler);\n  }\n  off(event, handler) {\n    const list = this.handlers.get(event);\n    if (!list) return;\n    const i = list.indexOf(handler);\n    if (i !== -1) list.splice(i, 1);\n  }\n  emit(event, ...args) {\n    const list = [...(this.handlers.get(event) || [])];\n    for (const handler of list) handler(...args);\n    return list.length;\n  }\n}",
    [
      "const Bus = new EventBus(); const Seen = [];",
      "const First = (x) => { Seen.push(\"first:\" + x); }; const Second = (x, y) => { Seen.push(\"second:\" + x + y); };",
      "Bus.on(\"save\", First); Bus.on(\"save\", Second);",
      ["Bus.emit(\"save\", \"a\", \"b\")", 2],
      ["Seen", ["first:a", "second:ab"]],
      "Bus.off(\"save\", First);",
      ["Bus.emit(\"save\", \"c\", \"d\")", 1],
      ["Seen", ["first:a", "second:ab", "second:cd"]],
      ["Bus.emit(\"missing\")", 0],
      "Bus.off(\"missing\", First); Bus.off(\"save\", First);",
      ["Bus.emit(\"save\", \"e\", \"f\")", 1],
    ]),
];

export const JsCodeCases: readonly CodeCase[] = [...JsTrivial, ...JsEasy, ...JsMedium];
