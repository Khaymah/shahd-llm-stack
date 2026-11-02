// Source-text features for the data filters (Plan: Brain-Strengthening §2.2). ONE lexical pass
// (ScanSource) separates code, comments and string/number literals without a real parser, and every
// filter layer reads from it so the same notion of "comment line" or "string literal" is used
// everywhere (rule #4):
//   - layer 1, structural (CodeFileFilter content gate + ScoreCodeQuality comment band),
//   - layer 3, educational value (EducationalScore's logistic regression over ExtractEduFeatures).
//
// The scanner is a heuristic, language-agnostic state machine: `//`, `/* */`, `<!-- -->` comments,
// `#` / `--` line comments only when preceded by a line start or whitespace AND followed by a space
// (so `#include`, CSS `#fff` and `i--` stay code), quoted strings that end at the line break (a stray apostrophe in prose or a
// regex cannot swallow the file), multi-line backtick and triple-quoted strings, and Python docstrings
// (a triple-quoted string that starts a line) counted as documentation, not data.

/** Everything one pass over the text yields. Char counts are NON-whitespace characters. */
export type SourceScan = {
  LineCount: number;
  NonEmptyLines: number;
  CommentLines: number; // non-empty lines whose only content is comment/docstring text
  CodeLines: number; // non-empty lines with some code (string literals included)
  /** Per line: the code with comments removed and every string literal collapsed to its quotes. */
  CodeLineTexts: string[];
  /** Per line: whether the line carries any code (vs blank / comment-only). */
  LineHasCode: boolean[];
  StringChars: number;
  CommentChars: number;
  NumberChars: number;
  CodeChars: number; // code characters outside strings/comments (numbers included)
  NonSpaceChars: number;
  DocComments: number; // `/** … */` blocks + Python docstrings
};

type ScanState = "Code" | "LineComment" | "BlockComment" | "HtmlComment" | "String";

const IdentifierChar = /[\w$]/;
const NumberBody = /[0-9a-fA-FxXoObB_.]/;

function IsSpace(Char: string): boolean {
  return Char === " " || Char === "\t" || Char === "\r" || Char === "\f" || Char === "\v";
}

function IsLineCommentAfterMarker(Text: string, At: number): boolean {
  const Next = Text[At] ?? "\n";
  return Next === " " || Next === "\t" || Next === "\n" || Next === "\r" || Next === "#" || Next === "-";
}

/** One lexical pass over source text (see the module header for the rules). Pure and deterministic. */
export function ScanSource(Text: string): SourceScan {
  const Scan: SourceScan = {
    LineCount: 0, NonEmptyLines: 0, CommentLines: 0, CodeLines: 0, CodeLineTexts: [], LineHasCode: [],
    StringChars: 0, CommentChars: 0, NumberChars: 0, CodeChars: 0, NonSpaceChars: 0, DocComments: 0,
  };
  let State: ScanState = "Code";
  let Quote = "";
  let Triple = false;
  let IsDocString = false;
  let HasCode = false;
  let HasComment = false;
  let LineCode = "";

  const EndLine = (): void => {
    Scan.LineCount++;
    Scan.CodeLineTexts.push(LineCode);
    Scan.LineHasCode.push(HasCode);
    if (HasCode) Scan.CodeLines++;
    else if (HasComment) Scan.CommentLines++;
    if (HasCode || HasComment) Scan.NonEmptyLines++;
    if (State === "LineComment") State = "Code";
    // A plain quote never spans lines here: an unterminated one (apostrophe in a comment-less prose
    // line, a regex literal) is closed at the line break instead of eating the rest of the file.
    if (State === "String" && !Triple && Quote !== "`") State = "Code";
    HasCode = false;
    HasComment = false;
    LineCode = "";
  };

  for (let I = 0; I < Text.length; I++) {
    const Char = Text[I]!;
    if (Char === "\n") {
      EndLine();
      continue;
    }
    const Space = IsSpace(Char);
    if (!Space) Scan.NonSpaceChars++;

    if (State === "LineComment" || State === "BlockComment" || State === "HtmlComment") {
      if (!Space) {
        Scan.CommentChars++;
        HasComment = true;
      }
      if (State === "BlockComment" && Char === "*" && Text[I + 1] === "/") {
        Scan.CommentChars++;
        Scan.NonSpaceChars++;
        I++;
        State = "Code";
      } else if (State === "HtmlComment" && Char === "-" && Text.startsWith("-->", I)) {
        Scan.CommentChars += 2;
        Scan.NonSpaceChars += 2;
        I += 2;
        State = "Code";
      }
      continue;
    }

    if (State === "String") {
      const Closes = Triple ? Text.startsWith(Quote.repeat(3), I) : Char === Quote;
      if (!Space) {
        if (IsDocString) {
          Scan.CommentChars++;
          HasComment = true;
        } else {
          Scan.StringChars++;
          HasCode = true;
        }
      }
      if (Char === "\\" && Text[I + 1] !== "\n" && I + 1 < Text.length) {
        I++;
        if (IsDocString) Scan.CommentChars++;
        else Scan.StringChars++;
        Scan.NonSpaceChars++;
        continue;
      }
      if (Closes) {
        if (Triple) {
          const Extra = 2;
          if (IsDocString) Scan.CommentChars += Extra;
          else Scan.StringChars += Extra;
          Scan.NonSpaceChars += Extra;
          I += Extra;
        }
        if (!IsDocString) LineCode += Quote;
        State = "Code";
      }
      continue;
    }

    // State === "Code"
    if (Space) {
      LineCode += Char;
      continue;
    }
    const AtStatementStart = !HasCode;
    const Next = Text[I + 1];
    if (Char === "/" && Next === "/") {
      State = "LineComment";
    } else if (Char === "/" && Next === "*") {
      State = "BlockComment";
      if (Text[I + 2] === "*" && Text[I + 3] !== "/") Scan.DocComments++;
    } else if (Char === "<" && Text.startsWith("<!--", I)) {
      State = "HtmlComment";
    } else if (Char === "#" && (AtStatementStart || IsSpace(Text[I - 1] ?? "\n")) && (IsLineCommentAfterMarker(Text, I + 1) || (I === 0 && Next === "!"))) {
      State = "LineComment";
    } else if (Char === "-" && Next === "-" && (I === 0 || IsSpace(Text[I - 1]!) || Text[I - 1] === "\n") && IsLineCommentAfterMarker(Text, I + 2)) {
      State = "LineComment";
    }
    if (State !== "Code") {
      Scan.CommentChars++;
      HasComment = true;
      continue;
    }

    if (Char === '"' || Char === "'" || Char === "`") {
      Triple = Char !== "`" && Text.startsWith(Char.repeat(3), I);
      IsDocString = Triple && AtStatementStart;
      Quote = Char;
      State = "String";
      if (IsDocString) {
        Scan.DocComments++;
        Scan.CommentChars += 3;
        HasComment = true;
      } else {
        Scan.StringChars += Triple ? 3 : 1;
        HasCode = true;
        LineCode += Char;
      }
      if (Triple) {
        Scan.NonSpaceChars += 2;
        I += 2;
      }
      continue;
    }

    HasCode = true;
    Scan.CodeChars++;
    if (Char >= "0" && Char <= "9" && (I === 0 || !IdentifierChar.test(Text[I - 1]!))) {
      // A numeric literal: digits plus hex/octal/binary prefixes, separators, fraction and exponent.
      let End = I + 1;
      while (End < Text.length && (NumberBody.test(Text[End]!) || ((Text[End] === "+" || Text[End] === "-") && /[eE]/.test(Text[End - 1]!)))) End++;
      const Length = End - I;
      Scan.NumberChars += Length;
      Scan.CodeChars += Length - 1;
      Scan.NonSpaceChars += Length - 1;
      LineCode += Text.slice(I, End);
      I = End - 1;
      continue;
    }
    LineCode += Char;
  }
  EndLine();
  return Scan;
}

// ---------------------------------------------------------------------------------------------
// Line-level classifiers shared by the structural gate (CodeFileFilter) and the features below.

// A module-wiring line: ES/CommonJS imports and re-exports, Go/Java/Kotlin package + import, C
// #include, Rust/PHP `use`, C# `using`, Python `from x import y`, and the "use strict" directive.
const ImportLine =
  /^\s*(import\b|export\s+(\*|\{|type\s+\{|default\s+from\b)|(const|let|var)\s+[\w${}\s,:]+=\s*require\s*\(|require\s*\(|package\s+[\w.]+\s*;?\s*$|#\s*include\b|using\s+[\w.]+\s*;|use\s+[\w:\\{}, ]+;|from\s+[\w.]+\s+import\b|""\s*;?\s*$|''\s*;?\s*$)/;

/** Fraction of CODE lines (comment-only and blank lines excluded) that only wire modules together:
 *  import/export-from/require/include/package/use lines, including the continuation lines of a
 *  multi-line `import { … }` or Go `import ( … )` block. 0 when there are no code lines. */
export function ImportLineFraction(Scan: SourceScan): number {
  let ImportLines = 0;
  let OpenBlock = false;
  for (let I = 0; I < Scan.CodeLineTexts.length; I++) {
    if (!Scan.LineHasCode[I]) continue;
    const Line = Scan.CodeLineTexts[I]!;
    if (OpenBlock) {
      ImportLines++;
      if (/[})]/.test(Line)) OpenBlock = false;
      continue;
    }
    if (ImportLine.test(Line)) {
      ImportLines++;
      // An opening `{` / `(` that is not closed on the same line continues onto the next lines.
      const Opens = (Line.match(/[{(]/g) ?? []).length;
      const Closes = (Line.match(/[})]/g) ?? []).length;
      OpenBlock = Opens > Closes;
    }
  }
  return Scan.CodeLines === 0 ? 0 : ImportLines / Scan.CodeLines;
}

const LiteralToken = /""|''|``|\b\d[\w.]*\b|\b(true|false|null|undefined|nil|None|True|False|NULL)\b/g;
const ObjectKey = /[A-Za-z_$][\w$-]*\s*:(?![:=])/g;

/** Fraction of "content" code lines that are pure data: literals (strings/numbers/true/false/null),
 *  `key:` labels and punctuation only (`{ id: 1, name: "x" },` is a data row). Lines with no literal and no identifier
 *  (`}`, `];`) are neutral and not counted. 0 when no line carries content. */
export function DataLineFraction(Scan: SourceScan): number {
  let Content = 0;
  let Data = 0;
  for (let I = 0; I < Scan.CodeLineTexts.length; I++) {
    if (!Scan.LineHasCode[I]) continue;
    const Line = Scan.CodeLineTexts[I]!;
    const HasLiteral = LiteralToken.test(Line);
    LiteralToken.lastIndex = 0;
    const Rest = Line.replace(ObjectKey, "").replace(LiteralToken, "");
    const HasIdentifier = /[A-Za-z_$]/.test(Rest);
    if (!HasLiteral && !HasIdentifier) continue;
    Content++;
    if (HasLiteral && !HasIdentifier) Data++;
  }
  return Content === 0 ? 0 : Data / Content;
}

/** Fraction of the non-comment content that sits inside string literals. */
export function StringLiteralFraction(Scan: SourceScan): number {
  const Content = Scan.CodeChars + Scan.StringChars;
  return Content === 0 ? 0 : Scan.StringChars / Content;
}

/** Comment-only lines over non-empty lines (the comment-to-code ratio, 0..1). */
export function CommentLineRatio(Scan: SourceScan): number {
  return Scan.NonEmptyLines === 0 ? 0 : Scan.CommentLines / Scan.NonEmptyLines;
}

// Declarations across the supported language families: keyword declarations (JS/TS/Go/Python/Rust/
// Kotlin/Swift/Ruby/Elixir/Java/C#…), C-style function definitions, top-level `name = …` / `name := …`
// bindings, CSS rules with properties, SQL DDL/DML statements and HTML document elements.
const DeclarationPatterns: readonly RegExp[] = [
  /\b(function|class|const|let|var|def|defp|defmodule|func|fn|fun|struct|interface|type|enum|trait|impl|module|object|record|macro|sub|proc|namespace)\b/,
  /\b(public|private|protected|static|internal)\b[^;\n]*\(/,
  /^[ \t]*[A-Za-z_][\w*&<>[\], ]*[ \t*&]+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(const\s*)?\{/m,
  /^[ \t]*(export\s+)?[A-Za-z_]\w*\s*(:[^=\n]*)?=[^=>]/m,
  /\{[^{}]*[\w-]+\s*:\s*[^;{}]+;[^{}]*\}/,
  /\b(select\s+[\w*]|insert\s+into|create\s+(table|view|index|function|procedure|type|or|unique|schema)|alter\s+table|update\s+\w+\s+set|delete\s+from|with\s+\w+\s+as\s*\()/i,
  /<(!doctype|html|head|body|template|main|section|div|span|nav|header|footer|form|table|ul|ol|button|input|svg|article)\b/i,
];

/** Does the code (comments removed, strings collapsed) declare anything at all? */
export function HasDeclaration(Scan: SourceScan): boolean {
  const Code = Scan.CodeLineTexts.join("\n");
  return DeclarationPatterns.some((Pattern) => Pattern.test(Code));
}

// Test-file and runnable-example signals (Jest/Vitest/Bun/Mocha, Go testing, pytest/unittest, JUnit,
// doctests, JSDoc @example, Go Example funcs, "Example:" sections).
const TestSignal = /\b(describe|it|test|expect)\s*\(|\bfunc\s+(Test|Benchmark)\w*\s*\(|\bdef\s+test_\w*|@Test\b|\bassert(Equal|True|False|\.\w+)?\s*\(|\bunittest\b/;
const ExampleSignal = /@example\b|\bfunc\s+Example\w*\s*\(|^\s*(\/\/|#|\*)?\s*Examples?:|^\s*>>>\s/m;

/** Does the text carry tests? (Checked on the raw text: test names live in strings and calls.) */
export function HasTestSignals(Text: string): boolean {
  return TestSignal.test(Text);
}

/** Does the text carry tests OR runnable examples / documented usage? */
export function HasTestsOrExamples(Text: string): boolean {
  return TestSignal.test(Text) || ExampleSignal.test(Text);
}

// ---------------------------------------------------------------------------------------------
// The educational-value feature vector (layer 3). Order is part of the weights-file contract: a
// weights file stores its FeatureNames and is rejected when they differ from EduFeatureNames.

/** The one-hot language slices (the curriculum slices of the eval suite, plus "other"). */
export const EduSlices = ["ts", "js", "html", "css", "go", "sql", "other"] as const;
export type EduSlice = (typeof EduSlices)[number];

// Language tags as they appear on documents — LangForPath names ("typescript"), extensions ("tsx")
// and spec slice names ("ts") — mapped to the one-hot slice.
const SliceByLang: Record<string, EduSlice> = {
  ts: "ts", tsx: "ts", mts: "ts", cts: "ts", typescript: "ts",
  js: "js", jsx: "js", mjs: "js", cjs: "js", javascript: "js",
  html: "html", htm: "html", vue: "html", svelte: "html", astro: "html",
  css: "css", scss: "css", sass: "css", less: "css", styl: "css", stylus: "css",
  go: "go", golang: "go",
  sql: "sql", pgsql: "sql", plpgsql: "sql",
};

export function EduSliceForLang(Lang: string): EduSlice {
  return SliceByLang[Lang.trim().toLowerCase()] ?? "other";
}

const CoreFeatureNames = [
  "CommentRatio", // comment-only lines / non-empty lines
  "DocCommentCoverage", // JSDoc blocks + docstrings per declared function/class, capped at 1
  "AvgIdentifierLength", // mean identifier length (minified code ≈ 1–2)
  "IdentifierTokenRatio", // identifiers / all code tokens
  "FunctionsPerKb", // function/class/method declarations per KB, capped
  "NestingExcess", // max(0, max nesting depth - ComfortableNestingDepth)
  "AvgLineLength", // mean non-empty line length, capped
  "LogMaxLineLength", // ln(1 + longest line)
  "BlankLineRatio", // blank lines / all lines
  "StringLiteralRatio", // string-literal chars / non-comment content chars
  "NumberLiteralRatio", // numeric-literal chars / non-comment content chars
  "TodoDensity", // TODO/FIXME/HACK/XXX per 100 non-empty lines
  "HasTypeAnnotations", // 0/1: type annotations / declarations present
  "HasErrorHandling", // 0/1: try/catch/throw/raise/except/`if err != nil`/…
  "HasTestSignals", // 0/1: describe(/it(/expect(/func Test…
  "ConsoleLogDensity", // console.log/debug + debugger per 100 non-empty lines
] as const;
type CoreFeatureName = (typeof CoreFeatureNames)[number];
export type EduFeatureName = CoreFeatureName | `Lang_${EduSlice}`;
export type EduFeatures = Record<EduFeatureName, number>;

/** Every feature, in vector order: the core features, then the one-hot language slice. */
export const EduFeatureNames: readonly EduFeatureName[] = [...CoreFeatureNames, ...EduSlices.map((Slice): EduFeatureName => `Lang_${Slice}`)];

/** Nesting this deep is normal structure; only depth beyond it counts as "extreme". */
export const ComfortableNestingDepth = 5; // e.g. class → method → loop → branch → callback
/** Caps that keep a few pathological files (one 50 KB line, a 40-byte stub) from dominating. */
const MaxFunctionsPerKb = 8;
const MaxAvgLineLength = 400;
const BytesPerKb = 1024;
const PerHundredLines = 100;
const TabWidth = 4;

const IdentifierPattern = /[A-Za-z_$][\w$]*/g;
const TokenPattern = /[A-Za-z_$][\w$]*|\d[\w.]*|""|''|``|[^\s\w]/g;
const FunctionPattern =
  /\b(function|class|def|func|fn|fun|interface|struct|trait|impl)\b|=>|^[ \t]*(?:(?:public|private|protected|static|async|override|readonly|get|set)\s+)*(?!(?:if|for|while|switch|catch|return|function|else|do|with)\b)[A-Za-z_$][\w$]*\s*\([^)]*\)\s*(?::\s*[^{=;]+)?\{/gm;
const TodoPattern = /\b(TODO|FIXME|HACK|XXX)\b/g;
const ConsoleLogPattern = /\bconsole\.(log|debug)\s*\(|\bdebugger\s*;/g;
const TypeAnnotationPattern =
  /[\w)\]?]\s*:\s*(string|number|boolean|void|unknown|never|bigint|readonly\b|[A-Z][\w.]*(<[^>\n]*>)?(\[\])?)\s*[,)=;{|&]|\binterface\s+\w|\btype\s+\w+\s*(<[^>\n]*>)?\s*=|->\s*[\w[]|\bfunc\s+(\([^)]*\)\s*)?\w+\([^)]*\w+\s+[\w*[\]]+/;
const ErrorHandlingPattern = /\b(try|catch|throw|raise|except|finally|rescue)\b|if\s+err\s*!=\s*nil|\.catch\s*\(|\berrors\.(New|Is|As)\b|\bResult\s*</;

function CountMatches(Text: string, Pattern: RegExp): number {
  return (Text.match(Pattern) ?? []).length;
}

function MaxBraceDepth(Code: string): number {
  let Depth = 0;
  let Max = 0;
  for (const Char of Code) {
    if (Char === "{") Max = Math.max(Max, ++Depth);
    else if (Char === "}") Depth = Math.max(0, Depth - 1);
  }
  return Max;
}

/** Deepest indentation in units of the smallest indent the file uses (tabs = TabWidth columns). */
function MaxIndentDepth(Scan: SourceScan): number {
  const Indents: number[] = [];
  for (let I = 0; I < Scan.CodeLineTexts.length; I++) {
    if (!Scan.LineHasCode[I]) continue;
    const Leading = /^[ \t]*/.exec(Scan.CodeLineTexts[I]!)?.[0] ?? "";
    let Columns = 0;
    for (const Char of Leading) Columns += Char === "\t" ? TabWidth : 1;
    if (Columns > 0) Indents.push(Columns);
  }
  if (Indents.length === 0) return 0;
  const Unit = Math.min(...Indents);
  return Math.floor(Math.max(...Indents) / Unit);
}

/** The educational-value features of one document, deterministic for a given (Text, Lang). */
export function ExtractEduFeatures(Text: string, Lang: string, Scan: SourceScan = ScanSource(Text)): EduFeatures {
  const Code = Scan.CodeLineTexts.join("\n");
  const Lines = Text.split("\n");
  const NonEmpty = Lines.filter((Line) => Line.trim().length > 0);
  const NonEmptyCount = Math.max(1, NonEmpty.length);
  const Kb = Math.max(1, Buffer.byteLength(Text, "utf8")) / BytesPerKb;
  const Identifiers = Code.match(IdentifierPattern) ?? [];
  const Tokens = CountMatches(Code, TokenPattern);
  const Functions = CountMatches(Code, FunctionPattern);
  const Content = Math.max(1, Scan.CodeChars + Scan.StringChars);
  const MaxLine = NonEmpty.reduce((Max, Line) => Math.max(Max, Line.length), 0);
  const AvgLine = NonEmpty.reduce((Sum, Line) => Sum + Line.length, 0) / NonEmptyCount;
  const Depth = Math.max(MaxBraceDepth(Code), MaxIndentDepth(Scan));
  const Slice = EduSliceForLang(Lang);

  const Core: Record<CoreFeatureName, number> = {
    CommentRatio: CommentLineRatio(Scan),
    DocCommentCoverage: Math.min(1, Scan.DocComments / Math.max(1, Functions)),
    AvgIdentifierLength: Identifiers.length === 0 ? 0 : Identifiers.reduce((Sum, Id) => Sum + Id.length, 0) / Identifiers.length,
    IdentifierTokenRatio: Tokens === 0 ? 0 : Identifiers.length / Tokens,
    FunctionsPerKb: Math.min(MaxFunctionsPerKb, Functions / Kb),
    NestingExcess: Math.max(0, Depth - ComfortableNestingDepth),
    AvgLineLength: Math.min(MaxAvgLineLength, AvgLine),
    LogMaxLineLength: Math.log1p(MaxLine),
    BlankLineRatio: Lines.length === 0 ? 0 : (Lines.length - NonEmpty.length) / Lines.length,
    StringLiteralRatio: StringLiteralFraction(Scan),
    NumberLiteralRatio: Scan.NumberChars / Content,
    TodoDensity: (CountMatches(Text, TodoPattern) * PerHundredLines) / NonEmptyCount,
    HasTypeAnnotations: TypeAnnotationPattern.test(Code) ? 1 : 0,
    HasErrorHandling: ErrorHandlingPattern.test(Code) ? 1 : 0,
    HasTestSignals: HasTestSignals(Text) ? 1 : 0,
    ConsoleLogDensity: (CountMatches(Code, ConsoleLogPattern) * PerHundredLines) / NonEmptyCount,
  };
  const OneHot = Object.fromEntries(EduSlices.map((Each) => [`Lang_${Each}`, Each === Slice ? 1 : 0]));
  return { ...Core, ...OneHot } as EduFeatures;
}

/** The features as a vector in EduFeatureNames order (the order weights are stored in). */
export function EduFeatureVector(Features: EduFeatures): number[] {
  return EduFeatureNames.map((Name) => Features[Name]);
}
