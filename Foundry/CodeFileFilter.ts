// Selecting GOOD code files from a repository (M6/M8, Plan: Brain-Strengthening §2.2 layer 1).
// Naively taking the first files in git-tree order pulls in dotfiles, config, benchmarks, generated
// stubs — the low-value data that dilutes the corpus. These pure predicates keep only substantive
// source across a broad set of programming AND web/markup/style languages (html/css included):
//   path gate (IsSubstantiveCodePath):
//     - an extension allowlist per language (optionally narrowed to some languages),
//     - path exclusions for non-source dirs (tests, examples, dist, vendor, docs, dotfiles…),
//     - file exclusions for declarations/minified/generated/config/lockfiles/snapshots/locale files;
//   content gate (CodeContentVerdict / IsSubstantiveCodeContent):
//     - a file-size window (default 200 B … 50 KB, overridable — the spec ingest passes its own),
//     - embedded blobs, generated-file banners, minified layout,
//     - license-header-only files and symbol soup,
//     - files that are >90% string literals or data-table lines (i18n tables, fixtures, dumps),
//     - "imports-only" files (≥80% of code lines wire modules together: barrels, re-export hubs),
//     - files with no function/class/const/rule/statement declaration at all.
//   The code-STRUCTURE checks (minified layout, string/data mass, imports-only, no declaration) are
//   skipped for prose documents (README/guides collected for a docs slice): when the caller names a
//   non-code Lang (e.g. "markdown"), or — when no Lang is given — when most text lines read as
//   sentences (LooksLikeProse). A paragraph is one long line and "don't" opens no string there.
// A rank puts real source dirs (src/lib/…) first.

import { DataLineFraction, HasDeclaration, ImportLineFraction, ScanSource, StringLiteralFraction } from "../Brain/Data/EducationalFeatures.ts";
import type { SourceScan } from "../Brain/Data/EducationalFeatures.ts";

/** Knobs for both gates. Every field is optional; omitted fields use DefaultFileFilter. */
export type FileFilterOptions = {
  MinBytes?: number; // UTF-8 bytes; smaller files are stubs
  MaxBytes?: number; // UTF-8 bytes; larger files are usually generated/bundled
  /** LangForPath tags allowed through the path gate (e.g. ["typescript","javascript"]). Default: all. */
  Langs?: readonly string[];
  /** Content gate: the file's language tag. A non-code tag (e.g. "markdown") marks it as prose. */
  Lang?: string;
};

export const DefaultFileFilter = { MinBytes: 200, MaxBytes: 50 * 1024 } as const;

// Content-gate thresholds (fractions are of the relevant line/char population, see each check).
const MinCodeChars = 150; // non-space code+string chars left after comments are stripped
const MinAlphaFraction = 0.25;
const MaxLiteralFraction = 0.9; // string-literal chars, or data-table lines
const MaxImportLineFraction = 0.8;
const MinifiedLineLength = 500; // a line this long is minified/bundled output…
const MaxMinifiedByteFraction = 0.5; // …and when such lines hold half the bytes, the file is
const GeneratedBannerChars = 1000; // generated-file banners live at the top of the file
const MinProseWords = 6; // a text line with this many words and no code punctuation is a sentence…
const MinProseLineFraction = 0.5; // …and when half the text lines are sentences, the file is prose
const ProseWord = /\b[A-Za-z][a-z]+\b/g;
const CodePunctuation = /[;{}=]/;
const ListOrTableRow = /^\s*["'`]|[,;]\s*$/; // `"key": "a sentence",` rows of a string table are not prose
// Markdown structure: a fenced code block, or headings together with a link. (Headings alone are not
// enough — `# comment` lines are ordinary Python/shell comments.)
const MarkdownFence = /^[ \t]*```[\w+-]*[ \t]*$/m;
const MarkdownHeading = /^#{1,6} \S/gm;
const MarkdownLink = /\]\([^)\s]+\)/;
const MinMarkdownHeadings = 2;

// Extension → LangForPath tag. The KEYS are the extension allowlist; the VALUES are the language
// families the allowlist can be narrowed to (FileFilterOptions.Langs).
const LangByExtension: Record<string, string> = {
  ts: "typescript", tsx: "typescript", mts: "typescript", cts: "typescript",
  js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  go: "go", py: "python", pyw: "python", rb: "ruby", php: "php", java: "java", kt: "kotlin", kts: "kotlin",
  scala: "scala", swift: "swift", rs: "rust", c: "c", cc: "cpp", cpp: "cpp", cxx: "cpp", cs: "csharp",
  h: "c", hh: "cpp", hpp: "cpp", dart: "dart", lua: "lua", ex: "elixir", exs: "elixir", elm: "elm",
  erl: "erlang", hs: "haskell", clj: "clojure", cljs: "clojure", cljc: "clojure", ml: "ocaml", mli: "ocaml",
  fs: "fsharp", fsx: "fsharp", jl: "julia", r: "r", pl: "perl", pm: "perl", nim: "nim", zig: "zig",
  cr: "crystal", v: "v", groovy: "groovy", gradle: "groovy", sql: "sql", sh: "shell", bash: "shell",
  zsh: "shell", ps1: "powershell", vb: "vbnet", pas: "pascal", html: "html", htm: "html", css: "css",
  scss: "scss", sass: "sass", less: "less", styl: "stylus", vue: "vue", svelte: "svelte", astro: "astro",
};

// Every tag a document's Lang may carry for source code: LangForPath names AND bare extensions
// (callers and tests use both, e.g. "typescript" and "ts").
const CodeLangs = new Set([...Object.keys(LangByExtension), ...Object.values(LangByExtension)]);

// A path segment that means "not substantive product source". Compared lowercased, so entries are
// lowercase ("pods" = CocoaPods' Pods/, "carthage" = Carthage/). Vendored / third-party directories
// matter for licensing too: their code is under ITS OWN license, not the repo's. Dot-dirs (".yarn",
// ".git" …) are excluded separately by the leading-"." rule in IsSubstantiveCodePath.
const ExcludeSegment = new Set([
  "node_modules", "bower_components", "dist", "build", "out", "bin", "obj", "target", "vendor", "third_party",
  "third-party", "thirdparty", "3rdparty", "external", "extern", "contrib", "pods", "carthage", "deps",
  "coverage", "__tests__", "__mocks__", "__snapshots__", "__pycache__", "test", "tests", "spec", "specs",
  "e2e", "example", "examples", "sample", "samples", "demo", "demos", "benchmark", "benchmarks", "bench",
  "fixture", "fixtures", "mock", "mocks", "doc", "docs", "website", "site", "generated", "gen", "proto",
  "migration", "migrations", "locale", "locales", "i18n", "l10n", "translations", "asset", "assets", "public",
  "static",
]);

// Declarations, minified/bundled, test/spec files, lockfiles, snapshots, source maps, generated code
// (protobuf/grpc stubs, Go zz_generated, Dart build_runner, *_generated / *.gen / *.generated).
const ExcludeFile =
  /(\.min\.[a-z0-9]+$|\.d\.[mc]?ts$|\.test\.[a-z0-9]+$|\.spec\.[a-z0-9]+$|[.-]lock\.[a-z]+$|\.lock$|\.lockb$|\.generated\.[a-z0-9]+$|\.gen\.[a-z0-9]+$|_gen\.go$|(^|\/)zz_generated[^/]*$|_generated\.[a-z0-9]+$|\.pb\.[a-z0-9]+$|_pb2(_grpc)?\.py$|_grpc\.pb\.[a-z0-9]+$|\.g\.dart$|\.freezed\.dart$|\.snap$|\.map$|(^|\/)(bundle|vendor|chunk-[^/]*)\.js$)$/i;
// Package-manager lockfiles by name (some have no ".lock" suffix).
const LockFile = /(^|\/)(package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.ya?ml|bun\.lockb?|Cargo\.lock|Gemfile\.lock|composer\.lock|poetry\.lock|Pipfile\.lock|go\.sum|mix\.lock|pubspec\.lock|packages\.lock\.json)$/i;
// Locale / translation files: `*.i18n.*`, `*.locale.*`, `messages_xx.*`, and locale-named files
// (`en.json`, `pt-BR.ts`, `zh_Hans.js`) — the latter only for data-ish extensions or under a
// translation-ish directory, so `go.ts` / `io.ts` / `db.go` stay code.
const LocaleFile =
  /(\.(i18n|locale|l10n|translations?)\.[a-z0-9]+$|(^|\/)messages[_.-][a-z]{2}([_-][A-Za-z]{2,4})?\.[a-z0-9]+$|(^|\/)[a-z]{2}([_-][A-Z][A-Za-z]{1,3})?\.(json|ya?ml|po|pot|properties|strings|resx|arb|xlf|xliff)$|(^|\/)(intl|nls|strings|resources|lang|langs|messages)\/[a-z]{2}([_-][A-Za-z]{2,4})?\.[a-z0-9]+$)/;
const ConfigFile = /(^|\/)([^/]*\.config\.[a-z0-9]+|\.[a-z].*rc(\.[a-z]+)?|babel|webpack|rollup|vite\.config|jest\.config|tsconfig[^/]*\.json|setup\.py|conftest\.py)$/i;
const GeneratedBanner = /@generated\b|Code generated .{0,80}DO NOT EDIT|\bauto-?generated\b|\bautomatically generated\b|\bThis file (is|was) generated\b|\bDO NOT EDIT\b/i;
const EmbeddedBlob = /data:(font|image|application)|;base64,/i;

export function LangForPath(Path: string): string {
  const Ext = (Path.split(".").pop() ?? "").toLowerCase();
  return LangByExtension[Ext] ?? "unknown";
}

/** Is this Lang tag a source-code language (a LangForPath name or a code extension)? */
export function IsCodeLang(Lang: string): boolean {
  return CodeLangs.has(Lang.trim().toLowerCase());
}

/** Is this repo path a substantive source-code file worth training on? */
export function IsSubstantiveCodePath(Path: string, Options: FileFilterOptions = {}): boolean {
  const Lang = LangForPath(Path);
  if (Lang === "unknown") return false; // extension allowlist
  if (Options.Langs !== undefined && !Options.Langs.includes(Lang)) return false;
  if (ExcludeFile.test(Path) || LockFile.test(Path) || LocaleFile.test(Path) || ConfigFile.test(Path)) return false;
  for (const Segment of Path.split("/")) {
    if (Segment.startsWith(".")) return false; // dotfiles / dot-dirs
    if (ExcludeSegment.has(Segment.toLowerCase())) return false;
  }
  return true;
}

/** Lower = preferred: real source directories first. */
export function RankCodePath(Path: string): number {
  return /(^|\/)(src|lib|source|app|packages|internal|pkg|core)\//i.test(Path) ? 0 : 1;
}

/** The content gate's decision, with the first failed check as a human-readable reason. */
export type CodeContentVerdict = { Passed: boolean; Reason: string | null };

function Reject(Reason: string): CodeContentVerdict {
  return { Passed: false, Reason };
}

function MinifiedByteFraction(Text: string): number {
  let Long = 0;
  for (const Line of Text.split("\n")) if (Line.length >= MinifiedLineLength) Long += Line.length;
  return Text.length === 0 ? 0 : Long / Text.length;
}

function AlphaFraction(Text: string): number {
  let Alpha = 0;
  for (let I = 0; I < Text.length; I++) {
    const Code = Text.charCodeAt(I);
    if ((Code >= 65 && Code <= 90) || (Code >= 97 && Code <= 122)) Alpha++;
  }
  return Text.length === 0 ? 0 : Alpha / Text.length;
}

/** Is this a prose document: markdown structure, or most text lines (the scanner's non-comment
 *  lines) reading as sentences rather than code? */
function LooksLikeProse(Text: string, Scan: SourceScan): boolean {
  if (MarkdownFence.test(Text)) return true;
  if ((Text.match(MarkdownHeading) ?? []).length >= MinMarkdownHeadings && MarkdownLink.test(Text)) return true;
  const Lines = Text.split("\n");
  let Prose = 0;
  for (let I = 0; I < Lines.length; I++) {
    if (!Scan.LineHasCode[I]) continue;
    const Line = Lines[I]!;
    if (!CodePunctuation.test(Line) && !ListOrTableRow.test(Line) && (Line.match(ProseWord) ?? []).length >= MinProseWords) Prose++;
  }
  return Scan.CodeLines > 0 && Prose / Scan.CodeLines >= MinProseLineFraction;
}

/** Normalize the legacy numeric MaxBytes argument and the options object into one shape. */
function ResolveFileFilter(Options: number | FileFilterOptions): { MinBytes: number; MaxBytes: number; Lang: string | undefined } {
  const Given: FileFilterOptions = typeof Options === "number" ? { MaxBytes: Options } : Options;
  return { MinBytes: Given.MinBytes ?? DefaultFileFilter.MinBytes, MaxBytes: Given.MaxBytes ?? DefaultFileFilter.MaxBytes, Lang: Given.Lang };
}

/** Does the fetched content look like substantive, human-written code? Returns the first failed
 *  check as Reason. Options may be the legacy MaxBytes number or a FileFilterOptions object. */
export function CodeContentVerdict(Text: string, Options: number | FileFilterOptions = {}): CodeContentVerdict {
  const { MinBytes, MaxBytes, Lang } = ResolveFileFilter(Options);
  const Bytes = Buffer.byteLength(Text.trim(), "utf8");
  if (Bytes < MinBytes) return Reject(`too small: ${Bytes} B < ${MinBytes} B`);
  if (Bytes > MaxBytes) return Reject(`too large: ${Bytes} B > ${MaxBytes} B (likely generated/bundled)`);
  if (EmbeddedBlob.test(Text)) return Reject("embedded binary blob (data:/base64)");
  if (GeneratedBanner.test(Text.slice(0, GeneratedBannerChars))) return Reject("generated-file banner");
  const Scan = ScanSource(Text);
  const Prose = Lang !== undefined ? !IsCodeLang(Lang) : LooksLikeProse(Text, Scan);
  const Minified = Prose ? 0 : MinifiedByteFraction(Text);
  if (Minified >= MaxMinifiedByteFraction) return Reject(`minified: ${(Minified * 100).toFixed(0)}% of bytes on lines >= ${MinifiedLineLength} chars`);

  // Enough real code left once comments are gone (a license header alone is not a source file).
  const CodeChars = Scan.CodeChars + Scan.StringChars;
  const NeededCode = Math.min(MinCodeChars, MinBytes);
  if (CodeChars < NeededCode) return Reject(`comments only: ${CodeChars} code chars < ${NeededCode}`);
  if (AlphaFraction(Text) < MinAlphaFraction) return Reject("symbol soup: alpha fraction < 0.25");
  if (Prose) return { Passed: true, Reason: null }; // the rest judges code structure
  const Strings = StringLiteralFraction(Scan);
  if (Strings > MaxLiteralFraction) return Reject(`string literals: ${(Strings * 100).toFixed(0)}% of content`);
  const DataLines = DataLineFraction(Scan);
  if (DataLines > MaxLiteralFraction) return Reject(`data table: ${(DataLines * 100).toFixed(0)}% of lines are literals only`);
  const Imports = ImportLineFraction(Scan);
  if (Imports >= MaxImportLineFraction) return Reject(`imports only: ${(Imports * 100).toFixed(0)}% of code lines are import/export/require`);
  if (!HasDeclaration(Scan)) return Reject("no function/class/const declaration");
  return { Passed: true, Reason: null };
}

/** Boolean form of CodeContentVerdict (the original signature: an optional MaxBytes number still works). */
export function IsSubstantiveCodeContent(Text: string, Options: number | FileFilterOptions = {}): boolean {
  return CodeContentVerdict(Text, Options).Passed;
}
