// Selecting GOOD code files from a repository (M6/M8). Naively taking the first files in git-tree
// order pulls in dotfiles, config, benchmarks, generated stubs — the low-value data that dilutes the
// corpus. These pure predicates keep only substantive source across a broad set of programming AND
// web/markup/style languages (html/css included):
//   - a wide extension allowlist,
//   - path exclusions for non-source dirs (tests, examples, dist, vendor, docs, dotfiles…),
//   - file exclusions for declarations/minified/generated/config/lockfiles,
//   - a content gate rejecting tiny stubs, license-header-only files, embedded blobs, and symbol soup.
// A rank puts real source dirs (src/lib/…) first.

const CodeExtension =
  /\.(ts|tsx|js|jsx|mjs|cjs|go|py|pyw|rb|php|java|kt|kts|scala|swift|rs|c|cc|cpp|cxx|cs|h|hh|hpp|dart|lua|ex|exs|elm|erl|hs|clj|cljs|cljc|ml|mli|fs|fsx|jl|r|pl|pm|nim|zig|cr|v|groovy|gradle|sql|sh|bash|zsh|ps1|vb|pas|html|htm|css|scss|sass|less|styl|vue|svelte|astro)$/i;

const LangByExtension: Record<string, string> = {
  ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  go: "go", py: "python", pyw: "python", rb: "ruby", php: "php", java: "java", kt: "kotlin", kts: "kotlin",
  scala: "scala", swift: "swift", rs: "rust", c: "c", cc: "cpp", cpp: "cpp", cxx: "cpp", cs: "csharp",
  h: "c", hh: "cpp", hpp: "cpp", dart: "dart", lua: "lua", ex: "elixir", exs: "elixir", elm: "elm",
  erl: "erlang", hs: "haskell", clj: "clojure", cljs: "clojure", cljc: "clojure", ml: "ocaml", mli: "ocaml",
  fs: "fsharp", fsx: "fsharp", jl: "julia", r: "r", pl: "perl", pm: "perl", nim: "nim", zig: "zig",
  cr: "crystal", v: "v", groovy: "groovy", gradle: "groovy", sql: "sql", sh: "shell", bash: "shell",
  zsh: "shell", ps1: "powershell", vb: "vbnet", pas: "pascal", html: "html", htm: "html", css: "css",
  scss: "scss", sass: "sass", less: "less", styl: "stylus", vue: "vue", svelte: "svelte", astro: "astro",
};

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
  "migration", "migrations", "locale", "locales", "i18n", "asset", "assets", "public", "static",
]);

const ExcludeFile = /(\.min\.[a-z0-9]+$|\.d\.ts$|\.test\.[a-z0-9]+$|\.spec\.[a-z0-9]+$|[.-]lock\.[a-z]+$|\.lock$|\.generated\.[a-z0-9]+$|\.gen\.[a-z0-9]+$|\.pb\.[a-z0-9]+$|\.snap$|\.map$)$/i;
const ConfigFile = /(^|\/)([^/]*\.config\.[a-z0-9]+|\.[a-z].*rc(\.[a-z]+)?|babel|webpack|rollup|vite\.config|jest\.config|tsconfig[^/]*\.json|setup\.py|conftest\.py)$/i;

export function LangForPath(Path: string): string {
  const Ext = (Path.split(".").pop() ?? "").toLowerCase();
  return LangByExtension[Ext] ?? "unknown";
}

/** Is this repo path a substantive source-code file worth training on? */
export function IsSubstantiveCodePath(Path: string): boolean {
  if (!CodeExtension.test(Path)) return false;
  if (ExcludeFile.test(Path)) return false;
  if (ConfigFile.test(Path)) return false;
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

/** Does the fetched content look like substantive, human-written code (not a stub/blob/comment header)? */
export function IsSubstantiveCodeContent(Text: string, MaxBytes = 512_000): boolean {
  const Trimmed = Text.trim();
  if (Trimmed.length < 300) return false; // stub / near-empty
  if (Trimmed.length > MaxBytes) return false; // beyond the size budget (likely generated/bundled)
  if (/data:(font|image|application)|;base64,/i.test(Text)) return false; // embedded blobs
  // Strip block + line comments; require enough real code left (not just a license header).
  const NoComments = Text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/^[ \t]*\/\/.*$/gm, " ")
    .replace(/^[ \t]*#.*$/gm, " ")
    .trim();
  if (NoComments.length < 200) return false;
  let Alpha = 0;
  for (let I = 0; I < Text.length; I++) {
    const Code = Text.charCodeAt(I);
    if ((Code >= 65 && Code <= 90) || (Code >= 97 && Code <= 122)) Alpha++;
  }
  return Alpha / Text.length >= 0.25;
}
