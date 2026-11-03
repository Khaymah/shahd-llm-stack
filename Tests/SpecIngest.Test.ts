import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createTarGzip } from "nanotar";
import { ParseDataSpec } from "../Foundry/DataSpec.ts";
import type { DataSpec } from "../Foundry/DataSpec.ts";
import { InMemoryDocumentStore } from "../Foundry/InMemoryDocumentStore.ts";
import type { HttpJson, FetchBytes, RepoLicense } from "../Foundry/GitHubHttp.ts";
import { CreateGitHubRepoListProvider, SelectListedPath } from "../Foundry/GitHubRepoListProvider.ts";
import type { RepoListEntry } from "../Foundry/GitHubRepoListProvider.ts";
import { IngestSpec, MeasureSpec, PlanSpec, RenderSpecStatus, RenderSpecPlan, ValidateSpecFilter, BytesPerMb } from "../Foundry/SpecIngest.ts";
import { FileLicenseConflictTag } from "../Foundry/FileLicenseHeader.ts";

const Now = "2026-11-02T00:00:00.000Z";
const Quiet = (): void => {};

// A substantive TS file (> 300 chars, passes the content + quality gates), distinct per name.
function Code(Name: string): string {
  return `import { readFileSync, writeFileSync } from "node:fs";\n\nexport function ${Name}(path: string): string {\n  const raw = readFileSync(path, "utf8");\n  const lines = raw.split("\\n").filter((l) => l.trim().length > 0);\n  const normalized = lines.map((l) => l.trimEnd()).join("\\n");\n  return normalized;\n}\n\nexport function ${Name}ToFile(source: string, dest: string): void {\n  writeFileSync(dest, ${Name}(source));\n}\n\nexport const ${Name}Version = "1.0.0";\n`;
}

function GoCode(Name: string): string {
  return `package ${Name}\n\nimport (\n\t"errors"\n\t"strings"\n)\n\n// Normalize trims every line of the input and drops the empty ones.\nfunc Normalize${Name}(input string) (string, error) {\n\tif input == "" {\n\t\treturn "", errors.New("empty input")\n\t}\n\tlines := strings.Split(input, "\\n")\n\tout := make([]string, 0, len(lines))\n\tfor _, line := range lines {\n\t\tif trimmed := strings.TrimSpace(line); trimmed != "" {\n\t\t\tout = append(out, trimmed)\n\t\t}\n\t}\n\treturn strings.Join(out, "\\n"), nil\n}\n`;
}

// Prose with short lines (the code-quality scorer penalises long lines) and > 300 characters.
function Markdown(Title: string): string {
  return `# ${Title}\n\nThis guide explains how routing works in the framework.\nEach route maps a path pattern to a handler function.\nHandlers receive a context object and return a response.\n\n## Middleware\n\nMiddleware runs before the handler and can short-circuit.\nUse it for authentication, logging and request parsing.\nKeep middleware small so it stays easy to test and reuse.\n`;
}

const GplHeader = "// SPDX-License-Identifier: GPL-3.0-only\n// This file is part of a GNU General Public License project.\n\n";

type FakeRepo = { License: string; Files: Record<string, string> };

/** A fake GitHub: /repos/{name} metadata + /tarball/{ref} bytes, counting tarball downloads. */
async function FakeGitHub(Repos: Record<string, FakeRepo>): Promise<{ Http: HttpJson; FetchBytes: FetchBytes; Downloads: string[] }> {
  const Tarballs = new Map<string, Uint8Array>();
  for (const [Name, Repo] of Object.entries(Repos)) {
    const Root = `${Name.replace("/", "-")}-abc123`;
    Tarballs.set(Name, await createTarGzip(Object.entries(Repo.Files).map(([Path, Data]) => ({ name: `${Root}/${Path}`, data: Data }))));
  }
  const Downloads: string[] = [];
  const Http: HttpJson = async (Url) => {
    const Name = Url.replace("https://api.github.com/repos/", "");
    const Repo = Repos[Name];
    if (Repo === undefined) throw new Error("GitHub API 404");
    return { default_branch: "main", license: { spdx_id: Repo.License } };
  };
  const FetchBytes: FetchBytes = async (Url) => {
    const Name = Url.replace("https://api.github.com/repos/", "").replace(/\/tarball\/.*$/, "");
    Downloads.push(Name);
    const Tar = Tarballs.get(Name);
    if (Tar === undefined) throw new Error("GitHub API 404");
    return Tar;
  };
  return { Http, FetchBytes, Downloads };
}

function MakeSpec(Repos: unknown[], TsTargetMb = 10): DataSpec {
  return ParseDataSpec({
    Version: "test",
    Domain: "test",
    Slices: {
      ts: { TargetMb: TsTargetMb, Langs: ["typescript"] },
      go: { TargetMb: 10, Langs: ["go"] },
      docs: { TargetMb: 10, Langs: ["text-en", "markdown"] },
    },
    Exclusions: { PathGlobs: ["**/fixtures/**", "**/*.min.*"], MinFileBytes: 200, MaxFileBytes: 4000 },
    Training: { MaxEpochs: 4, ContextTokens: 1024, VocabSize: 16384 },
    Repos,
  });
}

const SliceOfLang = (Lang: string): string | null => ({ typescript: "ts", go: "go", markdown: "docs" })[Lang] ?? null;
const Bounds = { PathGlobs: ["**/fixtures/**", "**/*.min.*"], MinFileBytes: 200, MaxFileBytes: 4000 };

test("include/exclude globs, global PathGlobs and the built-in gate decide which paths are kept", () => {
  const Entry: RepoListEntry = { Repo: "acme/lib", Slice: "ts", Include: ["src/**"], Exclude: ["src/internal/**"] };
  const Kept = (Path: string): boolean => SelectListedPath(Path, Entry, Bounds.PathGlobs, SliceOfLang) !== null;
  expect(Kept("src/core/Parser.ts")).toBe(true);
  expect(Kept("src/internal/Secret.ts")).toBe(false); // entry Exclude
  expect(Kept("lib/Other.ts")).toBe(false); // not under Include
  expect(Kept("src/fixtures/Data.ts")).toBe(false); // spec PathGlobs
  expect(Kept("src/app.min.ts")).toBe(false); // spec PathGlobs
  expect(Kept("src/vendor/Lib.ts")).toBe(false); // built-in gate: vendored code
  expect(Kept("src/types.d.ts")).toBe(false); // built-in gate: declarations
  expect(Kept("src/__tests__/Parser.test.ts")).toBe(true); // tests are opted back in by the spec
  expect(Kept("src/main.go")).toBe(false); // go is not the entry's slice
  expect(Kept("src/docs/Guide.ts")).toBe(false); // "docs" not named by the Include -> stays excluded
  const Docs: RepoListEntry = { Repo: "acme/lib", Slice: "docs", Include: ["docs/**", "README.md"] };
  expect(SelectListedPath("docs/guide/Intro.md", Docs, Bounds.PathGlobs, SliceOfLang)).toEqual({ Lang: "markdown", Slice: "docs" });
  expect(SelectListedPath("README.md", Docs, Bounds.PathGlobs, SliceOfLang)).toEqual({ Lang: "markdown", Slice: "docs" });
  expect(SelectListedPath("docs/node_modules/x/README.md", Docs, Bounds.PathGlobs, SliceOfLang)).toBeNull();
  expect(SelectListedPath("docs/Example.ts", Docs, Bounds.PathGlobs, SliceOfLang)).toBeNull(); // code is not docs
});

test("provider: size bounds, docs .md handling, provenance, and ONE tarball download per repo", async () => {
  const Big = Array.from({ length: 20 }, (_, I) => Code(`big${I}`)).join("\n"); // > MaxFileBytes
  const GitHub = await FakeGitHub({
    "acme/lib": {
      License: "MIT",
      Files: {
        "src/Parser.ts": Code("parse"),
        "src/Tiny.ts": "export const x = 1;\n", // < MinFileBytes
        "src/Huge.ts": Big,
        "src/Engine.test.ts": Code("engineTest"),
        "docs/Routing.md": Markdown("Routing"),
        "README.md": Markdown("Lib"),
        "src/notes.md": Markdown("Notes"), // markdown outside the docs Include
      },
    },
  });
  expect(Buffer.byteLength(Big)).toBeGreaterThan(Bounds.MaxFileBytes);
  const Provider = CreateGitHubRepoListProvider(
    [
      { Repo: "acme/lib", Slice: "ts", Include: ["src/**"] },
      { Repo: "acme/lib", Slice: "docs", Include: ["docs/**", "README.md"] },
    ],
    { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Bounds, SliceOfLang, Log: Quiet },
  );
  const Docs = await Provider.Fetch("", 10);
  const ByPath = new Map(Docs.map((Doc) => [Doc.Provenance.split(":").pop() ?? "", Doc]));
  expect([...ByPath.keys()].sort()).toEqual(["README.md", "docs/Routing.md", "src/Engine.test.ts", "src/Parser.ts"]);
  expect(ByPath.get("docs/Routing.md")?.Lang).toBe("markdown");
  expect(ByPath.get("docs/Routing.md")?.Content).toBe(Markdown("Routing")); // prose kept as-is
  expect(ByPath.get("src/Parser.ts")).toMatchObject({ Source: "acme/lib", License: "MIT", Lang: "typescript", Origin: "web-permissive", Provenance: "acme/lib@main:src/Parser.ts" });
  expect(GitHub.Downloads).toEqual(["acme/lib"]); // two entries, one download
});

test("a per-file GPL header inside a permissive repo is stored Rejected (file_license_conflict)", async () => {
  const GitHub = await FakeGitHub({ "acme/lib": { License: "MIT", Files: { "src/Clean.ts": Code("clean"), "src/Gpl.ts": GplHeader + Code("gpl") } } });
  const Store = new InMemoryDocumentStore();
  const Spec = MakeSpec([{ Repo: "acme/lib", Slice: "ts", Include: ["src/**"], Note: "t" }]);
  await IngestSpec(Spec, Store, { IngestedAt: Now, Provider: { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Log: Quiet }, Log: Quiet });
  const Rows = await Store.All();
  const Gpl = Rows.find((Row) => Row.Provenance.endsWith("src/Gpl.ts"));
  const Clean = Rows.find((Row) => Row.Provenance.endsWith("src/Clean.ts"));
  expect(Gpl?.Tier).toBe("Rejected");
  expect(Gpl?.License.startsWith(FileLicenseConflictTag)).toBe(true);
  expect(Gpl?.Content.startsWith("// SPDX-License-Identifier: GPL-3.0-only")).toBe(true); // raw evidence kept
  expect(Clean?.Tier).toBe("Filtered");
});

test("a slice stops collecting new repos once its Filtered bytes reach TargetMb", async () => {
  const GitHub = await FakeGitHub({
    "acme/one": { License: "MIT", Files: { "src/A.ts": Code("a"), "src/B.ts": Code("b") } },
    "acme/two": { License: "MIT", Files: { "src/C.ts": Code("c") } },
    "acme/gopher": { License: "MIT", Files: { "pkg/Norm.go": GoCode("norm") } },
  });
  const TargetBytes = 500; // one repo of two files overshoots it
  const Spec = MakeSpec(
    [
      { Repo: "acme/one", Slice: "ts", Note: "t" },
      { Repo: "acme/two", Slice: "ts", Note: "t" },
      { Repo: "acme/gopher", Slice: "go", Note: "t" },
    ],
    TargetBytes / BytesPerMb,
  );
  const Store = new InMemoryDocumentStore();
  const Result = await IngestSpec(Spec, Store, { IngestedAt: Now, Provider: { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Log: Quiet }, Log: Quiet });
  expect(GitHub.Downloads).toEqual(["acme/one", "acme/gopher"]); // acme/two skipped: ts slice is full
  const Ts = Result.After.Slices.find((Slice) => Slice.Slice === "ts");
  expect(Ts?.CollectedBytes).toBeGreaterThanOrEqual(TargetBytes);
  expect(Result.Stats.ByTier.Filtered).toBe(3);
  expect(PlanSpec(Spec, Result.After, {}).map((Row) => Row.Action)).toEqual(["done", "slice-full", "done"]);
});

test("--Status numbers come from the store, and re-runs skip collected entries unless Force", async () => {
  const GitHub = await FakeGitHub({
    "acme/lib": { License: "MIT", Files: { "src/A.ts": Code("a"), "docs/Guide.md": Markdown("Guide") } },
    "acme/later": { License: "MIT", Files: { "src/L.ts": Code("later") } },
  });
  const Spec = MakeSpec([
    { Repo: "acme/lib", Slice: "ts", Include: ["src/**"], Note: "t" },
    { Repo: "acme/lib", Slice: "docs", Include: ["docs/**"], Note: "t" },
    { Repo: "acme/later", Slice: "ts", Note: "t" },
  ]);
  const Store = new InMemoryDocumentStore();
  const Provider = { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Log: Quiet };
  await IngestSpec(Spec, Store, { IngestedAt: Now, Provider, Log: Quiet, Repo: "acme/lib" }); // --Repo filter
  expect(GitHub.Downloads).toEqual(["acme/lib"]);

  const Progress = await MeasureSpec(Spec, Store);
  const Filtered = (await Store.All()).filter((Row) => Row.Tier === "Filtered");
  const TsBytes = Filtered.filter((Row) => Row.Lang === "typescript").reduce((Sum, Row) => Sum + Row.Bytes, 0);
  const DocsBytes = Filtered.filter((Row) => Row.Lang === "markdown").reduce((Sum, Row) => Sum + Row.Bytes, 0);
  expect(Progress.Slices.find((Slice) => Slice.Slice === "ts")).toMatchObject({ CollectedBytes: TsBytes, Files: 1, EntriesDone: 1, EntriesTotal: 2, TargetBytes: 10 * BytesPerMb });
  expect(Progress.Slices.find((Slice) => Slice.Slice === "docs")).toMatchObject({ CollectedBytes: DocsBytes, Files: 1, EntriesDone: 1, EntriesTotal: 1 });
  expect(DocsBytes).toBeGreaterThan(0);
  const Table = RenderSpecStatus(Progress);
  expect(Table).toMatch(/^ts\s+10\.0\s+0\.0\s+0%\s+1\s+1\/2$/m);
  expect(Table).toMatch(/^docs\s+10\.0\s+0\.0\s+0%\s+1\s+1\/1$/m);
  expect(RenderSpecPlan(PlanSpec(Spec, Progress, {}))).toContain("1 entry to fetch");

  await IngestSpec(Spec, Store, { IngestedAt: Now, Provider, Log: Quiet, Repo: "acme/lib" }); // already collected
  expect(GitHub.Downloads).toEqual(["acme/lib"]);
  const Forced = await IngestSpec(Spec, Store, { IngestedAt: Now, Provider, Log: Quiet, Repo: "acme/lib", Force: true });
  expect(GitHub.Downloads).toEqual(["acme/lib", "acme/lib"]);
  expect(Forced.Stats.New).toBe(0); // same bytes -> content-hash duplicates, nothing double-counted
});

test("--Slice narrows the plan; unknown --Slice/--Repo are rejected instead of silently doing nothing", async () => {
  const Spec = MakeSpec([
    { Repo: "acme/lib", Slice: "ts", Note: "t" },
    { Repo: "acme/gopher", Slice: "go", Note: "t" },
    { Repo: "acme/mixed", Note: "unsliced: per-file slice" },
  ]);
  const Progress = await MeasureSpec(Spec, new InMemoryDocumentStore());
  expect(PlanSpec(Spec, Progress, { Slice: "go" }).map((Row) => Row.Action)).toEqual(["filtered", "fetch", "fetch"]);
  expect(() => ValidateSpecFilter(Spec, { Slice: "cobol" })).toThrow(/not a slice/);
  expect(() => ValidateSpecFilter(Spec, { Repo: "nobody/nothing" })).toThrow(/not listed/);
  expect(() => ValidateSpecFilter(Spec, { Slice: "go", Repo: "ACME/gopher" })).not.toThrow();
});

test("repo license gate: non-permissive repos are skipped before download; NOASSERTION is resolved from LICENSE", async () => {
  const GitHub = await FakeGitHub({
    "acme/copyleft": { License: "GPL-3.0", Files: { "src/A.ts": Code("a") } },
    "acme/unclear": { License: "NOASSERTION", Files: { "src/B.ts": Code("b") } },
  });
  const Bsd3 = readFileSync(join(import.meta.dir, "Fixtures", "Licenses", "Clean", "Bsd3.txt"), "utf8");
  const FetchLicense = async (Name: string): Promise<RepoLicense | null> => (Name === "acme/unclear" ? { Spdx: "NOASSERTION", Text: Bsd3 } : null);
  const Provider = CreateGitHubRepoListProvider(
    [
      { Repo: "acme/copyleft", Slice: "ts" },
      { Repo: "acme/unclear", Slice: "ts" },
    ],
    { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, FetchLicense, Bounds, SliceOfLang, Log: Quiet },
  );
  const Docs = await Provider.Fetch("", 10);
  expect(GitHub.Downloads).toEqual(["acme/unclear"]);
  expect(Docs.map((Doc) => Doc.License)).toEqual(["BSD-3-Clause"]);
});

test("a failing repo is logged and skipped; the rest of the list still runs", async () => {
  const GitHub = await FakeGitHub({ "acme/ok": { License: "MIT", Files: { "src/A.ts": Code("a") } } });
  const Lines: string[] = [];
  const Provider = CreateGitHubRepoListProvider(
    [
      { Repo: "acme/missing", Slice: "ts" },
      { Repo: "acme/ok", Slice: "ts" },
    ],
    { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Bounds, SliceOfLang, Log: (Line) => Lines.push(Line) },
  );
  const Docs = await Provider.Fetch("", 10);
  expect(Docs.map((Doc) => Doc.Source)).toEqual(["acme/ok"]);
  expect(Lines.some((Line) => Line.includes("skip acme/missing: error: GitHub API 404"))).toBe(true);
});

test("unsafe repo names and refs never reach a URL", async () => {
  const GitHub = await FakeGitHub({ "acme/ok": { License: "MIT", Files: { "src/A.ts": Code("a") } } });
  const Provider = CreateGitHubRepoListProvider(
    [
      { Repo: "../etc", Slice: "ts" },
      { Repo: "acme/ok", Slice: "ts", Ref: "../../admin" },
    ],
    { Http: GitHub.Http, FetchBytes: GitHub.FetchBytes, Bounds, SliceOfLang, Log: Quiet },
  );
  expect(await Provider.Fetch("", 10)).toEqual([]);
  expect(GitHub.Downloads).toEqual([]);
});
