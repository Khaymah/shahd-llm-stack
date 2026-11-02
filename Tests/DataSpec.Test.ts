import { test, expect } from "bun:test";
import { LoadDataSpec, ParseDataSpec, SliceForLang, DefaultDataSpecPath } from "../Foundry/DataSpec.ts";
import { ListedLangForPath, MarkdownLang } from "../Foundry/GitHubRepoListProvider.ts";
import { EntryKey } from "../Foundry/SpecIngest.ts";

const Spec = LoadDataSpec(DefaultDataSpecPath);

// Extensions the curriculum covers; every code Lang in the spec must be produced by one of them, so a
// typo ("typescipt") or a tag CodeFileFilter never emits can't silently collect nothing.
const ProbeExtensions = ["ts", "tsx", "js", "jsx", "mjs", "cjs", "html", "htm", "vue", "svelte", "astro", "css", "scss", "sass", "less", "go", "sql", "md", "mdx"];
const ProseLangs: readonly string[] = ["text-en", MarkdownLang];
// Repos the licensing policy rules out (Data/Spec.md explains each) — they must never creep back in.
const PolicyExcludedRepos: readonly string[] = ["mdn/content", "oven-sh/bun", "tailwindlabs/tailwindcss.com", "microsoft/TypeScript-Website"];

test("Data/Spec.json parses against the DataSpec schema", () => {
  expect(Spec.Version.length).toBeGreaterThan(0);
  expect(Spec.Repos.length).toBeGreaterThan(0);
  expect(Spec.Exclusions.MinFileBytes).toBeLessThan(Spec.Exclusions.MaxFileBytes);
});

test("every repo is a well-formed owner/name and every glob compiles", () => {
  for (const Entry of Spec.Repos) {
    expect(Entry.Repo).toMatch(/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/);
    for (const Pattern of [...(Entry.Include ?? []), ...(Entry.Exclude ?? [])]) expect(() => new Bun.Glob(Pattern).match("a/b.ts")).not.toThrow();
  }
  for (const Pattern of Spec.Exclusions.PathGlobs) expect(() => new Bun.Glob(Pattern).match("a/b.ts")).not.toThrow();
});

test("every entry's Slice exists in Slices, and every entry explains itself in a Note", () => {
  const Slices = Object.keys(Spec.Slices);
  for (const Entry of Spec.Repos) {
    if (Entry.Slice !== undefined) expect(Slices).toContain(Entry.Slice);
    expect((Entry.Note ?? "").length).toBeGreaterThan(0);
  }
});

test("every slice Lang is a tag the collector actually emits, and belongs to exactly one slice", () => {
  const Emitted = new Set(ProbeExtensions.map((Ext) => ListedLangForPath(`file.${Ext}`)));
  const Seen = new Map<string, string>();
  for (const [Slice, Budget] of Object.entries(Spec.Slices)) {
    for (const Lang of Budget.Langs) {
      expect(Emitted.has(Lang) || ProseLangs.includes(Lang)).toBe(true);
      expect(Seen.get(Lang)).toBeUndefined(); // one home per tag, or SliceForLang would be ambiguous
      Seen.set(Lang, Slice);
      expect(SliceForLang(Spec, Lang)).toBe(Slice as ReturnType<typeof SliceForLang>);
    }
  }
});

test("no duplicate entries: a repo appears at most once per slice", () => {
  const Keys = Spec.Repos.map(EntryKey);
  expect(new Set(Keys).size).toBe(Keys.length);
});

test("repos excluded by the licensing policy are absent", () => {
  const Repos = new Set(Spec.Repos.map((Entry) => Entry.Repo.toLowerCase()));
  for (const Repo of PolicyExcludedRepos) expect(Repos.has(Repo.toLowerCase())).toBe(false);
});

test("ParseDataSpec rejects an inverted size window and an unknown slice", () => {
  const Base = JSON.parse(JSON.stringify(Spec)) as Record<string, unknown>;
  expect(() => ParseDataSpec({ ...Base, Exclusions: { PathGlobs: [], MinFileBytes: 500, MaxFileBytes: 100 } })).toThrow();
  expect(() => ParseDataSpec({ ...Base, Repos: [{ Repo: "a/b", Slice: "cobol" }] })).toThrow();
  expect(() => ParseDataSpec({ ...Base, Repos: [{ Repo: "not-a-repo" }] })).toThrow();
});
