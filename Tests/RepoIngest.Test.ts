import { test, expect } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createTarGzip } from "nanotar";
import {
  FetchRepoFiles,
  AssessRepo,
  CreateGitHubRepoProvider,
  CreateLocalRepoProvider,
  InMemoryDocumentStore,
  IngestFromWeb,
  IsSubstantiveCodePath,
  LangForPath,
} from "../Foundry/FoundryBarrel.ts";
import type { HttpJson, FetchBytes, RepoFile } from "../Foundry/FoundryBarrel.ts";

// A substantive source file (>300 chars, passes the content gate), distinct per name.
function Code(Name: string): string {
  return `import { readFileSync, writeFileSync } from "node:fs";\n\nexport function ${Name}(path: string): string {\n  const raw = readFileSync(path, "utf8");\n  const lines = raw.split("\\n").filter((l) => l.trim().length > 0);\n  const normalized = lines.map((l) => l.trimEnd()).join("\\n");\n  return normalized;\n}\n\nexport function ${Name}ToFile(source: string, dest: string): void {\n  writeFileSync(dest, ${Name}(source));\n}\n\nexport const ${Name}Version = "1.0.0";\n`;
}

function Tar(Files: { name: string; data: string }[]): Promise<Uint8Array> {
  return createTarGzip(Files);
}

test("FetchRepoFiles extracts substantive source and skips junk/markup/declarations", async () => {
  const Gz = await Tar([
    { name: "repo-sha/src/Parser.ts", data: Code("parse") },
    { name: "repo-sha/lib/Engine.go", data: Code("engine") },
    { name: "repo-sha/benchmarks/demo.css", data: "form { margin: 0; }\n" }, // junk dir + markup
    { name: "repo-sha/src/types.d.ts", data: "export type X = number;\n" }, // declaration file
    { name: "repo-sha/README.md", data: "# hi\n" }, // not code
    { name: "repo-sha/.eslintrc.js", data: "module.exports = {};\n" }, // dotfile config
  ]);
  const Fetch: FetchBytes = async () => Gz;
  const Files = await FetchRepoFiles("http://tar", Fetch);
  expect(Files.map((F) => F.Path).sort()).toEqual(["lib/Engine.go", "src/Parser.ts"]);
});

test("AssessRepo grades a structured, quality repo 'high' and a thin one 'low'", () => {
  const Good: RepoFile[] = ["A", "B", "C", "D", "E"].map((N) => ({ Path: `src/${N}.ts`, Content: Code(N) }));
  expect(AssessRepo(Good).Level).toBe("high");
  const Thin: RepoFile[] = [{ Path: "main.py", Content: Code("m") }];
  expect(AssessRepo(Thin).Level).toBe("low");
});

test("whole-repo provider ingests every file of a qualifying repo and skips low-level ones", async () => {
  const GoodTar = await Tar(["A", "B", "C", "D", "E"].map((N) => ({ name: `good-sha/src/${N}.ts`, data: Code(N) })));
  const ThinTar = await Tar([{ name: "thin-sha/main.py", data: Code("m") }]);

  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/good", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/thin", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/good") ? GoodTar : ThinTar);

  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: BytesFetcher, MinLevel: "medium" });
  const Docs = await Provider.Fetch("q", 5);
  expect(Docs.length).toBe(5); // all 5 files of the good repo
  expect(Docs.every((D) => D.Source === "acme/good")).toBe(true); // thin repo skipped
  expect(Docs.every((D) => D.Origin === "web-permissive" && D.License === "MIT")).toBe(true);
});

test("incremental collect: OnRepoReady stores each repo before the next is downloaded", async () => {
  const Tar1 = await Tar(["A", "B", "C"].map((N) => ({ name: `r1-sha/src/${N}.ts`, data: Code(N) })));
  const Tar2 = await Tar(["D", "E"].map((N) => ({ name: `r2-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/r1", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/r2", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/r1") ? Tar1 : Tar2);
  const Order: string[] = [];
  const Ready: { repo: string; count: number }[] = [];
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    OnRepo: (Info) => Order.push("assess:" + Info.Repo),
    OnRepoReady: async (Source, Files) => {
      Order.push("store:" + Source);
      Ready.push({ repo: Source, count: Files.length });
    },
  });
  const Ret = await Provider.Fetch("q", 5);
  expect(Ret.length).toBe(0); // incremental mode: nothing collected/returned
  expect(Ready).toEqual([{ repo: "acme/r1", count: 3 }, { repo: "acme/r2", count: 2 }]);
  // each repo is STORED before the next is ASSESSED — proves store-as-you-go, not collect-all-then-store
  expect(Order).toEqual(["assess:acme/r1", "store:acme/r1", "assess:acme/r2", "store:acme/r2"]);
});

test("GitHub multi-query: a ;-separated query runs each and grows past one query's 1000-cap", async () => {
  // Growth lever: two DISTINCT queries surface distinct repos, so one run collects both — the way the
  // code corpus grows beyond a single query's ceiling. The search URL carries q=<query>, so the mock
  // returns a different repo per query.
  const TarA = await Tar(["A", "B", "C", "D", "E"].map((N) => ({ name: `a-sha/src/${N}.ts`, data: Code(N) })));
  const TarB = await Tar(["F", "G", "H", "I", "J"].map((N) => ({ name: `b-sha/src/${N}.ts`, data: Code(N) })));
  const Seen: string[] = [];
  const Http: HttpJson = async (Url) => {
    if (Url.includes("q=alpha")) return { items: [{ full_name: "acme/a", default_branch: "main", license: { spdx_id: "MIT" } }] };
    if (Url.includes("q=beta")) return { items: [{ full_name: "acme/b", default_branch: "main", license: { spdx_id: "MIT" } }] };
    return { items: [] };
  };
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/a") ? TarA : TarB);
  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: BytesFetcher, MinLevel: "medium", OnRepo: (I) => { if (I.Ingested) Seen.push(I.Repo); } });
  const Docs = await Provider.Fetch("alpha;beta", 10);
  const Sources = new Set(Docs.map((D) => D.Source));
  expect(Sources.has("acme/a")).toBe(true);
  expect(Sources.has("acme/b")).toBe(true); // the SECOND query's repo is collected too
  expect(Docs.length).toBe(10); // 5 files from each repo
  expect(Seen).toEqual(["acme/a", "acme/b"]);
});

test("GitHub multi-query respects the shared MaxRepos budget across queries", async () => {
  // Budget = 1 repo. Query alpha alone returns 2 repos; only the first is looked at, and query beta
  // never runs — the MaxRepos cap is a single budget spent across all queries, not per-query.
  const TarA = await Tar(["A", "B", "C"].map((N) => ({ name: `a-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async (Url) => {
    if (Url.includes("q=alpha")) return { items: [
      { full_name: "acme/a1", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/a2", default_branch: "main", license: { spdx_id: "MIT" } },
    ] };
    if (Url.includes("q=beta")) return { items: [{ full_name: "acme/b", default_branch: "main", license: { spdx_id: "MIT" } }] };
    return { items: [] };
  };
  const Looked: string[] = [];
  const BytesFetcher: FetchBytes = async () => TarA;
  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: BytesFetcher, MinLevel: "medium", OnRepoStart: (R) => Looked.push(R) });
  const Docs = await Provider.Fetch("alpha;beta", 1);
  expect(Looked).toEqual(["acme/a1"]); // budget of 1 spent on the first repo; a2 and all of beta skipped
  expect(Docs.every((D) => D.Source === "acme/a1")).toBe(true);
});

test("resilient collect: one repo failing (e.g. rate-limit) does not abort the whole run", async () => {
  // The exact failure mode that silently killed collection: a tarball fetch throwing mid-loop. The
  // run must log+skip the bad repo and still store the good ones — not abandon everything.
  const GoodTar = await Tar(["A", "B", "C"].map((N) => ({ name: `ok-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/boom", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/ok", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => {
    if (Url.includes("acme/boom")) throw new Error("GitHub API 403"); // secondary rate limit
    return GoodTar;
  };
  const Reasons: (string | null)[] = [];
  const Logs: string[] = [];
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    OnRepo: (Info) => Reasons.push(Info.Reason ?? null),
    Log: (M) => Logs.push(M),
  });
  const Docs = await Provider.Fetch("q", 5);
  expect(Docs.length).toBe(3); // the good repo's files survived the bad repo's failure
  expect(Docs.every((D) => D.Source === "acme/ok")).toBe(true);
  expect(Reasons.some((R) => R !== null && R.includes("error: GitHub API 403"))).toBe(true); // failure reported, not swallowed
  expect(Logs.some((L) => L.includes("ERROR acme/boom"))).toBe(true); // and logged to the console trail
  expect(Logs.some((L) => L.includes("errored=1") && L.includes("stored=1"))).toBe(true); // summary is auditable
});

test("NOASSERTION license is resolved from the real LICENSE: permissive -> detected SPDX, else unchanged", async () => {
  const Tar1 = await Tar(["A", "B", "C"].map((N) => ({ name: `mit-sha/src/${N}.ts`, data: Code(N) })));
  const Tar2 = await Tar(["D", "E", "F"].map((N) => ({ name: `gpl-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/mitrepo", default_branch: "main", license: { spdx_id: "NOASSERTION" } },
      { full_name: "acme/gplrepo", default_branch: "main", license: { spdx_id: "NOASSERTION" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/mitrepo") ? Tar1 : Tar2);
  // Full canonical MIT (a truncated disclaimer now fails the template-coverage gate by design).
  const MitText = readFileSync(join(import.meta.dir, "Fixtures", "Licenses", "Clean", "Mit.txt"), "utf8");
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    FetchLicense: async (Name) => (Name.includes("mitrepo") ? { Spdx: "NOASSERTION", Text: MitText } : { Spdx: "NOASSERTION", Text: "GNU GENERAL PUBLIC LICENSE Version 3" }),
  });
  const Docs = await Provider.Fetch("q", 5);
  const Mit = Docs.filter((D) => D.Source === "acme/mitrepo");
  const Gpl = Docs.filter((D) => D.Source === "acme/gplrepo");
  expect(Mit.length).toBe(3);
  expect(Mit.every((D) => D.License === "MIT")).toBe(true); // verified permissive -> promoted to real SPDX
  expect(Gpl.every((D) => D.License === "NOASSERTION")).toBe(true); // copyleft stays unresolved -> Rejected on license
});

test("expanded languages: css/html/vue/sql are recognized as code (M8)", () => {
  expect(IsSubstantiveCodePath("src/styles/main.css")).toBe(true);
  expect(IsSubstantiveCodePath("src/Page.vue")).toBe(true);
  expect(IsSubstantiveCodePath("db/schema.sql")).toBe(true);
  expect(IsSubstantiveCodePath("web/index.html")).toBe(true);
  expect(LangForPath("x.css")).toBe("css");
  expect(LangForPath("x.vue")).toBe("vue");
});

test("local repo provider learns from OUR own code as 'owned' (trained on despite license)", async () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-repo-"));
  try {
    mkdirSync(join(Root, "src"), { recursive: true });
    for (const N of ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]) writeFileSync(join(Root, "src", `${N}.ts`), Code(N));
    writeFileSync(join(Root, "src", "styles.css"), "body { margin: 0; padding: 0; color: #333; background: #fff; }\n".repeat(12));
    const Store = new InMemoryDocumentStore();
    const Provider = CreateLocalRepoProvider({ Roots: [Root], License: "proprietary", MinLevel: "medium" });
    const Stats = await IngestFromWeb([Provider], [""], Store, "2026-07-13T00:00:00.000Z");
    expect(Stats.ByTier.Filtered).toBeGreaterThanOrEqual(5); // owned code is training-eligible despite "proprietary"
    const Docs = await Store.All();
    expect(Docs.every((D) => D.Origin === "owned")).toBe(true);
    expect(Docs.some((D) => D.Lang === "css")).toBe(true); // css ingested (expanded languages)
  } finally {
    rmSync(Root, { recursive: true, force: true });
  }
});

// ---- Per-file license headers (vendored copyleft inside a permissive repo) ----

const GplHeader =
  "/*\n * ringbuf.c - lock-free ring buffer\n *\n * Copyright (C) 2009 Some Kernel Hacker\n *\n" +
  " * This program is free software; you can redistribute it and/or modify\n * it under the terms of the GNU General Public License as published by\n" +
  " * the Free Software Foundation; either version 2 of the License, or\n * (at your option) any later version.\n */\n";

// A substantive C file (>300 chars of real code after the header is stripped).
function CCode(Name: string): string {
  return `#include <stdlib.h>\n#include <string.h>\n\nstruct ${Name} {\n  unsigned char *data;\n  size_t head;\n  size_t tail;\n  size_t size;\n};\n\nint ${Name}_push(struct ${Name} *rb, unsigned char value) {\n  size_t next = (rb->head + 1) % rb->size;\n  if (next == rb->tail) return -1;\n  rb->data[rb->head] = value;\n  rb->head = next;\n  return 0;\n}\n\nint ${Name}_pop(struct ${Name} *rb, unsigned char *out) {\n  if (rb->head == rb->tail) return -1;\n  *out = rb->data[rb->tail];\n  rb->tail = (rb->tail + 1) % rb->size;\n  return 0;\n}\n`;
}

test("a GPL-headed .c file inside an MIT repo is Rejected with file_license_conflict (header read before stripping)", async () => {
  const RepoTar = await Tar([
    ...["A", "B", "C", "D", "E"].map((N) => ({ name: `mixed-sha/src/${N}.ts`, data: Code(N) })),
    { name: "mixed-sha/src/ringbuf.c", data: GplHeader + CCode("ringbuf") },
  ]);
  const Http: HttpJson = async () => ({ items: [{ full_name: "acme/mixed", default_branch: "main", license: { spdx_id: "MIT" } }] });
  const Logs: string[] = [];
  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: async () => RepoTar, MinLevel: "medium", Log: (M) => Logs.push(M) });
  const Store = new InMemoryDocumentStore();
  await IngestFromWeb([Provider], ["q"], Store, "2026-07-13T00:00:00.000Z");
  const Docs = await Store.All();
  const Gpl = Docs.filter((D) => D.Provenance.endsWith("/src/ringbuf.c"));
  expect(Gpl.length).toBe(1);
  expect(Gpl[0].Tier).toBe("Rejected");
  expect(Gpl[0].RejectReason).toContain("file_license_conflict");
  expect(Gpl[0].RejectReason).toContain("GNU General Public License");
  expect(Gpl[0].Content).toContain("GNU General Public License"); // evidence kept, not stripped away
  const Rest = Docs.filter((D) => !D.Provenance.endsWith("/src/ringbuf.c"));
  expect(Rest.length).toBe(5);
  expect(Rest.every((D) => D.License === "MIT" && D.Tier === "Filtered")).toBe(true); // repo's own files unaffected
  expect(Logs.some((L) => L.includes("acme/mixed: 1 file(s) with a conflicting license header"))).toBe(true);
});

test("a permissive SPDX header differing from a permissive repo license becomes the file's license", async () => {
  const RepoTar = await Tar([
    ...["A", "B", "C", "D"].map((N) => ({ name: `spdx-sha/src/${N}.ts`, data: Code(N) })),
    { name: "spdx-sha/src/Vendored.ts", data: "// SPDX-License-Identifier: Apache-2.0\n" + Code("Vendored") },
  ]);
  const Http: HttpJson = async () => ({ items: [{ full_name: "acme/spdx", default_branch: "main", license: { spdx_id: "MIT" } }] });
  const Docs = await CreateGitHubRepoProvider({ Http, FetchBytes: async () => RepoTar, MinLevel: "medium" }).Fetch("q", 5);
  expect(Docs.find((D) => D.Provenance.endsWith("/src/Vendored.ts"))?.License).toBe("Apache-2.0");
  expect(Docs.filter((D) => D.License === "MIT").length).toBe(4);
});

test("a per-file permissive SPDX header never upgrades an unresolved repo license", async () => {
  const RepoTar = await Tar(["A", "B", "C", "D", "E"].map((N) => ({ name: `nr-sha/src/${N}.ts`, data: "// SPDX-License-Identifier: MIT\n" + Code(N) })));
  const Http: HttpJson = async () => ({ items: [{ full_name: "acme/rider", default_branch: "main", license: { spdx_id: "NOASSERTION" } }] });
  const Rider = readFileSync(join(import.meta.dir, "Fixtures", "Licenses", "Riders", "MitNoCommercialNoMl.txt"), "utf8");
  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: async () => RepoTar, MinLevel: "medium", FetchLicense: async () => ({ Spdx: "NOASSERTION", Text: Rider }) });
  const Docs = await Provider.Fetch("q", 5);
  expect(Docs.length).toBe(5);
  expect(Docs.every((D) => D.License === "NOASSERTION")).toBe(true); // rider LICENSE stays unresolved -> Rejected
});

test("local repo provider: a GPL-headed vendored file in OUR repo is not 'owned' and is Rejected", async () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-repo-gpl-"));
  try {
    mkdirSync(join(Root, "src"), { recursive: true });
    for (const N of ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]) writeFileSync(join(Root, "src", `${N}.ts`), Code(N));
    writeFileSync(join(Root, "src", "ringbuf.c"), GplHeader + CCode("ringbuf"));
    const Store = new InMemoryDocumentStore();
    const Provider = CreateLocalRepoProvider({ Roots: [Root], License: "proprietary", MinLevel: "medium" });
    await IngestFromWeb([Provider], [""], Store, "2026-07-13T00:00:00.000Z");
    const Docs = await Store.All();
    const Gpl = Docs.find((D) => D.Provenance.endsWith("/src/ringbuf.c"));
    expect(Gpl?.Tier).toBe("Rejected");
    expect(Gpl?.RejectReason).toContain("file_license_conflict");
    expect(Docs.filter((D) => D.Origin === "owned").length).toBe(5);
  } finally {
    rmSync(Root, { recursive: true, force: true });
  }
});

test("vendored / third-party / build-output directories are excluded", () => {
  for (const Dir of ["external", "extern", "3rdparty", "third-party", "thirdparty", "third_party", "contrib", "node_modules", "bower_components", "Pods", "Carthage", ".yarn", "dist", "build", "out", "target", "vendor"]) {
    expect({ Dir, Kept: IsSubstantiveCodePath(`${Dir}/lib/zlib.c`) }).toEqual({ Dir, Kept: false });
  }
  expect(IsSubstantiveCodePath("src/lib/zlib.c")).toBe(true);
});
