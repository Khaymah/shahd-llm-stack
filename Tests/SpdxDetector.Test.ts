import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { DetectSpdx, NormalizeLicense } from "../Foundry/FoundryBarrel.ts";

const FixtureDir = join(import.meta.dir, "Fixtures", "Licenses");
function Fixture(Name: string): string {
  return readFileSync(join(FixtureDir, Name), "utf8");
}

// Real-world adversarial fixtures drawn from the exact repos that made GitHub's own matcher return
// NOASSERTION — the ones a naive "permissive words are present" check would wrongly promote. These
// are the permanent regression gate; the legal safety of the whole backfill depends on them.

const MitBody =
  'Permission is hereby granted, free of charge, to any person obtaining a copy\n' +
  'of this software and associated documentation files (the "Software"), to deal\n' +
  'in the Software without restriction, including without limitation the rights\n' +
  'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n' +
  'copies of the Software, and to permit persons to whom the Software is\n' +
  'furnished to do so, subject to the following conditions:\n\n' +
  'The above copyright notice and this permission notice shall be included in all\n' +
  'copies or substantial portions of the Software.\n\n' +
  'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n' +
  'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n' +
  'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n' +
  'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n' +
  'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n' +
  'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n' +
  'SOFTWARE.\n';
const Mit = "MIT License\n\nCopyright (c) 2025 Someone\n\n" + MitBody;

// Canonical BSD-3 / Apache-notice fixtures. The earlier inline BSD-3 was truncated to half the license
// (no warranty disclaimer), which the template-coverage gate now rejects by design.
const Bsd3 = Fixture("Clean/Bsd3.txt");
const ApacheNotice = Fixture("Clean/Apache2Notice.txt");

test("clean permissive licenses are recognized (MIT/BSD-3/Apache-notice)", () => {
  expect(DetectSpdx(Mit)).toMatchObject({ Spdx: "MIT", Permissive: true });
  expect(DetectSpdx(Bsd3)).toMatchObject({ Spdx: "BSD-3-Clause", Permissive: true });
  expect(DetectSpdx(ApacheNotice)).toMatchObject({ Spdx: "Apache-2.0", Permissive: true });
});

test("plain MIT is NOT tripped by its own 'substantial portions of the Software' wording", () => {
  // Regression: an over-broad "portions of" disqualifier once rejected every MIT license.
  expect(NormalizeLicense(Mit)).toContain("substantial portions of the software");
  expect(DetectSpdx(Mit).Permissive).toBe(true);
});

test("MIT + Commons Clause rider is rejected (source-available, NOT permissive)", () => {
  const CommonsClause = Mit + "\n\n\"Commons Clause\" License Condition v1.0\n\nThe Software is provided to you by the Licensor under the License, as defined below, subject to the following condition. Without limiting other conditions in the License, the grant of rights under the License will not include, and the License does not grant to you, the right to Sell the Software.\n";
  expect(DetectSpdx(CommonsClause).Permissive).toBe(false);
});

test("open-core split ('Portions ... are licensed as follows') is rejected", () => {
  const OpenCore = "# License\n\nPortions of this software are licensed as follows:\n\n- All content under packages/ee is licensed under the Enterprise Edition license.\n- All third party components are licensed under their respective licenses.\n- All other content is licensed under the MIT license.\n\n" + MitBody;
  expect(DetectSpdx(OpenCore).Permissive).toBe(false);
});

test("modified Apache (dify-style) is rejected", () => {
  const ModifiedApache = "# Open Source License\n\nDify is licensed under a modified version of the Apache License 2.0, with the following additional conditions:\n\n1. Dify may be used commercially as a backend service for other applications, except for multi-tenant SaaS offerings.\n";
  expect(DetectSpdx(ModifiedApache).Permissive).toBe(false);
});

test("copyleft licenses are rejected (GPL/AGPL/MPL/CC)", () => {
  expect(DetectSpdx("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n\nThis program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License.").Permissive).toBe(false);
  expect(DetectSpdx("This project is licensed under the AGPL-3.0-or-later License.").Permissive).toBe(false);
  expect(DetectSpdx("Mozilla Public License Version 2.0\n\n1. Definitions").Permissive).toBe(false);
  expect(DetectSpdx("Creative Commons Attribution 4.0 International Public License").Permissive).toBe(false);
});

test("a permissive core buried in a giant bundled-license file is rejected (low coverage)", () => {
  // node.js-style: the repo's own code IS MIT, but the LICENSE file bundles hundreds of dependency
  // licenses. We cannot prove every ingested file is the MIT part, so precision-first rejects it.
  const Filler = "\n\nThis product bundles the following third party software, used under their respective notices.\n".repeat(60);
  const Bundle = "Node.js is licensed for use as follows:\n\n" + MitBody + Filler;
  expect(DetectSpdx(Bundle).Permissive).toBe(false);
});

test("empty / non-license text is rejected", () => {
  expect(DetectSpdx("").Permissive).toBe(false);
  expect(DetectSpdx("See LICENSE file.").Permissive).toBe(false);
  expect(DetectSpdx("GNU-AGPL-3.0.txt").Permissive).toBe(false);
});

// ---- Residual-text rule (template subtraction) — regression gate for the audit's rider probe. ----

test("audit riders: MIT/Apache + a restriction rider are NOT permissive", () => {
  for (const Name of ["MitGoodNotEvil.txt", "MitNoCommercialNoMl.txt", "MitAnti996.txt", "MitAnti996Short.txt", "ApacheNoticeHostedService.txt"]) {
    const Result = DetectSpdx(Fixture(`Riders/${Name}`));
    expect({ Name, Permissive: Result.Permissive }).toEqual({ Name, Permissive: false });
  }
});

test("a long rider is rejected as non-canonical text beyond the template (absolute cap)", () => {
  expect(DetectSpdx(Fixture("Riders/MitAnti996.txt")).Note).toBe("non-canonical text beyond template");
  expect(DetectSpdx(Fixture("Riders/ApacheNoticeHostedService.txt")).Note).toBe("non-canonical text beyond template");
});

test("clean canonical licenses with realistic copyright lines stay permissive", () => {
  const Expected: [string, string][] = [
    ["Mit.txt", "MIT"],
    ["Bsd3.txt", "BSD-3-Clause"],
    ["Apache2Notice.txt", "Apache-2.0"],
    ["Apache2Full.txt", "Apache-2.0"],
    ["Isc.txt", "ISC"],
    ["Unlicense.txt", "Unlicense"],
    ["Zlib.txt", "Zlib"],
    ["ZeroBsd.txt", "0BSD"],
  ];
  for (const [Name, Spdx] of Expected) {
    const Result = DetectSpdx(Fixture(`Clean/${Name}`));
    expect({ Name, Spdx: Result.Spdx, Permissive: Result.Permissive, Note: Result.Note }).toEqual({ Name, Spdx, Permissive: true, Note: "clean permissive template" });
  }
});

test("Apache full text without the APPENDIX is still the canonical full form", () => {
  const Full = Fixture("Clean/Apache2Full.txt");
  const NoAppendix = Full.slice(0, Full.indexOf("APPENDIX"));
  expect(DetectSpdx(NoAppendix)).toMatchObject({ Spdx: "Apache-2.0", Permissive: true });
});

test("Apache has a coverage gate: signature phrases + custom terms are not the Apache license", () => {
  const Fake = "Copyright 2024 Foo Inc.\n\nThis project uses Apache License Version 2.0 wording from www.apache.org/licenses but grants only evaluation rights.\n";
  expect(DetectSpdx(Fake)).toMatchObject({ Spdx: "Apache-2.0", Permissive: false });
  expect(DetectSpdx(Fake).Note).toContain("low template coverage");
});

test("Apache full text + appended rider is rejected", () => {
  const Rider = "\n\nADDITIONAL TERMS: you may not offer the Work as a hosted service.\n";
  expect(DetectSpdx(Fixture("Clean/Apache2Full.txt") + Rider).Permissive).toBe(false);
});

test("BSD-4-Clause (advertising clause) is NOT reported as BSD-3-Clause and is not permissive", () => {
  const Result = DetectSpdx(Fixture("Bsd4.txt"));
  expect(Result.Spdx).toBe("BSD-4-Clause");
  expect(Result.Permissive).toBe(false);
});

test("0BSD is detected as 0BSD (not ISC), and ISC stays ISC", () => {
  expect(DetectSpdx(Fixture("Clean/ZeroBsd.txt")).Spdx).toBe("0BSD");
  expect(DetectSpdx(Fixture("Clean/Isc.txt")).Spdx).toBe("ISC");
  // Untitled 0BSD — the old "zero clause" signature missed this entirely and called it ISC.
  const Untitled = Fixture("Clean/ZeroBsd.txt").replace("Zero-Clause BSD\n=============\n", "");
  expect(DetectSpdx(Untitled)).toMatchObject({ Spdx: "0BSD", Permissive: true });
});

test("'restriction' is a residual-only marker: canonical MIT's 'without restriction' is fine, an added one is not", () => {
  expect(DetectSpdx(Mit).Permissive).toBe(true);
  const Added = Mit + "\nAdditional restriction: no resale.\n";
  expect(DetectSpdx(Added)).toMatchObject({ Spdx: "MIT", Permissive: false });
});

test("a short rider inserted mid-template is caught by the marker layer", () => {
  const Inserted = Mit.replace("copies or substantial portions of the Software.\n", "copies or substantial portions of the Software.\nNo commercial use without a paid license.\n");
  expect(DetectSpdx(Inserted).Permissive).toBe(false);
});

test("attribution lines are exempt from the cap but still marker-checked", () => {
  const ManyHolders = "MIT License\n\n" + Array.from({ length: 12 }, (_, I) => `Copyright (c) ${2010 + I}-2024 Holder Number ${I} Inc. and contributors`).join("\n") + "\nAll rights reserved.\n\n" + MitBody;
  expect(DetectSpdx(ManyHolders)).toMatchObject({ Spdx: "MIT", Permissive: true });
  const SneakyCopyright = "MIT License\n\nCopyright (c) 2024 Foo Inc. Non-commercial use only.\n\n" + MitBody;
  expect(DetectSpdx(SneakyCopyright).Permissive).toBe(false);
  const GiantCopyright = "MIT License\n\nCopyright (c) 2024 Foo Inc. " + "Every deployment must be registered with Foo and reported every quarter. ".repeat(4) + "\n\n" + MitBody;
  expect(DetectSpdx(GiantCopyright).Permissive).toBe(false);
});

test("a year range such as 1996 does not trip the 996 marker", () => {
  const Old = "MIT License\n\nCopyright (c) 1996-2004 Old Corp\n\n" + MitBody;
  expect(DetectSpdx(Old)).toMatchObject({ Spdx: "MIT", Permissive: true });
});
