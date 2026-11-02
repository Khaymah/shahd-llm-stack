import { test, expect } from "bun:test";
import { DetectFileLicenseHeader, ResolveFileLicense, FileLicenseConflictTag } from "../Foundry/FileLicenseHeader.ts";

const Body = "export function Add(A: number, B: number): number {\n  return A + B;\n}\n";

test("no header -> no per-file license, no conflict", () => {
  expect(DetectFileLicenseHeader(Body)).toEqual({ Spdx: null, Copyleft: false, Evidence: null });
});

test("a permissive SPDX tag is reported with canonical casing", () => {
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: Apache-2.0\n" + Body)).toMatchObject({ Spdx: "Apache-2.0", Copyleft: false });
  expect(DetectFileLicenseHeader("/* SPDX-License-Identifier: mit */\n" + Body)).toMatchObject({ Spdx: "MIT", Copyleft: false });
  expect(DetectFileLicenseHeader("# SPDX-License-Identifier: BSD-3-Clause\n" + Body).Spdx).toBe("BSD-3-Clause");
});

test("a non-permissive SPDX tag is a conflict with the tag as evidence", () => {
  const Header = DetectFileLicenseHeader("// SPDX-License-Identifier: GPL-2.0-or-later\n" + Body);
  expect(Header).toEqual({ Spdx: null, Copyleft: true, Evidence: "SPDX-License-Identifier: GPL-2.0-or-later" });
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: LicenseRef-Proprietary\n" + Body).Copyleft).toBe(true);
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: MPL-2.0\n" + Body).Copyleft).toBe(true);
});

test("SPDX expressions: OR may pick a permissive alternative; AND needs every id permissive", () => {
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: GPL-2.0 OR MIT\n" + Body)).toMatchObject({ Spdx: "MIT", Copyleft: false });
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: (MIT OR Apache-2.0)\n" + Body)).toMatchObject({ Spdx: "MIT", Copyleft: false });
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: MIT AND GPL-3.0-only\n" + Body).Copyleft).toBe(true);
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n" + Body)).toMatchObject({ Spdx: "Apache-2.0", Copyleft: false });
  expect(DetectFileLicenseHeader("// SPDX-License-Identifier: GPL-2.0 WITH Classpath-exception-2.0\n" + Body).Copyleft).toBe(true);
});

test("copyleft / source-available phrases are detected even when wrapped across comment lines", () => {
  const Cases: [string, string][] = [
    ["/*\n * under the terms of the GNU General\n * Public License version 3\n */\n", "GNU General Public License"],
    ["# GNU Affero General Public License v3\n", "GNU Affero General Public License"],
    ["// GNU Lesser General Public License 2.1\n", "GNU Lesser General Public License"],
    ["// Licensed CC BY-NC 4.0\n", "CC BY-NC"],
    ["// Licensed CC BY-SA 4.0\n", "CC BY-SA"],
    ["// Creative Commons Attribution-NonCommercial 4.0 International\n", "Creative Commons NonCommercial"],
    ["// Server Side Public License, v 1\n", "Server Side Public License"],
    ["// Use of this software is governed by the Business Source License\n", "Business Source License"],
    ["// MIT with the \"Commons Clause\" License Condition v1.0\n", "Commons Clause"],
    ["// PolyForm Noncommercial License 1.0.0\n", "PolyForm"],
    ["// Licensed under the Elastic License 2.0\n", "Elastic License"],
  ];
  for (const [Header, Evidence] of Cases) {
    expect({ Header, Result: DetectFileLicenseHeader(Header + Body) }).toEqual({ Header, Result: { Spdx: null, Copyleft: true, Evidence } });
  }
});

test("a permissive banner (incl. 'All rights reserved') is not a conflict", () => {
  const Bsd = "/*\n * Copyright (c) 2019-2024 Foo Inc. and contributors\n * All rights reserved.\n * Use of this source code is governed by a BSD-style license.\n */\n";
  expect(DetectFileLicenseHeader(Bsd + Body).Copyleft).toBe(false);
});

test("only the first 60 lines are scanned", () => {
  const Late = "\n".repeat(70) + "// GNU General Public License\n";
  expect(DetectFileLicenseHeader(Body + Late).Copyleft).toBe(false);
});

test("ResolveFileLicense: conflict always wins; permissive override only for a permissive repo", () => {
  const Gpl = DetectFileLicenseHeader("// SPDX-License-Identifier: GPL-3.0-only\n" + Body);
  expect(ResolveFileLicense("MIT", Gpl)).toEqual({ License: `${FileLicenseConflictTag}: SPDX-License-Identifier: GPL-3.0-only`, Conflict: true });
  const Apache = DetectFileLicenseHeader("// SPDX-License-Identifier: Apache-2.0\n" + Body);
  expect(ResolveFileLicense("MIT", Apache)).toEqual({ License: "Apache-2.0", Conflict: false });
  expect(ResolveFileLicense("NOASSERTION", Apache)).toEqual({ License: "NOASSERTION", Conflict: false });
  expect(ResolveFileLicense("MIT", undefined)).toEqual({ License: "MIT", Conflict: false });
});
