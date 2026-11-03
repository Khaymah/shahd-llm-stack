// Per-file license headers. A repo's LICENSE says what the AUTHORS' code is under, but individual files
// routinely carry their own header — vendored GPL code inside an MIT repo, an AGPL "ee/" module, a
// Commons-Clause'd helper. Those headers are exactly what StripLicenseHeader removes before storage, so
// they must be read from the RAW content first. A copyleft / source-available / non-permissive header
// overrides the repo license (the file is rejected with reason file_license_conflict); a permissive
// SPDX tag that differs from a permissive repo license becomes that file's own license.

import { IsPermissive, PermissiveLicenses } from "../Brain/Data/LicenseManifest.ts";
import { NormalizeLicense } from "./SpdxDetector.ts";

export type FileLicenseHeader = { Spdx: string | null; Copyleft: boolean; Evidence: string | null };

// Headers live at the top of a file; 60 lines covers a full GPL/Apache banner plus a shebang/package line.
const HeaderScanLines = 60;

// Reject-reason / license tag for a file whose own header conflicts with permissive-only training.
export const FileLicenseConflictTag = "file_license_conflict";

// `SPDX-License-Identifier: <expression>` — the rest of the line, minus a trailing comment closer.
const SpdxTag = /SPDX-License-Identifier:[ \t]*([^\r\n]*)/gi;
const TrailingCommentCloser = /\s*(\*\/|-->|\*\)|-\}|\]\]|%>|#\}|"""|''').*$/;

// Copyleft / non-permissive phrases, in NormalizeLicense form (lowercase words, single spaces), so a
// banner wrapped across comment lines ("GNU General\n * Public License") still matches. Each pair is
// [normalized phrase, human-readable evidence label]. Whole-word matched.
const NonPermissivePhrases: readonly (readonly [string, string])[] = [
  ["gnu general public", "GNU General Public License"],
  ["affero", "GNU Affero General Public License"],
  ["lesser general public", "GNU Lesser General Public License"],
  ["library general public", "GNU Library General Public License"],
  ["gnu gpl", "GNU GPL"],
  ["gpl", "GPL"],
  ["gplv2", "GPL"],
  ["gplv3", "GPL"],
  ["lgpl", "LGPL"],
  ["agpl", "AGPL"],
  ["cc by nc", "CC BY-NC"],
  ["cc by sa", "CC BY-SA"],
  ["creative commons attribution noncommercial", "Creative Commons NonCommercial"],
  ["creative commons attribution non commercial", "Creative Commons NonCommercial"],
  ["creative commons noncommercial", "Creative Commons NonCommercial"],
  ["attribution sharealike", "Creative Commons ShareAlike"],
  ["attribution share alike", "Creative Commons ShareAlike"],
  ["server side public license", "Server Side Public License"],
  ["business source license", "Business Source License"],
  ["commons clause", "Commons Clause"],
  ["polyform", "PolyForm"],
  ["elastic license", "Elastic License"],
  ["mozilla public license", "Mozilla Public License"],
  ["eclipse public license", "Eclipse Public License"],
  ["european union public licence", "European Union Public Licence"],
  ["common development and distribution license", "CDDL"],
];

const SpdxOperators = new Set(["and", "or", "with"]);

/** Canonical-cased permissive SPDX id for a (case-insensitive) id, or null when not permissive. */
function PermissiveId(Id: string): string | null {
  if (IsPermissive(Id)) return Id;
  const Lower = Id.toLowerCase();
  for (const Known of PermissiveLicenses) if (Known.toLowerCase() === Lower) return Known;
  return null;
}

/**
 * Evaluate an SPDX expression. Returns the permissive id the file may be used under, or null.
 * Conservative: `A WITH exception` is judged on A; any AND (or a mix of AND/OR) requires EVERY id to
 * be permissive (the first is reported); a pure OR picks the first permissive alternative.
 */
function ResolveSpdxExpression(Expression: string): string | null {
  const Tokens = Expression.replace(/[()]/g, " ").split(/\s+/).filter((Token) => Token.length > 0);
  const Ids: string[] = [];
  const Operators = new Set<string>();
  for (let I = 0; I < Tokens.length; I++) {
    const Lower = Tokens[I].toLowerCase();
    if (!SpdxOperators.has(Lower)) {
      Ids.push(Tokens[I]);
      continue;
    }
    Operators.add(Lower);
    if (Lower === "with") I++; // skip the exception id itself
  }
  if (Ids.length === 0) return null;
  const Resolved = Ids.map(PermissiveId);
  if (Operators.has("and")) return Resolved.every((Id) => Id !== null) ? Resolved[0] : null;
  return Resolved.find((Id) => Id !== null) ?? null;
}

function FindPhrase(NormalizedHeader: string): string | null {
  const Padded = ` ${NormalizedHeader} `;
  for (const [Phrase, Label] of NonPermissivePhrases) if (Padded.includes(` ${Phrase} `)) return Label;
  return null;
}

/**
 * Scan the first HeaderScanLines lines of RAW file content (before any header stripping) for an
 * SPDX-License-Identifier and for copyleft / non-permissive license phrases. Copyleft is true for any
 * non-permissive signal (copyleft, share-alike, non-commercial, source-available, unknown SPDX id);
 * Evidence names it. Spdx is the permissive id the file declares (null when none / not permissive).
 * Precision-first: the whole head is scanned (code included), so a GPL string literal near the top of
 * a file also rejects it — a false reject is acceptable, a missed GPL header is not.
 */
export function DetectFileLicenseHeader(Content: string): FileLicenseHeader {
  const Head = Content.split(/\r?\n/, HeaderScanLines).join("\n");
  let Spdx: string | null = null;
  for (const Match of Head.matchAll(SpdxTag)) {
    const Expression = Match[1].replace(TrailingCommentCloser, "").trim();
    if (Expression.length === 0) continue;
    const Permissive = ResolveSpdxExpression(Expression);
    if (Permissive === null) return { Spdx: null, Copyleft: true, Evidence: `SPDX-License-Identifier: ${Expression}` };
    Spdx ??= Permissive;
  }
  // SPDX lines were judged precisely above ("GPL-2.0 OR MIT" may be used as MIT); every other line is
  // phrase-scanned.
  const Phrase = FindPhrase(NormalizeLicense(Head.replace(SpdxTag, " ")));
  if (Phrase !== null) return { Spdx: null, Copyleft: true, Evidence: Phrase };
  return { Spdx, Copyleft: false, Evidence: Spdx === null ? null : `SPDX-License-Identifier: ${Spdx}` };
}

export type FileLicenseDecision = { License: string; Conflict: boolean };

/**
 * The license to record for one file of a repo. A non-permissive header always wins (tagged
 * `file_license_conflict: <evidence>`, which no permissive check accepts → Rejected tier). A permissive
 * per-file SPDX replaces the repo license ONLY when the repo license is itself permissive — a file
 * header must never upgrade a non-permissive / unresolved repo (e.g. LICENSE = MIT + rider) to permissive.
 */
export function ResolveFileLicense(RepoLicense: string, Header: FileLicenseHeader | undefined): FileLicenseDecision {
  if (Header === undefined) return { License: RepoLicense, Conflict: false };
  if (Header.Copyleft) return { License: `${FileLicenseConflictTag}: ${Header.Evidence ?? "non-permissive header"}`, Conflict: true };
  if (Header.Spdx !== null && IsPermissive(RepoLicense)) return { License: Header.Spdx, Conflict: false };
  return { License: RepoLicense, Conflict: false };
}
