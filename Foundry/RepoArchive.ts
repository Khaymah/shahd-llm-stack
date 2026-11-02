// Download and extract a WHOLE repository in one request (M7). The GitHub tarball endpoint returns
// the entire repo as one gzipped tar, so we fetch it once (instead of hundreds of per-file API
// calls) and keep every substantive source file — the model learns from complete projects, not a
// scattered sample. Extraction reuses CodeFileFilter for path + content quality.
//
// PrepareRepoFile is the single entry point for EVERY repo provider (GitHub tarball and local disk):
// it reads the per-file license header from the RAW content BEFORE StripLicenseHeader can erase it,
// so a GPL/AGPL/source-available file vendored into a permissive repo is caught and rejected.

import { parseTarGzip } from "nanotar";
import type { FetchBytes } from "./GitHubHttp.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent } from "./CodeFileFilter.ts";
import { StripLicenseHeader } from "./ContentNormalizer.ts";
import { DetectFileLicenseHeader } from "./FileLicenseHeader.ts";
import type { FileLicenseHeader } from "./FileLicenseHeader.ts";

// Header is the file's own license header, read from the raw content (optional so hand-built RepoFile
// values — tests, quality assessment — stay valid; absent means "no per-file information").
export type RepoFile = { Path: string; Content: string; Header?: FileLicenseHeader };

// Generous defaults so a large but organized repo is learned WHOLE; the byte budget only guards
// against a pathological monorepo. MaxContentBytes caps a single file (larger ones are dropped, not
// truncated). Raise them for truly massive repos.
export type RepoLimits = { MaxFiles?: number; MaxBytes?: number; MaxContentBytes?: number };
export const DefaultRepoLimits: Required<RepoLimits> = { MaxFiles: 8000, MaxBytes: 64_000_000, MaxContentBytes: 512_000 };

/** GitHub tarballs prefix every entry with "{repo}-{sha}/"; drop that top-level directory. */
function StripRoot(Name: string): string {
  const Slash = Name.indexOf("/");
  return Slash === -1 ? Name : Name.slice(Slash + 1);
}

/**
 * Turn one raw repo file into a RepoFile, or null when it is not substantive code. The license header
 * is detected on the RAW text first. A conflicting (non-permissive) file keeps its raw content so the
 * Rejected record shows the evidence; every other file is stored with its license banner stripped.
 */
export function PrepareRepoFile(Path: string, Raw: string, MaxContentBytes: number): RepoFile | null {
  const Header = DetectFileLicenseHeader(Raw);
  const Stripped = StripLicenseHeader(Raw);
  if (!IsSubstantiveCodeContent(Stripped, MaxContentBytes)) return null;
  return { Path, Content: Header.Copyleft ? Raw : Stripped, Header };
}

/** Fetch a repo tarball and return its substantive source files (path-filtered + content-gated). */
export async function FetchRepoFiles(TarballUrl: string, Fetch: FetchBytes, Limits: RepoLimits = {}): Promise<RepoFile[]> {
  const MaxFiles = Limits.MaxFiles ?? DefaultRepoLimits.MaxFiles;
  const MaxBytes = Limits.MaxBytes ?? DefaultRepoLimits.MaxBytes;
  const MaxContentBytes = Limits.MaxContentBytes ?? DefaultRepoLimits.MaxContentBytes;
  const Items = await parseTarGzip(await Fetch(TarballUrl));
  const Out: RepoFile[] = [];
  let Bytes = 0;
  for (const Item of Items) {
    if (Item.type !== "file") continue;
    const Path = StripRoot(Item.name);
    if (!IsSubstantiveCodePath(Path)) continue;
    const File = PrepareRepoFile(Path, Item.text ?? new TextDecoder().decode(Item.data), MaxContentBytes);
    if (File === null) continue;
    Out.push(File);
    Bytes += File.Content.length;
    if (Out.length >= MaxFiles || Bytes >= MaxBytes) break;
  }
  return Out;
}
