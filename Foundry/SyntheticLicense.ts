// The License column of a synthetic conversation records BOTH things its rights depend on: the SPDX id of
// the corpus document it was grounded on (or UNGROUNDED) and the teacher model that wrote it —
// "<source>;teacher:<model>", e.g. "MIT;teacher:qwen2.5-coder-1.5b-instruct-q4_k_m". Tiering accepts it
// when the SOURCE is permissive (or our own code, or no source at all); the teacher is chosen
// open-weights with a license that allows training on its outputs (Plan §8) and is recorded for audit.

import { IsPermissive } from "../Brain/Data/LicenseManifest.ts";
import { UngroundedSourceLicense } from "../Brain/Sft/Synthetic/SynthTypes.ts";

const TeacherSeparator = ";teacher:";
/** Our own code's license tag (LocalRepoProvider's default). */
const OwnedLicense = "OWNED";

/** Compose the License column of a synthetic document. */
export function SyntheticLicense(SourceLicense: string, Teacher: string): string {
  return `${SourceLicense}${TeacherSeparator}${Teacher}`;
}

/** Split a synthetic License column; null when it does not carry a teacher. */
export function ParseSyntheticLicense(License: string): { Source: string; Teacher: string } | null {
  const At = License.indexOf(TeacherSeparator);
  if (At <= 0) return null;
  const Teacher = License.slice(At + TeacherSeparator.length);
  return Teacher === "" ? null : { Source: License.slice(0, At), Teacher };
}

/** May a conversation grounded on a document under this source license be trained on? */
export function IsAcceptedSyntheticSource(Source: string): boolean {
  return Source === UngroundedSourceLicense || Source === OwnedLicense || IsPermissive(Source);
}
