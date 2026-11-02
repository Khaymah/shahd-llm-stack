// The eval suite as an "eval set" for training-data DECONTAMINATION (Brain/Data/Decontamination.ts):
// every string a case would show or check — prompts, references, tests, expected tool calls and the
// multi-turn contents — so a training document sharing a long n-gram with any of it is dropped before
// training and the suite stays an honest held-out measurement.
//
// Limitation (inherent to n-gram decontamination): a case text shorter than the n-gram size (13 code
// words by default) yields no n-grams and cannot be matched — short prompts like "What is 7 + 5?" are
// not protected this way; keep such cases out of authored training data by construction.

import { ProblemEvalDocs } from "../ProblemSet.ts";
import type { EvalCaseSet } from "./SuiteTypes.ts";
import { EvalCaseSetV1 } from "./Data/CaseSetV1.ts";

/** One decontamination document per case. */
export function SuiteEvalDocs(CaseSet: EvalCaseSet): string[] {
  const Docs: string[] = [];
  for (const C of CaseSet.Code) Docs.push([C.Prompt, C.Reference, C.Tests ?? ""].join("\n"));
  for (const C of CaseSet.Format) Docs.push(C.Messages.map((M) => M.Content).join("\n"));
  for (const C of CaseSet.Tools) Docs.push(`${C.User}\n${JSON.stringify({ name: C.Expected.Name, arguments: C.Expected.Arguments })}`);
  for (const C of CaseSet.MultiTurn) Docs.push([...C.Turns.map((T) => T.Content), ...(C.MustContain ?? [])].join("\n"));
  return Docs;
}

/** Everything training must be decontaminated against today: the coding-problem set (STaR / pass@k)
 *  plus the current frozen suite. The single list every training script uses. */
export function TrainingDecontaminationDocs(): string[] {
  return [...ProblemEvalDocs(), ...SuiteEvalDocs(EvalCaseSetV1)];
}
