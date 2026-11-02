// Persisting a SuiteReport: parsing one back from JSON (a --Baseline file, a checkpoint's Meta) with a
// Zod schema mirroring SuiteTypes.ts, and recording a report in a checkpoint FILE under Meta.Eval —
// the plan's rule that every checkpoint carries its suite results.
//
// Safe to rewrite Meta: the checkpoint checksum covers only the numeric payload (weights, optimizer,
// config — CheckpointFormat.ChecksumPayload), never Meta, and the write is atomic (temp + rename via
// WriteCheckpointObject, previous file kept as .bak). The write is verified by loading the file back
// through the checksum-checking reader and comparing the stored report.

import { z } from "zod";
import { isDeepStrictEqual } from "node:util";
import type { Checkpoint } from "../../Checkpoint/CheckpointFormat.ts";
import { LoadCheckpoint } from "../../Checkpoint/CheckpointReader.ts";
import { WriteCheckpointObject } from "../../Checkpoint/CheckpointWriter.ts";
import type { EvalSlice, MetricResult, SuiteReport } from "./SuiteTypes.ts";
import { EvalSlices } from "./SuiteTypes.ts";

/** The checkpoint Meta key a report is stored under. */
export const EvalMetaKey = "Eval";

const MetricSchema = z.object({
  Name: z.string(),
  Value: z.number(),
  Higher: z.enum(["better", "worse"]),
  N: z.number(),
  Details: z.record(z.unknown()).optional(),
});

const SliceSchema = z.enum(EvalSlices as [EvalSlice, ...EvalSlice[]]);

const ReportSchema = z.object({
  SuiteVersion: z.string(),
  Checkpoint: z.string(),
  ConfigHash: z.string(),
  Params: z.number(),
  RanAt: z.string(),
  DurationMs: z.number(),
  Metrics: z.array(MetricSchema),
  Slices: z.record(SliceSchema, z.array(MetricSchema)),
});

/** Validate an unknown value as a SuiteReport (throws with the Zod issues on a mismatch). */
export function ParseSuiteReport(Value: unknown): SuiteReport {
  const Parsed = ReportSchema.parse(Value);
  const Slices: Partial<Record<EvalSlice, MetricResult[]>> = {};
  for (const Slice of EvalSlices) {
    const List = Parsed.Slices[Slice];
    if (List !== undefined) Slices[Slice] = List;
  }
  return { ...Parsed, Slices };
}

/** A copy of the checkpoint with Report under Meta.Eval (the payload, and so the checksum, untouched). */
export function AttachEvalReport(Ckpt: Checkpoint, Report: SuiteReport): Checkpoint {
  return { ...Ckpt, Meta: { ...Ckpt.Meta, [EvalMetaKey]: Report } };
}

/** The report stored in a checkpoint's Meta, or null when it has none. */
export function ReadEvalReport(Ckpt: Checkpoint): SuiteReport | null {
  const Stored = Ckpt.Meta[EvalMetaKey];
  return Stored === undefined ? null : ParseSuiteReport(Stored);
}

/** Store Report in the checkpoint FILE at Path (atomic), then load it back and verify it round-trips. */
export function WriteEvalReportToFile(Path: string, Report: SuiteReport): void {
  WriteCheckpointObject(Path, AttachEvalReport(LoadCheckpoint(Path), Report));
  const Back = ReadEvalReport(LoadCheckpoint(Path)); // LoadCheckpoint re-verifies the payload checksum
  if (Back === null || !isDeepStrictEqual(Back, ParseSuiteReport(JSON.parse(JSON.stringify(Report))))) {
    throw new Error(`WriteEvalReportToFile: the report read back from ${Path} does not match what was written`);
  }
}
