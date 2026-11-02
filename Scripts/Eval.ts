// Run the fixed evaluation suite on one checkpoint (Plan: Brain-Strengthening §3 — "measure before
// training"). Loads the checkpoint (a FILE path, or a Postgres checkpoint NAME when DATABASE_URL is set),
// runs every metric (Brain/Eval/Suite/RunSuite.ts), prints the table — and the comparison against a
// baseline report when given — and writes the report to Runs/Eval/<checkpoint>-<timestamp>.json.
// With --WriteMeta the report is also stored under Meta.Eval in the checkpoint itself (verified by
// loading it back).
//
//   bun run eval --Checkpoint=Checkpoints/Last.ckpt --MaxCases=5
//   bun run eval --Checkpoint=Chat --Baseline=Runs/Eval/Chat-2026-11-01T10-00-00-000Z.json --WriteMeta
//
// Flags: --Checkpoint=<path|name> (required) · --Baseline=<report.json | checkpoint with Meta.Eval>
//        --HeldOut=<text file> (else the store's validation docs when DATABASE_URL is set, else the seed
//        corpus) · --TrainCorpus=<text file> (else the store's training-side Filtered docs; without
//        either the memorization probe is skipped) · --MaxCases=<n per category, 0 = all>
//        --CaseTimeoutMs=<ms> · --HeldOutDocs=<n per kind> · --TrainCorpusMb=<MB> · --WriteMeta

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { LoadRunnableModelFrom } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { LoadCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import {
  RunSuite, FormatReport, CompareReports, ParseSuiteReport, AttachEvalReport, ReadEvalReport, WriteEvalReportToFile,
  LoadHeldOutFromFile, LoadHeldOutFromSeed, LoadHeldOutFromStore, LoadTrainCorpusFromFile, LoadTrainCorpusFromStore,
  EvalCaseSetV1, SuiteDefaults,
} from "../Brain/Eval/Suite/SuiteBarrel.ts";
import type { HeldOutSet, SuiteDocumentSource, SuiteReport, TrainCorpus } from "../Brain/Eval/Suite/SuiteBarrel.ts";
import { DataKinds } from "../Foundry/DataKinds.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag, ReadFloatArg, ReadIntArg } from "./ScriptArgs.ts";

const ReportDir = join("Runs", "Eval");
const CheckpointExtension = ".ckpt";
const BytesPerMegabyte = 1_000_000;

const CheckpointArg = ReadArg("--Checkpoint=", "");
if (CheckpointArg === "") {
  console.error("usage: bun run eval --Checkpoint=<path|name> [--Baseline=<report.json>] [--HeldOut=<file>] [--TrainCorpus=<file>] [--MaxCases=<n>] [--WriteMeta]");
  process.exit(1);
}
const BaselineArg = ReadArg("--Baseline=", "");
const HeldOutArg = ReadArg("--HeldOut=", "");
const TrainCorpusArg = ReadArg("--TrainCorpus=", "");
const MaxCases = ReadIntArg("--MaxCases", SuiteDefaults.MaxCases, { Min: 0 });
const CaseTimeoutMs = ReadIntArg("--CaseTimeoutMs", SuiteDefaults.CaseTimeoutMs, { Min: 1 });
const HeldOutDocsPerKind = ReadIntArg("--HeldOutDocs", SuiteDefaults.HeldOutDocsPerKind, { Min: 1 });
const TrainCorpusMaxBytes = Math.round(ReadFloatArg("--TrainCorpusMb", SuiteDefaults.TrainCorpusMaxBytes / BytesPerMegabyte, { Min: 0.001 }) * BytesPerMegabyte);
const WriteMeta = ReadFlag("--WriteMeta");
const DbUrl = process.env["DATABASE_URL"];
const HasDb = DbUrl !== undefined && DbUrl !== "";

// 1) The checkpoint: a file on disk wins; otherwise a Postgres checkpoint name.
type Loaded = { Ckpt: Checkpoint; Label: string; File: string | null };
async function LoadTarget(): Promise<Loaded> {
  if (existsSync(CheckpointArg)) return { Ckpt: LoadCheckpoint(CheckpointArg), Label: basename(CheckpointArg, extname(CheckpointArg)), File: CheckpointArg };
  if (!HasDb) throw new Error(`--Checkpoint=${CheckpointArg}: no such file, and no DATABASE_URL to load it by name`);
  const Store = new PostgresCheckpointStore(DbUrl);
  try {
    const Ckpt = await Store.Load(CheckpointArg);
    if (Ckpt === null) throw new Error(`--Checkpoint=${CheckpointArg}: no such file and no Postgres checkpoint by that name`);
    return { Ckpt, Label: CheckpointArg, File: null };
  } finally {
    await Store.Close();
  }
}

// 2) The held-out set and the training text (store reads need DATABASE_URL; one shared pool).
async function LoadCorpora(): Promise<{ HeldOut: HeldOutSet; Train: TrainCorpus | null }> {
  const NeedStore = HasDb && (HeldOutArg === "" || TrainCorpusArg === "");
  const Stores = NeedStore ? ResolveFoundryStores() : null;
  try {
    const Sources = new Map<string, SuiteDocumentSource>();
    if (Stores !== null) for (const Kind of DataKinds) Sources.set(Kind, Stores.Kind(Kind));
    let HeldOut: HeldOutSet;
    if (HeldOutArg !== "") HeldOut = LoadHeldOutFromFile(HeldOutArg);
    else if (Stores !== null) HeldOut = await LoadHeldOutFromStore(Sources, HeldOutDocsPerKind, SuiteDefaults.HeldOutMaxScan);
    else HeldOut = LoadHeldOutFromSeed();
    if (HeldOut.Docs.length === 0) {
      console.warn(`[eval] held-out source is empty (${HeldOut.Note}) — falling back to the seed corpus`);
      HeldOut = LoadHeldOutFromSeed();
    }
    let Train: TrainCorpus | null = null;
    if (TrainCorpusArg !== "") Train = LoadTrainCorpusFromFile(TrainCorpusArg, TrainCorpusMaxBytes);
    else if (Stores !== null) Train = await LoadTrainCorpusFromStore(Sources, TrainCorpusMaxBytes);
    return { HeldOut, Train };
  } finally {
    await Stores?.Close();
  }
}

function LoadBaseline(Path: string): SuiteReport {
  if (extname(Path) === CheckpointExtension) {
    const Stored = ReadEvalReport(LoadCheckpoint(Path));
    if (Stored === null) throw new Error(`--Baseline=${Path}: checkpoint has no Meta.Eval report (run eval on it with --WriteMeta first)`);
    return Stored;
  }
  return ParseSuiteReport(JSON.parse(readFileSync(Path, "utf8")));
}

// Postgres-backed checkpoints are re-saved through the store (its row records the payload hash, so the
// payload file must never be rewritten behind its back); CreatedAt is preserved so listings keep order.
async function WriteMetaToStore(Name: string, Ckpt: Checkpoint, Report: SuiteReport): Promise<void> {
  const Store = new PostgresCheckpointStore(DbUrl ?? "");
  try {
    const CreatedAt = (await Store.List()).find((S) => S.Name === Name)?.CreatedAt ?? new Date().toISOString();
    await Store.Save(Name, AttachEvalReport(Ckpt, Report), CreatedAt);
    const Back = await Store.Load(Name);
    if (Back === null || ReadEvalReport(Back)?.RanAt !== Report.RanAt) throw new Error(`Meta.Eval did not round-trip for "${Name}"`);
  } finally {
    await Store.Close();
  }
}

const Target = await LoadTarget();
const Runnable = LoadRunnableModelFrom(Target.Ckpt);
const Corpora = await LoadCorpora();
const Baseline = BaselineArg !== "" ? LoadBaseline(BaselineArg) : null; // fail before the long run, not after
console.log(`[eval] ${Target.Label}: ${Runnable.Chat ? "chat" : "base"} model, suite ${EvalCaseSetV1.Version}, held-out ${Corpora.HeldOut.Origin}, train text ${Corpora.Train === null ? "none (memorization skipped)" : Corpora.Train.Origin}`);

const Report = await RunSuite(Runnable, EvalCaseSetV1, {
  Checkpoint: Target.Label,
  Settings: { MaxCases, CaseTimeoutMs, HeldOutDocsPerKind, TrainCorpusMaxBytes },
  HeldOut: Corpora.HeldOut,
  TrainCorpus: Corpora.Train,
  Log: (Line) => console.log(Line),
});

console.log(`\n${FormatReport(Report)}`);
if (Baseline !== null) console.log(`\ncomparison against ${BaselineArg}:\n${CompareReports(Baseline, Report).Table}`);

mkdirSync(ReportDir, { recursive: true });
const OutPath = join(ReportDir, `${Target.Label}-${Report.RanAt.replace(/[:.]/g, "-")}.json`);
writeFileSync(OutPath, JSON.stringify(Report, null, 2));
console.log(`\nreport written to ${OutPath}`);

if (WriteMeta) {
  if (Target.File !== null) WriteEvalReportToFile(Target.File, Report);
  else await WriteMetaToStore(Target.Label, Target.Ckpt, Report);
  console.log(`Meta.Eval stored in ${Target.File ?? `postgres:${Target.Label}`} (verified by loading it back)`);
}
process.exit(0);
