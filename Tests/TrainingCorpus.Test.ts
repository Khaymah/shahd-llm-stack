import { test, expect } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryDocumentStore } from "../Foundry/InMemoryDocumentStore.ts";
import type { DocumentRecord, Tier } from "../Foundry/DocumentRecord.ts";
import { ParseDataSpec } from "../Foundry/DataSpec.ts";
import type { DataSpec } from "../Foundry/DataSpec.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { IsValidationDocument, PartitionByDocumentSplit, TrainingSideReadSize, AssignValidationSides, ValidationPercent } from "../Brain/Data/DocumentSplit.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import {
  AssembleSpecMix,
  ChatSpecials,
  CheckEpochs,
  ChooseTokenizerFile,
  DecideResume,
  DefaultTokenizerPath,
  PickMerges,
  PretrainSpecials,
  ResolveSpec,
  SliceBudgets,
  SpecTotalMb,
  StepsForEpochs,
} from "../Scripts/TrainingCorpus.ts";
import type { CorpusSource } from "../Scripts/TrainingCorpus.ts";

const Isolated = { UseCli: false, UseEnv: false } as const;

function Spec(Slices: DataSpec["Slices"]): DataSpec {
  return ParseDataSpec({
    Version: "test-1",
    Domain: "web development assistant",
    Slices,
    Repos: [{ Repo: "owner/name" }],
    Exclusions: { PathGlobs: [], MinFileBytes: 10, MaxFileBytes: 100_000 },
    Training: { MaxEpochs: 4, ContextTokens: 1024, VocabSize: 16_384 },
  });
}

let Serial = 0;
function Record(Lang: string, Bytes: number, Tier: Tier = "Filtered"): DocumentRecord {
  const Content = `${Lang} doc ${Serial++} `.padEnd(Bytes, "x");
  const Id = createHash("sha256").update(Content).digest("hex");
  return { Id, Tier, Origin: "owned", Source: "src", License: "MIT", Lang, Content, Bytes: Buffer.byteLength(Content), QualityScore: 1, ContentHash: Id, Embedding: [], RejectReason: null, Provenance: `p/${Id}`, IngestedAt: "2026-11-02T00:00:00.000Z" };
}

async function Fill(Store: InMemoryDocumentStore, Lang: string, Count: number, Bytes: number, Tier: Tier = "Filtered"): Promise<DocumentRecord[]> {
  const Out: DocumentRecord[] = [];
  for (let I = 0; I < Count; I++) {
    const Doc = Record(Lang, Bytes, Tier);
    await Store.Upsert(Doc);
    Out.push(Doc);
  }
  return Out;
}

// ---- spec budgets -----------------------------------------------------------------------------------

test("SliceBudgets splits the total by Weight (default TargetMb), sorted by slice", () => {
  const S = Spec({ ts: { TargetMb: 300, Langs: ["typescript"] }, docs: { TargetMb: 100, Langs: ["text-en"], Weight: 100 }, go: { TargetMb: 50, Langs: ["go"], Weight: 100 } });
  const Budgets = SliceBudgets(S, 50);
  expect(Budgets.map((B) => B.Slice)).toEqual(["docs", "go", "ts"]);
  expect(Budgets.map((B) => B.TargetMb)).toEqual([10, 10, 30]); // weights 100 : 100 : 300
  expect(Budgets.reduce((A, B) => A + B.Share, 0)).toBeCloseTo(1, 12);
  expect(SpecTotalMb(S)).toBe(450);
});

test("ResolveSpec: none disables, a missing named file is an error, the default only when it exists", () => {
  expect(ResolveSpec("none", true)).toBeNull();
  expect(ResolveSpec("", true, () => false)).toBeNull();
  expect(ResolveSpec("", false, () => true)).toBeNull();
  expect(() => ResolveSpec("Missing/Spec.json", false, () => false)).toThrow(/not found/);
});

// ---- per-slice assembly over the document store -----------------------------------------------------

test("AssembleSpecMix samples each slice by its Langs, in proportion, deterministically", async () => {
  const Code = new InMemoryDocumentStore();
  const Knowledge = new InMemoryDocumentStore();
  await Fill(Code, "typescript", 700, 1000); // > one 500-row page: exercises keyset paging
  await Fill(Code, "javascript", 300, 1000);
  await Fill(Code, "go", 50, 1000);
  await Fill(Code, "python", 100, 1000); // no slice wants python
  await Fill(Code, "typescript", 50, 1000, "Raw"); // never trained
  await Fill(Knowledge, "text-en", 200, 1000);
  const Sources: CorpusSource[] = [{ Name: "code", Store: Code }, { Name: "knowledge", Store: Knowledge }];
  // 0.6 MB total: ts+js 0.4, go 0.1, docs 0.1 (weights 4 : 1 : 1).
  const S = Spec({ ts: { TargetMb: 4, Langs: ["typescript", "javascript"] }, go: { TargetMb: 1, Langs: ["go"] }, docs: { TargetMb: 1, Langs: ["text-en"] } });
  const Mix = await AssembleSpecMix(S, Sources, 0.6, false);
  expect(Mix.PerSlice["ts"]?.Docs).toBe(400);
  expect(Mix.PerSlice["go"]?.Docs).toBe(50); // short: only 50 KB of go exists
  expect(Mix.PerSlice["go"]?.Mb).toBeLessThan(Mix.PerSlice["go"]?.TargetMb ?? 0);
  expect(Mix.PerSlice["docs"]?.Docs).toBe(100);
  const Ts = Mix.Docs.filter((D) => D.Slice === "ts");
  // Both langs of the slice appear, roughly in proportion to their share (700 : 300).
  const JsShare = Ts.filter((D) => D.Content.startsWith("javascript")).length / Ts.length;
  expect(JsShare).toBeGreaterThan(0.15);
  expect(JsShare).toBeLessThan(0.45);
  // Within a slice and source the sample is the lowest ids (ascending id = deterministic hash sample).
  const TsIds = Ts.map((D) => D.Id);
  expect(TsIds).toEqual([...TsIds].sort());
  expect(Mix.Docs.some((D) => D.Content.startsWith("python"))).toBe(false);
  const Again = await AssembleSpecMix(S, Sources, 0.6, false);
  expect(Again.Docs.map((D) => D.Id)).toEqual(Mix.Docs.map((D) => D.Id));
  expect(new Set(Mix.Docs.map((D) => D.Id)).size).toBe(Mix.Docs.length);
});

test("AssembleSpecMix can exclude the held-out side (the tokenizer sample)", async () => {
  const Code = new InMemoryDocumentStore();
  await Fill(Code, "typescript", 400, 500);
  const S = Spec({ ts: { TargetMb: 1, Langs: ["typescript"] } });
  const All = await AssembleSpecMix(S, [{ Name: "code", Store: Code }], 1, false);
  const TrainOnly = await AssembleSpecMix(S, [{ Name: "code", Store: Code }], 1, true);
  expect(All.Docs.some((D) => IsValidationDocument(D.Id))).toBe(true);
  expect(TrainOnly.Docs.some((D) => IsValidationDocument(D.Id))).toBe(false);
  expect(TrainOnly.Docs.length).toBe(All.Docs.length - All.Docs.filter((D) => IsValidationDocument(D.Id)).length);
});

// ---- epoch guard --------------------------------------------------------------------------------------

test("CheckEpochs: epochs = Steps x Batch x Context / UniqueTokens, refused above MaxEpochs with the max Steps", () => {
  const Over = CheckEpochs({ UniqueTokens: 1_000_000, Steps: 1000, Batch: 16, Context: 512, MaxEpochs: 4, AllowEpochs: null });
  expect(Over.TokensSeen).toBe(8_192_000);
  expect(Over.Epochs).toBeCloseTo(8.192, 9);
  expect(Over.Ok).toBe(false);
  expect(Over.MaxSteps).toBe(488); // floor(4 * 1e6 / 8192)
  expect(Over.Message).toContain("--Steps=488");
  const Fits = CheckEpochs({ UniqueTokens: 1_000_000, Steps: 488, Batch: 16, Context: 512, MaxEpochs: 4, AllowEpochs: null });
  expect(Fits.Ok).toBe(true);
  const Allowed = CheckEpochs({ UniqueTokens: 1_000_000, Steps: 1000, Batch: 16, Context: 512, MaxEpochs: 4, AllowEpochs: 10 });
  expect(Allowed.Ok).toBe(true);
  expect(Allowed.Limit).toBe(10);
  const NoSpec = CheckEpochs({ UniqueTokens: 1000, Steps: 1000, Batch: 16, Context: 512, MaxEpochs: null, AllowEpochs: null });
  expect(NoSpec.Ok).toBe(true); // report only
  expect(CheckEpochs({ UniqueTokens: 0, Steps: 1, Batch: 1, Context: 1, MaxEpochs: 4, AllowEpochs: null }).Ok).toBe(false);
});

// ---- tokenizer choice -----------------------------------------------------------------------------------

test("ChooseTokenizerFile: explicit path, the default when present, or fresh", () => {
  expect(ChooseTokenizerFile("", () => true)).toEqual({ Path: DefaultTokenizerPath, Explicit: false });
  expect(ChooseTokenizerFile("", () => false)).toBeNull();
  expect(ChooseTokenizerFile("fresh", () => true)).toBeNull();
  expect(ChooseTokenizerFile("Data/Tokenizer.v2.json", () => true)).toEqual({ Path: "Data/Tokenizer.v2.json", Explicit: true });
  expect(() => ChooseTokenizerFile("Data/Nope.json", () => false)).toThrow(/not found/);
});

test("PickMerges: a resume keeps the checkpoint's merges and refuses an explicit, disagreeing --Tokenizer", () => {
  const Dir = mkdtempSync(join(tmpdir(), "shahd-tok-"));
  try {
    const FileMerges: [number, number][] = [[97, 98], [256, 99]];
    const Path = join(Dir, "Tokenizer.v1.json");
    writeFileSync(Path, JSON.stringify({ Kind: "Bpe", Version: "v1", Merges: FileMerges, TrainedOn: { SampleMb: 1, PerSlice: {} }, CreatedAt: "2026-11-02T00:00:00.000Z" }));
    const Ckpt = { Merges: [[120, 121]] as [number, number][], From: "checkpoint" as const, Name: "Base" };
    const Never = (): [number, number][] => {
      throw new Error("must not train");
    };
    expect(() => PickMerges(Ckpt, { Path, Explicit: true }, Never)).toThrow(/disagrees with the checkpoint "Base"/);
    const Implicit = PickMerges(Ckpt, { Path, Explicit: false }, Never);
    expect(Implicit.Merges).toEqual(Ckpt.Merges);
    expect(Implicit.Origin).toContain("ignored");
    expect(PickMerges({ ...Ckpt, Merges: FileMerges }, { Path, Explicit: true }, Never).Merges).toEqual(FileMerges);
    expect(PickMerges(null, { Path, Explicit: false }, Never).Merges).toEqual(FileMerges);
    expect(PickMerges(null, null, () => [[1, 2]]).Merges).toEqual([[1, 2]]);
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
});

// ---- resume / overwrite guard ----------------------------------------------------------------------------

function FakeCheckpoint(MaxSteps: number, Step: number, Specials: string[] | null): Checkpoint {
  const Config = LoadConfig({ Overrides: { Schedule: { MaxSteps, WarmupSteps: 1 } }, ...Isolated });
  return { Config, Meta: { Step }, TokenizerState: { Kind: "Bpe", Merges: [[1, 2]], ...(Specials !== null ? { Specials } : {}) } } as unknown as Checkpoint;
}

test("DecideResume: crash recovery, explicit extension, and the overwrite guard", () => {
  const Base = { Name: "Run", RequireSpecials: false, What: "model", ArchMatches: () => true };
  expect(DecideResume({ ...Base, Existing: null, Steps: 100, ResumeFlag: false })).toEqual({ Resume: null, Refusal: null });
  // Same MaxSteps, not finished -> auto-resume at the saved step.
  expect(DecideResume({ ...Base, Existing: FakeCheckpoint(100, 40, null), Steps: 100, ResumeFlag: false }).Resume?.Step).toBe(40);
  // Finished model, same Steps -> refused (would overwrite), names the escape hatches.
  expect(DecideResume({ ...Base, Existing: FakeCheckpoint(100, 100, null), Steps: 100, ResumeFlag: false }).Refusal).toContain("--Fresh");
  // --Resume extends a finished model to more steps.
  expect(DecideResume({ ...Base, Existing: FakeCheckpoint(100, 100, null), Steps: 200, ResumeFlag: true }).Resume?.Step).toBe(100);
  // Architecture mismatch under --Resume -> a "cannot resume" refusal.
  expect(DecideResume({ ...Base, ArchMatches: () => false, Existing: FakeCheckpoint(100, 10, null), Steps: 200, ResumeFlag: true }).Refusal).toContain("cannot resume");
  // A chat run needs the specials in the saved state.
  expect(DecideResume({ ...Base, RequireSpecials: true, Existing: FakeCheckpoint(100, 10, null), Steps: 100, ResumeFlag: false }).Resume).toBeNull();
  expect(DecideResume({ ...Base, RequireSpecials: true, Existing: FakeCheckpoint(100, 10, ["<|user|>"]), Steps: 100, ResumeFlag: false }).Resume?.State.Specials).toEqual(["<|user|>"]);
});

// ---- the document split used by SFT -------------------------------------------------------------------

test("PartitionByDocumentSplit keeps the held-out side out of SFT; read size compensates for it", () => {
  const Docs = Array.from({ length: 500 }, (_V, I) => ({ Id: createHash("sha256").update(`doc-${I}`).digest("hex") }));
  const Split = PartitionByDocumentSplit(Docs);
  expect(Split.Train.length + Split.Validation.length).toBe(500);
  expect(Split.Validation.every((D) => IsValidationDocument(D.Id))).toBe(true);
  expect(Split.Train.some((D) => IsValidationDocument(D.Id))).toBe(false);
  expect(Split.Validation.length).toBeGreaterThan(20); // ~10%
  expect(TrainingSideReadSize(900)).toBe(Math.ceil(900 / (1 - ValidationPercent / 100)));
  expect(TrainingSideReadSize(0)).toBe(0);
});

test("AssignValidationSides follows the bucket rule and forces both sides non-empty", () => {
  const Ids = Array.from({ length: 200 }, (_V, I) => `id-${I}`);
  expect(AssignValidationSides(Ids)).toEqual(Ids.map(IsValidationDocument));
  const TrainOnly = Ids.filter((Id) => !IsValidationDocument(Id)).slice(0, 3);
  const Sides = AssignValidationSides(TrainOnly);
  expect(Sides.filter(Boolean).length).toBe(1);
  expect(() => AssignValidationSides(["only"])).toThrow(/>= 2/);
});

test("the special-token layouts: pretraining reserves EOS + FIM + every chat/tool token, in a fixed order", () => {
  expect(PretrainSpecials[0]).toBe(SpecialTokens.Eos);
  expect(PretrainSpecials.slice(-ChatSpecials.length)).toEqual([...ChatSpecials]);
  expect(new Set(PretrainSpecials).size).toBe(PretrainSpecials.length);
});

test("StepsForEpochs is the inverse of the epoch guard", () => {
  expect(StepsForEpochs(2, 1_100_805, 16, 256)).toBe(537);
  expect(StepsForEpochs(4, 1_100_805, 16, 256)).toBe(1075); // the step count the guard suggested for 4 epochs
  expect(CheckEpochs({ UniqueTokens: 1_100_805, Steps: StepsForEpochs(4, 1_100_805, 16, 256), Batch: 16, Context: 256, MaxEpochs: 4, AllowEpochs: null }).Ok).toBe(true);
  expect(StepsForEpochs(0.001, 100, 16, 256)).toBe(1);
  expect(() => StepsForEpochs(0, 100, 16, 256)).toThrow();
});
