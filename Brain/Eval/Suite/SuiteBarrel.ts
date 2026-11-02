// The evaluation suite's public surface (Plan: Brain-Strengthening, phase 1 — "measure before training").

export type { EvalSlice, MetricResult, CodeCase, FormatCase, ToolCase, MultiTurnCase, EvalCaseSet, SuiteReport } from "./SuiteTypes.ts";
export { EvalSlices } from "./SuiteTypes.ts";
export { RunSuite } from "./RunSuite.ts";
export type { SuiteRunOptions } from "./RunSuite.ts";
export { SuiteDefaults, ResolveSuiteSettings } from "./SuiteSettings.ts";
export type { SuiteSettings } from "./SuiteSettings.ts";
export { CompareReports, FormatReport } from "./SuiteReportTable.ts";
export type { Comparison, ComparisonRow, Direction } from "./SuiteReportTable.ts";
export { ParseSuiteReport, AttachEvalReport, ReadEvalReport, WriteEvalReportToFile, EvalMetaKey } from "./SuiteReportIo.ts";
export { SuiteEvalDocs, TrainingDecontaminationDocs } from "./SuiteEvalDocs.ts";
export {
  LoadHeldOutFromStore, LoadHeldOutFromFile, LoadHeldOutFromSeed, LoadTrainCorpusFromFile, LoadTrainCorpusFromStore,
} from "./SuiteCorpora.ts";
export type { SuiteDocumentSource, HeldOutSet, HeldOutDoc, TrainCorpus } from "./SuiteCorpora.ts";
export type { GeneratorFactory, GenerationSettings } from "./SuiteGeneration.ts";
export { EvalCaseSetV1 } from "./Data/CaseSetV1.ts";
