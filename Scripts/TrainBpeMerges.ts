// Offline BPE merge training (kept out of the training loop, per the spec's train/serve split). Learns
// N merges and writes a PERSISTED tokenizer file that every training run reuses:
//   Data/Tokenizer.<Version>.json = { Kind: "Bpe", Version, Merges, TrainedOn: { SampleMb, PerSlice, Source }, CreatedAt }
//
// Two sources:
//   --Spec=Data/Spec.json (the default when that file exists): sample --SampleMb (default 30) from the
//     Filtered tier, split across the spec's slices by weight (each slice's Langs -> documents; held-out
//     documents excluded), deterministic order. --Merges defaults to Spec.Training.VocabSize - 256 -
//     the reserved specials, so the model's vocab lands exactly on the spec's VocabSize.
//   --Corpus=<file>: a UTF-8 text file, as before.
//
//   bun run train:bpe                                         # spec sample -> Data/Tokenizer.v1.json
//   bun run train:bpe --SampleMb=50 --Version=v2
//   bun run train:bpe --Corpus=Corpus/Code.txt --Merges=4000 --Out=Data/Tokenizer.code.json
//
// An existing output file is never replaced without --Force: runs that default to it would silently
// switch vocabularies (resumed checkpoints keep their own merges either way).

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { ByteVocab, TokenizerFilePath } from "../Brain/Tokenizer/BytePairEncoder.ts";
import type { BpeModelFile, SliceSample } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { ReadArg, ReadFlag, ReadFloatArg, ReadIntArg } from "./ScriptArgs.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { AssembleSpecMix, DescribeMix, PretrainSources, PretrainSpecials, ResolveSpec } from "./TrainingCorpus.ts";

const DefaultSampleMb = 30;
const DefaultCorpusMerges = 1000;

const CorpusPath = ReadArg("--Corpus=", "");
// --Spec > FOUNDRY_SPEC (same env as the ingest scripts) > Data/Spec.json when it exists.
const SpecChoice = CorpusPath === "" ? ResolveSpec(ReadArg("--Spec=", process.env["FOUNDRY_SPEC"] ?? ""), true) : null;
const SampleMb = ReadFloatArg("--SampleMb", DefaultSampleMb, { Min: 0.001 });
const Version = ReadArg("--Version=", "v1");
const OutPath = ReadArg("--Out=", TokenizerFilePath(Version));
const Force = ReadFlag("--Force");

if (CorpusPath === "" && SpecChoice === null) {
  console.error("TrainBpeMerges: provide --Corpus=<UTF-8 text file> or --Spec=<Data/Spec.json> (no Data/Spec.json found)");
  process.exit(1);
}
if (CorpusPath !== "" && !existsSync(CorpusPath)) {
  console.error(`TrainBpeMerges: --Corpus=${CorpusPath} not found`);
  process.exit(1);
}
if (existsSync(OutPath) && !Force) {
  console.error(`TrainBpeMerges: ${OutPath} already exists — pick a new --Version/--Out, or pass --Force to replace it`);
  process.exit(1);
}

// Merges that make 256 bytes + merges + reserved specials equal the spec's VocabSize.
const SpecMerges = SpecChoice !== null ? SpecChoice.Spec.Training.VocabSize - ByteVocab - PretrainSpecials.length : DefaultCorpusMerges;
if (SpecMerges <= 0) {
  console.error(`TrainBpeMerges: Spec.Training.VocabSize ${SpecChoice?.Spec.Training.VocabSize} leaves no room for merges (256 bytes + ${PretrainSpecials.length} specials)`);
  process.exit(1);
}
const NumMerges = ReadIntArg("--Merges", SpecMerges, { Min: 1 });

let Corpus: string | string[];
let TrainedOn: BpeModelFile["TrainedOn"];
if (SpecChoice !== null) {
  const Stores = ResolveFoundryStores();
  try {
    const Mix = await AssembleSpecMix(SpecChoice.Spec, PretrainSources(Stores), SampleMb, true);
    for (const Line of DescribeMix(Mix)) console.log(`sample: ${Line}`);
    Corpus = Mix.Docs.map((D) => D.Content);
    const PerSlice: Record<string, SliceSample> = {};
    for (const [Slice, Tally] of Object.entries(Mix.PerSlice)) PerSlice[Slice] = { Mb: Tally.Mb, Docs: Tally.Docs, TargetMb: Tally.TargetMb };
    TrainedOn = { SampleMb: Mix.Mb, PerSlice, Source: `spec:${SpecChoice.Path}` };
  } finally {
    await Stores.Close();
  }
  if (Corpus.length === 0) {
    console.error("TrainBpeMerges: the spec sample is empty — no Filtered documents match the spec's slice Langs");
    process.exit(1);
  }
} else {
  Corpus = readFileSync(CorpusPath, "utf8");
  const Mb = Number((Buffer.byteLength(Corpus, "utf8") / 1e6).toFixed(3));
  TrainedOn = { SampleMb: Mb, PerSlice: {}, Source: `corpus:${CorpusPath}` };
}

const Start = Date.now();
const Model = TrainBpe(Corpus, NumMerges);
const Seconds = (Date.now() - Start) / 1000;
const File: BpeModelFile = { Kind: "Bpe", Version, Merges: Model.Merges, TrainedOn, CreatedAt: new Date().toISOString() };
const Dir = dirname(OutPath);
if (Dir !== "" && !existsSync(Dir)) mkdirSync(Dir, { recursive: true });
writeFileSync(OutPath, JSON.stringify(File));
console.log(
  `TrainBpeMerges: ${Model.Merges.length} merges over ${TrainedOn.SampleMb}MB in ${Seconds.toFixed(1)}s -> ${OutPath} ` +
    `(base vocab ${ByteVocab + Model.Merges.length}; + ${PretrainSpecials.length} pretraining specials = ${ByteVocab + Model.Merges.length + PretrainSpecials.length})`,
);
if (Model.Merges.length < NumMerges) console.warn(`TrainBpeMerges: the sample ran out of pairs after ${Model.Merges.length}/${NumMerges} merges — sample more data`);
