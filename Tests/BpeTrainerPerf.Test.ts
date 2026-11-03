// Performance floor for the incremental BPE trainer: 2,000 merges over a 2 MB synthetic code corpus
// must finish well inside 10 s on CPU (the reference recount-every-merge trainer needs minutes here).
// The corpus is deterministic (seeded) and code-shaped: identifiers drawn from a Zipf-like vocabulary,
// keywords, punctuation and indentation — so the pair statistics resemble real source.

import { test, expect } from "bun:test";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

const TargetBytes = 2_000_000;
const Merges = 2_000;
const BudgetMs = 10_000;
const VocabularySize = 4_000;
const Keywords = ["const", "let", "function", "return", "if", "else", "for", "while", "export", "import", "class", "new", "await", "async", "type", "interface"];
const Operators = [" = ", " + ", " - ", " * ", " === ", " !== ", " && ", " || ", ".", ", ", "(", ")", "[", "]", " => ", ": "];

function SyntheticIdentifier(Rng: SeededRng): string {
  const Letters = "abcdefghijklmnopqrstuvwxyz";
  const Parts = 1 + Math.floor(Rng.NextFloat() * 3);
  let Name = "";
  for (let P = 0; P < Parts; P++) {
    const Length = 2 + Math.floor(Rng.NextFloat() * 6);
    for (let I = 0; I < Length; I++) {
      const Letter = Letters[Math.floor(Rng.NextFloat() * Letters.length)];
      Name += I === 0 && P > 0 ? Letter.toUpperCase() : Letter;
    }
  }
  return Name;
}

function SyntheticCorpus(Seed: number): string[] {
  const Rng = new SeededRng(Seed);
  const Vocabulary: string[] = [];
  for (let I = 0; I < VocabularySize; I++) Vocabulary.push(SyntheticIdentifier(Rng));
  // Zipf-like: squaring a uniform draw favors the low ranks, like real identifier frequencies.
  const Pick = (): string => Vocabulary[Math.floor(Rng.NextFloat() ** 2 * Vocabulary.length)];
  const Documents: string[] = [];
  let Bytes = 0;
  while (Bytes < TargetBytes) {
    const Lines: string[] = [];
    const LineCount = 20 + Math.floor(Rng.NextFloat() * 60);
    for (let L = 0; L < LineCount; L++) {
      const Indent = "  ".repeat(Math.floor(Rng.NextFloat() * 4));
      let Line = `${Indent}${Keywords[Math.floor(Rng.NextFloat() * Keywords.length)]} ${Pick()}`;
      const Terms = 1 + Math.floor(Rng.NextFloat() * 5);
      for (let T = 0; T < Terms; T++) Line += `${Operators[Math.floor(Rng.NextFloat() * Operators.length)]}${Rng.NextFloat() < 0.15 ? String(Math.floor(Rng.NextFloat() * 1000)) : Pick()}`;
      Lines.push(`${Line};`);
    }
    const Document = Lines.join("\n");
    Documents.push(Document);
    Bytes += Document.length;
  }
  return Documents;
}

test(`TrainBpe learns ${Merges} merges over a 2 MB synthetic code corpus in under ${BudgetMs / 1000} s`, () => {
  const Corpus = SyntheticCorpus(42);
  const Start = performance.now();
  const Model = TrainBpe(Corpus, Merges);
  const ElapsedMs = performance.now() - Start;
  console.log(`BpeTrainerPerf: ${Model.Merges.length} merges over ${(Corpus.reduce((A, D) => A + D.length, 0) / 1e6).toFixed(2)} MB in ${(ElapsedMs / 1000).toFixed(2)} s`);
  expect(Model.Merges.length).toBe(Merges);
  expect(ElapsedMs).toBeLessThan(BudgetMs);
}, BudgetMs * 3);
