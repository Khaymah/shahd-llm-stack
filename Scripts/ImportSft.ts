// Import synthetic SFT JSONL (Scripts/SynthesizeSft.ts output, or the merged SyntheticSft artifact) into
// documents_conversation, idempotently — re-importing a file stores nothing new. Every record is
// re-validated (schema, structure, tool replay against Tests/Fixtures/SftWorkspace) before ingest; the
// tier is decided by Foundry/Tiering.ts (Origin "synthetic"). Needs DATABASE_URL.
//
//   bun run sft:import Runs/SyntheticSft/SyntheticSft.jsonl
//   bun run sft:import --In=Runs/Sft/shard-0.jsonl,Runs/Sft/shard-1.jsonl
//   bun run sft:import SyntheticSft.jsonl --Generators=tools,multiturn,textbook   # only these generators

import { existsSync, readFileSync } from "node:fs";
import { ParseSynthRecords } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import type { SynthRecord } from "../Brain/Sft/Synthetic/SynthRecord.ts";
import { SynthToolReplay } from "../Brain/Sft/Synthetic/SynthTools.ts";
import { ImportSyntheticRecords } from "../Foundry/SyntheticImport.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

// How many malformed lines are printed per file before the rest are only counted.
const ShownInvalidLines = 5;

const Files = [
  ...process.argv.slice(2).filter((Arg) => !Arg.startsWith("--")),
  ...ReadArg("--In=", "").split(",").map((Path) => Path.trim()).filter((Path) => Path !== ""),
];
if (Files.length === 0) {
  console.error("usage: bun run sft:import <file.jsonl> [more.jsonl …]   (or --In=a.jsonl,b.jsonl)");
  process.exit(1);
}

// --Generators=a,b: import only records from these generators (the rest are counted and skipped).
// Used to keep a generator whose output failed a quality review out of a training run without
// regenerating the dataset (e.g. grounded answers that did not match the question's task).
const OnlyGenerators = ReadArg("--Generators=", "").split(",").map((Name) => Name.trim()).filter((Name) => Name !== "");
const SkippedByGenerator = new Map<string, number>();

const Records: SynthRecord[] = [];
for (const File of Files) {
  if (!existsSync(File)) {
    console.error(`sft:import: ${File} not found`);
    process.exit(1);
  }
  const Parsed = ParseSynthRecords(readFileSync(File, "utf8"));
  for (const Record of Parsed.Records) {
    if (OnlyGenerators.length === 0 || OnlyGenerators.includes(Record.Generator)) Records.push(Record);
    else SkippedByGenerator.set(Record.Generator, (SkippedByGenerator.get(Record.Generator) ?? 0) + 1);
  }
  console.log(`${File}: ${Parsed.Records.length} records${Parsed.Invalid.length > 0 ? `, ${Parsed.Invalid.length} malformed lines skipped` : ""}`);
  for (const Bad of Parsed.Invalid.slice(0, ShownInvalidLines)) console.warn(`  line ${Bad.Line}: ${Bad.Reason}`);
}

if (SkippedByGenerator.size > 0) {
  console.log(`sft:import: --Generators=${OnlyGenerators.join(",")} skipped ${[...SkippedByGenerator].map(([Name, N]) => `${Name} ${N}`).join(", ")}`);
}

const Stores = ResolveFoundryStores();
try {
  const Result = await ImportSyntheticRecords(Records, Stores.Kind("conversation"), new Date().toISOString(), SynthToolReplay());
  const Invalid = Object.entries(Result.Invalid).map(([Code, N]) => `${Code} ${N}`).join(", ");
  console.log(`sft:import: ${Result.Stats.New} new, ${Result.Stats.Duplicate} already stored, ${Result.Stats.Failed} failed; tiers Filtered ${Result.Stats.ByTier.Filtered} / Rejected ${Result.Stats.ByTier.Rejected}${Invalid === "" ? "" : `; invalid (not imported): ${Invalid}`}`);
  if (Result.Stats.Failed > 0) process.exitCode = 1;
} finally {
  await Stores.Close();
}
