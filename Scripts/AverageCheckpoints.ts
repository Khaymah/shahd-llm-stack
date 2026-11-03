// Average the worker checkpoints of one parallel training round into the next round's start
// (Brain/Checkpoint/AverageCheckpoints.ts). Every input is parsed with its checksum verified first.
//
//   bun run Scripts/AverageCheckpoints.ts --Out=Runs/Global.ckpt Runs/Workers/*.ckpt [--Round=3]

import { LoadCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import { WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { CheckpointAverager } from "../Brain/Checkpoint/AverageCheckpoints.ts";
import { ReadArg, ReadIntArg } from "./ScriptArgs.ts";

const Out = ReadArg("--Out=", "");
const Round = ReadIntArg("--Round", 0, { Min: 0 });
const Files = process.argv.slice(2).filter((Arg) => !Arg.startsWith("--"));

if (Out === "" || Files.length === 0) {
  console.error("AverageCheckpoints: usage --Out=<file> <worker.ckpt>... [--Round=<n>]");
  process.exit(1);
}

const Averager = new CheckpointAverager();
for (const File of Files) {
  Averager.Add(LoadCheckpoint(File)); // one worker resident at a time
  console.log(`added ${File}`);
}
const { Ckpt, Report } = Averager.Finish(Round > 0 ? { Round } : {});
WriteCheckpointObject(Out, Ckpt);
console.log(`averaged ${Report.Workers} workers (steps ${Report.WorkerSteps.join(", ")}) -> step ${Report.Step} -> ${Out}`);
