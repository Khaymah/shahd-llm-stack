// Register a checkpoint FILE (e.g. one downloaded from a train-model artifact) in the Postgres
// checkpoint store under a name, so name-based tools can use it: `TrainSftChat --From=<Name>`,
// `bun run eval --Checkpoint=<Name>`, the dashboard's model picker. The file is parsed (checksum
// verified) before anything is written, and an existing name is refused unless --Force.
//
//   bun run Scripts/RegisterCheckpoint.ts --File=Runs/Base/MicroChat2Base.ckpt --Name=MicroChat2Base

import { existsSync } from "node:fs";
import { LoadCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ReadArg, ReadFlag, ReadNameArg } from "./ScriptArgs.ts";

const File = ReadArg("--File=", "");
const Name = ReadNameArg("--Name", "");
const Force = ReadFlag("--Force");
const DbUrl = process.env["DATABASE_URL"];

if (File === "" || !existsSync(File)) {
  console.error(`RegisterCheckpoint: --File=<path to a .ckpt> is required and must exist (got "${File}")`);
  process.exit(1);
}
if (Name === "") {
  console.error("RegisterCheckpoint: --Name=<checkpoint name> is required");
  process.exit(1);
}
if (DbUrl === undefined || DbUrl === "") {
  console.error("RegisterCheckpoint: DATABASE_URL is not set");
  process.exit(1);
}

const Ckpt = LoadCheckpoint(File); // throws on a corrupt / truncated payload (checksum)
const Store = new PostgresCheckpointStore(DbUrl);
try {
  if (!Force && (await Store.Load(Name)) !== null) {
    console.error(`RegisterCheckpoint: a checkpoint named "${Name}" already exists — pass --Force to replace it`);
    process.exitCode = 1;
  } else {
    await Store.Save(Name, Ckpt, new Date().toISOString());
    const M = Ckpt.Config.Model;
    console.log(`registered "${Name}" from ${File} (emb ${M.EmbedDim} · ${M.NumLayers}L · ${M.NumHeads}h · ctx ${M.BlockSize} · vocab ${M.VocabSize}; format ${String(Ckpt.Meta["Format"] ?? "base")}; step ${String(Ckpt.Optimizer.StepCount)})`);
  }
} finally {
  await Store.Close();
}
