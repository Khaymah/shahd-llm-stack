// Serve a trained checkpoint over the OpenAI-compatible HTTP API (Brain/Serving/InferenceServer.ts).
// Binds to loopback by default; exposing it is an explicit choice via the environment.
//
//   bun run Scripts/Serve.ts --Checkpoint=Checkpoints/Last.ckpt [--MaxInFlight=2]
//   SHAHD_SERVE_BIND=127.0.0.1 SHAHD_SERVE_PORT=8080 (defaults shown)

import { basename } from "node:path";
import { LoadRunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { StartServer, DefaultMaxInFlight } from "../Brain/Serving/InferenceServer.ts";
import { ReadArg, ReadIntArg } from "./ScriptArgs.ts";

const DefaultCheckpoint = "Checkpoints/Last.ckpt";

const CkptPath = ReadArg("--Checkpoint=", DefaultCheckpoint);
const MaxInFlight = ReadIntArg("--MaxInFlight", DefaultMaxInFlight, { Min: 1 });

const { Model, Tokenizer, Config, Chat } = LoadRunnableModel(CkptPath);
const Server = StartServer(Model, Tokenizer, Config, { MaxInFlight, ModelId: basename(CkptPath) });

console.log(
  `[serve] ${CkptPath} (${Chat ? "chat" : "base"} model) on http://${Server.hostname}:${Server.port} ` +
    `— POST /v1/chat/completions, GET /v1/models, GET /health (max ${MaxInFlight} concurrent generations)`,
);
