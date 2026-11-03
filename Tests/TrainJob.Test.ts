import { test, expect } from "bun:test";
import { CreateTrainRunner, BuildTrainArgs, PumpLines } from "../Foundry/TrainJob.ts";
import type { TrainProcess } from "../Foundry/TrainJob.ts";
import { ModelHolder, IncrementalDecoder } from "../Foundry/ModelHolder.ts";
import type { ModelHolderDeps } from "../Foundry/ModelHolder.ts";
import type { TrainEvent, TrainSettings } from "../Foundry/DashboardTypes.ts";
import type { RunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import type { ModelInfo } from "../Foundry/ModelInfo.ts";
import type { Tokenizer } from "../Brain/Tokenizer/TokenizerTypes.ts";

const Base: TrainSettings = {
  Kind: "pretrain", Name: "M1", Resume: false, Steps: 10, CorpusMb: 1, EmbedDim: 32, NumLayers: 1, NumHeads: 2, BlockSize: 16, Merges: 0, BatchSize: 2, KnowledgeMb: 0, ConvCount: 0, CodeSamples: 0,
};

const StreamOf = (Text: string): ReadableStream<Uint8Array> =>
  new ReadableStream({
    start(Controller) {
      if (Text !== "") Controller.enqueue(new TextEncoder().encode(Text));
      Controller.close();
    },
  });

// A fake trainer process: fixed stdout/stderr, an exit code, and a kill spy.
function FakeProcess(Stdout: string, Stderr: string, Exited: Promise<number>): TrainProcess & { Killed: boolean } {
  const Proc = { Stdout: StreamOf(Stdout), Stderr: StreamOf(Stderr), Exited, Killed: false, Kill: (): void => { Proc.Killed = true; } };
  return Proc;
}

const Silent = { Log: (): void => {}, LogError: (): void => {}, StopFilePath: (): string => "/tmp/never-written", RemoveStopFile: (): void => {} };

test("trainer: injected spawn gets the built args; progress/info parsed; exit 0 -> train-done", async () => {
  const Spawned: string[][] = [];
  const Train = CreateTrainRunner({
    ...Silent,
    Spawn: (Args) => {
      Spawned.push(Args);
      return FakeProcess('setup line\n{"Step":5,"TrainLoss":2.5,"ElapsedMs":100}\n{"Step":10,"TrainLoss":1.5,"ValLoss":1.7}', "", Promise.resolve(0));
    },
  });
  const Events: TrainEvent[] = [];
  await Train(Base, (E) => Events.push(E));
  expect(Spawned).toEqual([BuildTrainArgs(Base, "/tmp/never-written")]);
  expect(Spawned[0]!.slice(0, 3)).toEqual(["bun", "run", "Scripts/TrainOnFoundry.ts"]);
  expect(Events[0]).toEqual({ kind: "train-info", text: "setup line" });
  expect(Events.filter((E) => E.kind === "train-progress").map((E) => (E as { step: number }).step)).toEqual([5, 10]); // unterminated last line too
  expect(Events.at(-1)).toEqual({ kind: "train-done", savedTo: "postgres:M1" });
});

test("trainer: nonzero exit -> train-error, stderr surfaced as info", async () => {
  const Train = CreateTrainRunner({ ...Silent, Spawn: () => FakeProcess("", "boom: out of memory\n", Promise.resolve(3)) });
  const Events: TrainEvent[] = [];
  await Train(Base, (E) => Events.push(E));
  expect(Events).toContainEqual({ kind: "train-info", text: "[stderr] boom: out of memory" });
  expect(Events.at(-1)).toEqual({ kind: "train-error", message: "trainer exited with code 3 (see server console)" });
});

test("trainer: Stop writes the stop file; a graceful exit 0 reports a paused train-error", async () => {
  let Exit: (Code: number) => void = () => {};
  const Exited = new Promise<number>((Resolve) => {
    Exit = Resolve;
  });
  const Written: string[] = [];
  const Proc = FakeProcess("", "", Exited);
  const Train = CreateTrainRunner({ ...Silent, KillGraceMs: 60_000, WriteStopFile: (Path) => Written.push(Path), Spawn: () => Proc });
  const Events: TrainEvent[] = [];
  const Stop = new AbortController();
  const Running = Train(Base, (E) => Events.push(E), Stop.signal);
  Stop.abort();
  Exit(0);
  await Running;
  expect(Written).toEqual(["/tmp/never-written"]);
  expect(Proc.Killed).toBe(false); // exited inside the grace window — no hard kill
  expect(Events.at(-1)).toMatchObject({ kind: "train-error", paused: true });
});

test("BuildTrainArgs: chat SFT carries the warm-start base only when one was picked", () => {
  const Chat = BuildTrainArgs({ ...Base, Kind: "chat", From: "Base1", Resume: true }, "/s");
  expect(Chat[2]).toBe("Scripts/TrainSftChat.ts");
  expect(Chat).toContain("--From=Base1");
  expect(Chat).toContain("--Resume");
  expect(BuildTrainArgs({ ...Base, Kind: "chat", From: "" }, "/s").some((A) => A.startsWith("--From="))).toBe(false);
});

test("PumpLines drops blank lines and keeps a final unterminated line", async () => {
  const Lines: string[] = [];
  await PumpLines(StreamOf("a\n\n  b  \nc"), (L) => Lines.push(L));
  expect(Lines).toEqual(["a", "b", "c"]);
});

// --- ModelHolder (ReloadModel) -------------------------------------------------------------------

const FakeRunnable = (Tag: string): RunnableModel => ({ Model: { Tag }, Chat: false } as unknown as RunnableModel);
const FakeInfo = (Model: unknown): ModelInfo => ({ Tag: (Model as { Tag: string }).Tag } as unknown as ModelInfo);

function Holder(Over: Partial<ModelHolderDeps> = {}): { Holder: ModelHolder; FileLoads: string[] } {
  const FileLoads: string[] = [];
  const Deps: ModelHolderDeps = {
    Store: { Load: async () => null },
    DefaultName: "Seed",
    FallbackFile: "Checkpoints/Foundry.ckpt",
    FromCheckpoint: (Ckpt) => FakeRunnable(String((Ckpt as unknown as { Name: string }).Name)),
    FromFile: (Path) => {
      FileLoads.push(Path);
      return FakeRunnable(`file:${Path}`);
    },
    FileExists: () => true,
    Describe: (Model) => FakeInfo(Model),
    Warn: () => {},
    ...Over,
  };
  return { Holder: new ModelHolder(Deps), FileLoads };
}

test("ReloadModel: the file fallback applies ONLY to the default checkpoint name", async () => {
  const { Holder: H, FileLoads } = Holder();
  expect(await H.Reload("Seed")).toBe(true); // default name, not in Postgres -> file fallback
  expect(H.Source).toBe("file:Checkpoints/Foundry.ckpt");
  expect(H.Name).toBe("Seed");
  expect(await H.Reload("Other")).toBe(false); // another name must NOT get the default file
  expect(FileLoads).toEqual(["Checkpoints/Foundry.ckpt"]);
  expect(H.Name).toBe("Seed"); // name only changes on a successful load
  expect(H.Source).toBe("file:Checkpoints/Foundry.ckpt"); // current model kept serving
});

test("ReloadModel: Postgres wins; a failed load keeps the current model and name", async () => {
  const Store = { Load: async (Name: string): Promise<Checkpoint | null> => (Name === "Good" ? ({ Name } as unknown as Checkpoint) : null) };
  const { Holder: H } = Holder({ Store });
  expect(await H.Reload("Good")).toBe(true);
  expect(H.Source).toBe("postgres:Good");
  expect(await H.Reload("Missing")).toBe(false);
  expect(H.Name).toBe("Good");
  expect(H.Source).toBe("postgres:Good");
});

test("ReloadModel: loads are serialized — a slow load cannot be overtaken by a later one", async () => {
  const Order: string[] = [];
  let ReleaseSlow: () => void = () => {};
  const SlowGate = new Promise<void>((Resolve) => {
    ReleaseSlow = Resolve;
  });
  const Store = {
    Load: async (Name: string): Promise<Checkpoint | null> => {
      Order.push(`start:${Name}`);
      if (Name === "Slow") await SlowGate;
      Order.push(`end:${Name}`);
      return { Name } as unknown as Checkpoint;
    },
  };
  const { Holder: H } = Holder({ Store });
  const First = H.Reload("Slow");
  const Second = H.Reload("Fast");
  await Promise.resolve();
  ReleaseSlow();
  await Promise.all([First, Second]);
  expect(Order).toEqual(["start:Slow", "end:Slow", "start:Fast", "end:Fast"]);
  expect(H.Name).toBe("Fast"); // the LAST requested load is what serves
});

// --- IncrementalDecoder ----------------------------------------------------------------------------

// A byte-level tokenizer stub: each id is one UTF-8 byte, so multibyte chars span several tokens.
const ByteTokenizer = (): Tokenizer & { Calls: number[] } => {
  const Tok = {
    Calls: [] as number[],
    VocabSize: 256,
    Encode: (Text: string): number[] => [...new TextEncoder().encode(Text)],
    Decode: (Ids: number[]): string => {
      Tok.Calls.push(Ids.length);
      return new TextDecoder().decode(new Uint8Array(Ids));
    },
  };
  return Tok as unknown as Tokenizer & { Calls: number[] };
};

test("IncrementalDecoder: multibyte chars assemble correctly and each step decodes a bounded window", () => {
  const Tok = ByteTokenizer();
  const Decoder = new IncrementalDecoder(Tok);
  const Text = "شهد says héllo — ".repeat(20);
  let Out = "";
  for (const Id of Tok.Encode(Text)) Out = Decoder.Push(Id);
  expect(Out).toBe(Text);
  expect(Out.includes(String.fromCharCode(0xfffd))).toBe(false);
  expect(Math.max(...Tok.Calls)).toBeLessThan(16); // never re-decodes the whole (~500-byte) output
});
