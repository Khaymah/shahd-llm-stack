// The Train runner (STAGE 2 of the Foundry pipeline): turn the collected corpus into a checkpoint by
// supervising a trainer SUBPROCESS (so training never blocks the dashboard event loop and can run
// alongside collection). Extracted from the dashboard script with the spawn function and the stop-file
// I/O injected, so the supervision logic (arg building, stdout/stderr pumping, graceful stop + kill
// fallback, terminal-event mapping) is unit-testable with a fake process.
//
// Event contract (what OnEvent receives; the dashboard emits train-start itself):
//   train-info     { text }                         — setup lines, stderr lines, stop notice
//   train-progress { step, steps, trainLoss, valLoss?, elapsedMs?, stepMs? }
//   train-done     { savedTo }                      — terminal: exit 0, not stopped
//   train-error    { message, paused? }             — terminal: stopped (paused when the child saved
//                                                     at the exact step and exited 0) or crashed

import { writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { TrainFn, TrainSettings, TrainEvent } from "./DashboardTypes.ts";

/** How long a stopped trainer gets to save its exact-step checkpoint before it is hard-killed. */
export const TrainKillGraceMs = 180_000;
/** SFT stitched multi-turn conversations when the request leaves it unset (~15-25% of the mix). */
export const DefaultMultiTurn = 3000;

/** The supervised child process — the subset of Bun's Subprocess the runner needs. */
export type TrainProcess = {
  Stdout: ReadableStream<Uint8Array>;
  Stderr: ReadableStream<Uint8Array>;
  Exited: Promise<number>;
  Kill: () => void;
};
export type SpawnTrainer = (Args: string[]) => TrainProcess;

export type TrainJobDeps = {
  Spawn?: SpawnTrainer; // default: Bun.spawn with piped stdout/stderr and the parent's env
  StopFilePath?: () => string; // default: a unique file under the OS temp dir
  WriteStopFile?: (Path: string) => void;
  RemoveStopFile?: (Path: string) => void;
  KillGraceMs?: number;
  Log?: (Message: string) => void;
  LogError?: (Message: string) => void;
};

/** Production spawn: the trainer inherits the dashboard's env (DATABASE_URL etc.). */
export const BunSpawnTrainer: SpawnTrainer = (Args) => {
  const Proc = Bun.spawn(Args, { stdout: "pipe", stderr: "pipe", env: { ...process.env } });
  return { Stdout: Proc.stdout, Stderr: Proc.stderr, Exited: Proc.exited, Kill: () => Proc.kill() };
};

/** The trainer command line: pretrain -> base model (TrainOnFoundry); chat -> SFT model (TrainSftChat).
 *  Both workers print the same {Step,TrainLoss,ElapsedMs} JSON progress lines. */
export function BuildTrainArgs(Settings: TrainSettings, StopFile: string): string[] {
  const Common = [
    `--Name=${Settings.Name}`, `--Steps=${Settings.Steps}`, `--Merges=${Settings.Merges}`,
    `--EmbedDim=${Settings.EmbedDim}`, `--Layers=${Settings.NumLayers}`, `--Heads=${Settings.NumHeads}`,
    `--Block=${Settings.BlockSize}`, `--Batch=${Settings.BatchSize}`, `--Workers=${Settings.Workers ?? 0}`,
    `--Precision=${Settings.Precision ?? "F64"}`, `--StopFile=${StopFile}`,
    ...(Settings.Resume ? ["--Resume"] : []), // continue/EXTEND an existing checkpoint of this name
  ];
  if (Settings.Kind === "chat") {
    return ["bun", "run", "Scripts/TrainSftChat.ts", ...Common, `--CodeSamples=${Settings.CodeSamples}`, `--ConvCount=${Settings.ConvCount}`, `--MultiTurn=${Settings.MultiTurn ?? DefaultMultiTurn}`,
      // Warm start from a pretrained base (the pretrain->SFT bridge) — only when a base was picked.
      ...(Settings.From !== undefined && Settings.From !== "" ? [`--From=${Settings.From}`] : [])];
  }
  return ["bun", "run", "Scripts/TrainOnFoundry.ts", ...Common, `--CorpusMb=${Settings.CorpusMb}`, `--KnowledgeMb=${Settings.KnowledgeMb}`];
}

/** One trainer stdout line -> a progress event (JSON with a numeric Step) or an info line. */
export function ParseTrainLine(Line: string, Settings: TrainSettings, OnEvent: (Event: TrainEvent) => void): void {
  if (Line.startsWith("{")) {
    try {
      const J = JSON.parse(Line) as { Step?: number; TrainLoss?: number; ValLoss?: number; ElapsedMs?: number; StepMs?: number };
      if (typeof J.Step === "number") {
        OnEvent({
          kind: "train-progress",
          step: J.Step,
          steps: Settings.Steps,
          trainLoss: J.TrainLoss ?? 0,
          valLoss: typeof J.ValLoss === "number" ? J.ValLoss : undefined,
          elapsedMs: typeof J.ElapsedMs === "number" ? J.ElapsedMs : undefined,
          stepMs: typeof J.StepMs === "number" ? J.StepMs : undefined,
        });
        return;
      }
    } catch {
      // not a progress line — falls through to info
    }
  }
  OnEvent({ kind: "train-info", text: Line });
}

/** Pump a byte stream into whole trimmed lines (blank lines dropped); a final unterminated line is
 *  delivered too, so a crash message without a trailing newline is never lost. */
export async function PumpLines(Stream: ReadableStream<Uint8Array>, OnLine: (Line: string) => void): Promise<void> {
  const Reader = Stream.getReader();
  const Decoder = new TextDecoder();
  let Buf = "";
  for (;;) {
    const { done: Done, value: Value } = await Reader.read();
    if (Done) break;
    if (Value !== undefined) {
      Buf += Decoder.decode(Value, { stream: true });
      let Idx = Buf.indexOf("\n");
      while (Idx >= 0) {
        const Line = Buf.slice(0, Idx).trim();
        Buf = Buf.slice(Idx + 1);
        if (Line !== "") OnLine(Line);
        Idx = Buf.indexOf("\n");
      }
    }
  }
  const Tail = (Buf + Decoder.decode()).trim();
  if (Tail !== "") OnLine(Tail);
}

/** Build the TrainFn the dashboard drives. Stop is GRACEFUL: the train loops are synchronous (stdin/IPC
 *  would starve), so the stop signal is a file the child polls once per step — it saves a checkpoint at
 *  the EXACT current step and exits 0; a hard kill follows only after the grace window. */
export function CreateTrainRunner(Deps: TrainJobDeps = {}): TrainFn {
  const Spawn = Deps.Spawn ?? BunSpawnTrainer;
  const StopFilePath = Deps.StopFilePath ?? ((): string => join(tmpdir(), `shahd-train-stop-${process.pid}-${Date.now()}`));
  const WriteStopFile = Deps.WriteStopFile ?? ((Path: string): void => writeFileSync(Path, "stop"));
  const RemoveStopFile = Deps.RemoveStopFile ?? ((Path: string): void => rmSync(Path, { force: true }));
  const KillGraceMs = Deps.KillGraceMs ?? TrainKillGraceMs;
  const Log = Deps.Log ?? ((Message: string): void => console.log(Message));
  const LogError = Deps.LogError ?? ((Message: string): void => console.error(Message));

  return async (Settings, OnEvent, Signal) => {
    const StopFile = StopFilePath();
    const Args = BuildTrainArgs(Settings, StopFile);
    Log(`[train] ${Settings.Kind} "${Settings.Name}": ${Settings.Steps} steps`);
    const Proc = Spawn(Args);
    const KillQuietly = (): void => {
      try {
        Proc.Kill();
      } catch {
        // already exited
      }
    };
    let KillTimer: ReturnType<typeof setTimeout> | null = null;
    const OnAbort = (): void => {
      try {
        // Ask nicely first; hard-kill only if the child hasn't exited after the grace window (a save
        // on a big model takes a while — the periodic checkpoint still bounds the loss).
        WriteStopFile(StopFile);
        OnEvent({ kind: "train-info", text: "stop requested — finishing the current step and saving a checkpoint at it…" });
        KillTimer = setTimeout(KillQuietly, KillGraceMs);
      } catch {
        KillQuietly();
      }
    };
    if (Signal?.aborted === true) OnAbort(); // stopped before the listener could attach
    else Signal?.addEventListener("abort", OnAbort, { once: true });
    try {
      // Consume stdout (progress) and stderr (crash output) CONCURRENTLY — an unread pipe fills its OS
      // buffer and can stall the child. Stderr lines are logged AND surfaced to the dashboard.
      await Promise.all([
        PumpLines(Proc.Stdout, (Line) => ParseTrainLine(Line, Settings, OnEvent)),
        PumpLines(Proc.Stderr, (Line) => {
          LogError(`[train:stderr] ${Line}`);
          OnEvent({ kind: "train-info", text: `[stderr] ${Line}` });
        }),
      ]);
      const Code = await Proc.Exited;
      if (Signal?.aborted === true) {
        OnEvent(Code === 0
          ? { kind: "train-error", paused: true, message: "paused — checkpoint saved at the exact step you stopped at (and loaded into Chat); press Train (Resume) to continue from there" }
          : { kind: "train-error", message: "stopped — last periodic checkpoint is kept; press Train (Resume) to continue" });
      } else if (Code === 0) {
        OnEvent({ kind: "train-done", savedTo: `postgres:${Settings.Name}` }); // the TRAINED model's name
      } else {
        OnEvent({ kind: "train-error", message: `trainer exited with code ${Code} (see server console)` });
      }
    } catch (Caught) {
      KillQuietly(); // supervision broke (pipe error) — never leave an orphaned trainer running
      throw Caught;
    } finally {
      Signal?.removeEventListener("abort", OnAbort);
      if (KillTimer !== null) clearTimeout(KillTimer);
      RemoveStopFile(StopFile);
    }
  };
}
