// Rebuild a runnable model + tokenizer from a checkpoint file in one place: load the checkpoint,
// reconstruct the exact config/model/optimizer, apply the weights, and rebuild the tokenizer from
// its persisted state. Shared by Sample (CLI) and the dashboard chat endpoint so the load path is
// not duplicated. Char tokenizer only (the only kind our trainers persist today). Policy sections
// (Tools/Safety/Limits/Compute) are NEVER taken from the checkpoint — see PolicySections below.

import { LoadConfig } from "../Config/LoadConfig.ts";
import { CreateRngStreams } from "../Random/SeededRng.ts";
import type { RngStreams } from "../Random/SeededRng.ts";
import { CharTokenizer } from "../Tokenizer/CharTokenizer.ts";
import { BytePairEncoder } from "../Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import { Shahd } from "../Nn/Shahd.ts";
import { CreateOptimizer } from "../Optim/OptimBarrel.ts";
import { LoadCheckpoint, ApplyCheckpoint } from "./CheckpointReader.ts";
import type { Checkpoint } from "./CheckpointFormat.ts";
import type { ConfigOverride, ResolvedConfig, ShahdConfig } from "../Config/ConfigTypes.ts";
import { DefaultConfig } from "../Config/Constants.ts";

// Chat = the checkpoint was SFT'd on the chat template (Meta.Format === "chat"): serving routes it
// through the agent loop (tools + thinking + reasoning trace). Base models keep the plain path.
export type RunnableModel = { Model: Shahd; Tokenizer: Tokenizer; Config: ResolvedConfig; Rng: RngStreams; Chat: boolean };

// POLICY sections decide what the served model is ALLOWED to do (tool/exec/web access, safety level,
// generation caps, compute backend). A checkpoint is data — possibly downloaded or shared — so it must
// never grant itself capabilities: these always come from the local defaults (Constants.ts), plus
// whatever the operator passes explicitly. Everything else (Model/Tokenizer/Training/Optimizer/
// Schedule/Data) still comes from the checkpoint so the weights load into the exact architecture.
const PolicySections = ["Tools", "Safety", "Limits", "Compute"] as const;
type PolicySection = (typeof PolicySections)[number];

// "Section.Key=value" for every policy value the checkpoint carried that differs from the default.
function NonDefaultPolicy(Config: ShahdConfig): string[] {
  const Found: string[] = [];
  for (const Section of PolicySections) {
    const Carried = (Config as Partial<ShahdConfig>)[Section] as Record<string, unknown> | undefined;
    const Defaults = DefaultConfig[Section] as Record<string, unknown>;
    if (Carried === undefined) continue;
    for (const Key of Object.keys(Carried)) {
      if (JSON.stringify(Carried[Key]) !== JSON.stringify(Defaults[Key])) Found.push(`${Section}.${Key}=${JSON.stringify(Carried[Key])}`);
    }
  }
  return Found;
}

function WithoutPolicy(Config: ShahdConfig): ConfigOverride {
  const Stripped: Partial<Record<keyof ShahdConfig, unknown>> = { ...Config };
  for (const Section of PolicySections) delete Stripped[Section];
  return Stripped as ConfigOverride;
}

// Only the policy sections of the operator's override — it must never reshape the model itself.
function PickPolicy(Overrides: ConfigOverride | undefined): ConfigOverride {
  const Picked: Partial<Record<PolicySection, unknown>> = {};
  if (Overrides === undefined) return Picked as ConfigOverride;
  for (const Section of PolicySections) {
    if (Overrides[Section] !== undefined) Picked[Section] = Overrides[Section];
  }
  return Picked as ConfigOverride;
}

// ApplyCheckpoint warns when the model's ConfigHash differs from the checkpoint's (stale optimizer
// moments under a changed config). Replacing policy changes the hash by design, so present the
// rebuilt hash ONLY when the checkpoint's own full config still reproduces its recorded hash — any
// genuine non-policy drift keeps the original hash and still warns.
function HashAligned(Ckpt: Checkpoint, Config: ResolvedConfig): Checkpoint {
  if (Ckpt.ConfigHash === Config.ConfigHash) return Ckpt;
  const Faithful = LoadConfig({ Overrides: Ckpt.Config as ConfigOverride, UseCli: false, UseEnv: false });
  return Faithful.ConfigHash === Ckpt.ConfigHash ? { ...Ckpt, ConfigHash: Config.ConfigHash } : Ckpt;
}

/** Build a runnable model from an already-parsed checkpoint (file or Postgres — storage-agnostic).
 *  PolicyOverrides (operator config) deep-merge on top of the local defaults for the policy sections;
 *  the checkpoint's own Tools/Safety/Limits/Compute are ignored (with one warning if non-default). */
export function LoadRunnableModelFrom(Ckpt: Checkpoint, PolicyOverrides?: ConfigOverride): RunnableModel {
  const Ignored = NonDefaultPolicy(Ckpt.Config);
  if (Ignored.length > 0) {
    console.warn(`LoadRunnableModel: ignoring policy carried by the checkpoint (local defaults apply): ${Ignored.join(", ")}`);
  }
  // Rebuild the config from the checkpoint minus policy, with the operator's policy (if any) merged
  // onto the local defaults by LoadConfig. UseCli/UseEnv off so nothing else perturbs it.
  const Overrides: ConfigOverride = { ...WithoutPolicy(Ckpt.Config), ...PickPolicy(PolicyOverrides) };
  const Config = LoadConfig({ Overrides, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  const Model = new Shahd(Config, Rng.InitRng);
  const Optimizer = CreateOptimizer(Model.Parameters(), Config);
  ApplyCheckpoint(HashAligned(Ckpt, Config), Model, Optimizer, Rng);

  const Tokenizer = RebuildTokenizer(Ckpt.TokenizerState);
  const Chat = (Ckpt.Meta as Record<string, unknown>)["Format"] === "chat";
  return { Model, Tokenizer, Config, Rng, Chat };
}

/** Build a runnable model from a checkpoint FILE (policy handling as in LoadRunnableModelFrom). */
export function LoadRunnableModel(Path: string, PolicyOverrides?: ConfigOverride): RunnableModel {
  return LoadRunnableModelFrom(LoadCheckpoint(Path), PolicyOverrides);
}

// Rebuild the persisted tokenizer for SERVING. Char tokenizers are built Lenient (an unseen char is
// substituted, never crashes the chat); byte-level BPE has a no-OOV guarantee so needs no leniency.
type CharState = { Kind: "Char"; Chars: string[]; Specials?: string[] };
type BpeState = { Kind: "Bpe"; Merges: [number, number][]; Specials?: string[] };

function RebuildTokenizer(State: unknown): Tokenizer {
  const S = State as CharState | BpeState | null;
  let Base: Tokenizer | null = null;
  if (S !== null && S.Kind === "Char" && Array.isArray(S.Chars)) {
    Base = new CharTokenizer(S.Chars, { Lenient: true });
  } else if (S !== null && S.Kind === "Bpe" && Array.isArray(S.Merges)) {
    Base = new BytePairEncoder({ Merges: S.Merges });
  }
  if (Base === null) throw new Error("LoadRunnableModel: checkpoint has no rebuildable tokenizer state (Char or Bpe)");
  // A chat/SFT checkpoint persists its special tokens; wrap the base so they decode/encode atomically.
  if (S !== null && Array.isArray(S.Specials) && S.Specials.length > 0) return new SpecialTokenizer(Base, S.Specials);
  return Base;
}
