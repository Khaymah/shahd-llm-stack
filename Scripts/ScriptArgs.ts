// Shared CLI-arg helpers for the entry scripts (single-source, rule #4).
//
// Prefer the TYPED readers (ReadIntArg / ReadFloatArg / ReadBoolArg / ReadNameArg): they fail fast
// with a message naming the flag, instead of `Number(ReadArg(...))` silently turning `--Steps=abc`
// into NaN and a training loop that runs zero steps (NaN < MaxSteps is false) or poisons the LR.
// Every reader takes an optional Argv (defaults to process.argv) so it is unit-testable.

import { z } from "zod";

/** Range constraint for the numeric readers (inclusive on both ends). */
export type NumberBounds = { Min?: number; Max?: number };

// Checkpoint/run names become storage keys (Postgres rows, file names). Mirrors the sanitizer in
// Foundry/Dashboard.ts (same character class, same 40-char cap), but REJECTS instead of rewriting.
const MaxSafeNameLength = 40;
const SafeNamePattern = /^[A-Za-z0-9_.-]+$/;

const FiniteNumber = z.coerce.number().finite();

function CliArgs(): string[] {
  return process.argv.slice(2);
}

/** "--Steps" and "--Steps=" both name the flag "--Steps". */
function FlagName(Name: string): string {
  return Name.endsWith("=") ? Name.slice(0, -1) : Name;
}

/** The raw text after "--Name=", or undefined when the flag is absent. A bare "--Name" (no "=")
 *  for a VALUED flag is a user mistake — it would otherwise be silently ignored. */
function FindValue(Name: string, Argv: readonly string[]): string | undefined {
  const Flag = FlagName(Name);
  const Prefix = `${Flag}=`;
  const Found = Argv.find((Arg) => Arg.startsWith(Prefix));
  if (Found !== undefined) return Found.slice(Prefix.length);
  if (Argv.includes(Flag)) throw new Error(`${Flag} expects a value: write ${Flag}=<value>`);
  return undefined;
}

function CheckBounds(Flag: string, Value: number, Bounds: NumberBounds, Shown: string): number {
  if (Bounds.Min !== undefined && Value < Bounds.Min) {
    throw new Error(`${Flag} must be >= ${Bounds.Min}, got ${Shown}`);
  }
  if (Bounds.Max !== undefined && Value > Bounds.Max) {
    throw new Error(`${Flag} must be <= ${Bounds.Max}, got ${Shown}`);
  }
  return Value;
}

function ParseNumber(Flag: string, Raw: string, Kind: "an integer" | "a number"): number {
  // z.coerce turns "" into 0 — an empty value is a mistake, not zero.
  const Parsed = Raw.trim() === "" ? null : FiniteNumber.safeParse(Raw);
  if (Parsed === null || !Parsed.success || (Kind === "an integer" && !Number.isInteger(Parsed.data))) {
    throw new Error(`${Flag} expects ${Kind}, got '${Raw}'`);
  }
  return Parsed.data;
}

/** Raw string value of "--Name=value" (Prefix includes the "="), or Fallback when absent. */
export function ReadArg(Prefix: string, Fallback: string, Argv: readonly string[] = CliArgs()): string {
  const Prefixed = Prefix.endsWith("=") ? Prefix : `${Prefix}=`;
  const Found = Argv.find((A) => A.startsWith(Prefixed));
  return Found !== undefined ? Found.slice(Prefixed.length) : Fallback;
}

/** Integer flag, e.g. ReadIntArg("--Steps", 600, { Min: 1 }). Throws on non-integers / out of range. */
export function ReadIntArg(Name: string, Default: number, Bounds: NumberBounds = {}, Argv: readonly string[] = CliArgs()): number {
  const Flag = FlagName(Name);
  const Raw = FindValue(Name, Argv);
  if (Raw === undefined) return CheckBounds(Flag, Default, Bounds, `default ${Default}`);
  return CheckBounds(Flag, ParseNumber(Flag, Raw, "an integer"), Bounds, `'${Raw}'`);
}

/** Finite numeric flag, e.g. ReadFloatArg("--Lr", 0.002, { Min: 0 }). Throws on NaN/Infinity/text. */
export function ReadFloatArg(Name: string, Default: number, Bounds: NumberBounds = {}, Argv: readonly string[] = CliArgs()): number {
  const Flag = FlagName(Name);
  const Raw = FindValue(Name, Argv);
  if (Raw === undefined) return CheckBounds(Flag, Default, Bounds, `default ${Default}`);
  return CheckBounds(Flag, ParseNumber(Flag, Raw, "a number"), Bounds, `'${Raw}'`);
}

const TrueWords: readonly string[] = ["true", "1", "yes", "on"];
const FalseWords: readonly string[] = ["false", "0", "no", "off"];

/** Boolean flag: bare "--Fresh" means true; "--Fresh=true|false|1|0|yes|no|on|off" is explicit. */
export function ReadBoolArg(Name: string, Default: boolean, Argv: readonly string[] = CliArgs()): boolean {
  const Flag = FlagName(Name);
  if (Argv.includes(Flag)) return true;
  const Found = Argv.find((A) => A.startsWith(`${Flag}=`));
  if (Found === undefined) return Default;
  const Raw = Found.slice(Flag.length + 1);
  const Word = Raw.trim().toLowerCase();
  if (TrueWords.includes(Word)) return true;
  if (FalseWords.includes(Word)) return false;
  throw new Error(`${Flag} expects a boolean (true/false), got '${Raw}'`);
}

/** True when a bare boolean flag (e.g. "--DryRun") is present (or set "=true") in argv. */
export function ReadFlag(Flag: string, Argv: readonly string[] = CliArgs()): boolean {
  return ReadBoolArg(Flag, false, Argv);
}

/** Return Value unchanged if it is a safe storage key, else throw. Names are NOT rewritten: a
 *  sanitizer mapping "my model" -> "my-model" would make two different names share one checkpoint
 *  key, and a later save would silently overwrite the other run. */
export function AssertSafeName(Value: string, What = "name"): string {
  if (Value === "") throw new Error(`${What} must not be empty`);
  if (Value.length > MaxSafeNameLength) {
    throw new Error(`${What} '${Value}' is longer than ${MaxSafeNameLength} characters`);
  }
  if (!SafeNamePattern.test(Value)) {
    const Suggestion = Value.trim().replace(/[^A-Za-z0-9_.-]+/g, "-").slice(0, MaxSafeNameLength);
    throw new Error(
      `${What} '${Value}' may only contain letters, digits, '_', '.' and '-' (no spaces); ` +
        `rename it explicitly, e.g. '${Suggestion}'`,
    );
  }
  return Value;
}

/** A run/checkpoint name flag, validated by AssertSafeName (e.g. ReadNameArg("--Name", "Chat")). */
export function ReadNameArg(Name: string, Default: string, Argv: readonly string[] = CliArgs()): string {
  const Flag = FlagName(Name);
  return AssertSafeName(ReadArg(Flag, Default, Argv), Flag);
}
