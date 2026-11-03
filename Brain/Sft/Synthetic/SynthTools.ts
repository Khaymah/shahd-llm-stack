// The tool surface synthetic conversations may use, and the deterministic context they run under. The
// same pair drives generation (ToolUseGenerator executes the REAL tool) and validation (the replay in
// ChatDocumentValidation re-executes it and compares), so a stored result can always be re-derived.
//
// Only side-effect-free or sandboxed tools: the math tools, the clock (injected per conversation), the
// in-memory key/value memory (fresh per conversation) and the READ-ONLY file tools confined to a small
// committed fixture workspace. run_code and web_search are never synthesized: their output is either a
// subprocess's or the network's, neither of which is reproducible training data.

import { fileURLToPath } from "node:url";
import { ToolRegistry } from "../../Serving/Tools/ToolRegistry.ts";
import { DefaultToolRegistry } from "../../Serving/Tools/BuildRegistry.ts";
import { DefaultToolContext } from "../../Serving/Tools/DefaultProviders.ts";
import { Workspace } from "../../Serving/Tools/Workspace.ts";
import type { ToolContext } from "../../Serving/Tools/ToolTypes.ts";
import type { ChatDocument } from "../ChatDocument.ts";
import type { ToolReplay } from "../ChatDocumentValidation.ts";

/** The tools a synthetic conversation may call. */
export const SynthToolNames: readonly string[] = [
  "calculator", "stats", "current_time", "memory_store", "memory_recall", "file_read", "file_list", "file_search",
];

/** The committed fixture workspace the file tools read (Tests/Fixtures/SftWorkspace). */
export const SftWorkspaceRoot = fileURLToPath(new URL("../../../Tests/Fixtures/SftWorkspace", import.meta.url));

/** The clock a document's tool steps ran under when its Meta records none (2025-01-01T00:00:00Z). */
export const DefaultSynthClockMs = 1_735_689_600_000;

/** The synthesis registry: the safe default registry narrowed to SynthToolNames. */
export function SynthToolRegistry(): ToolRegistry {
  const Source = DefaultToolRegistry();
  const Registry = new ToolRegistry();
  for (const Name of SynthToolNames) {
    const Tool = Source.Get(Name);
    if (Tool === undefined) throw new Error(`SynthToolRegistry: "${Name}" is not in the default registry`);
    Registry.Register(Tool);
  }
  return Registry;
}

/** A fresh context for one conversation: empty memory, the fixture workspace, the given clock. */
export function SynthToolContext(ClockMs: number, Root: string = SftWorkspaceRoot): ToolContext {
  return { ...DefaultToolContext({ Workspace: new Workspace(Root) }), Clock: () => ClockMs };
}

/** Generation + validation share this: the registry, and a per-document context whose clock is the one
 *  recorded in the document's Meta. */
export function SynthToolReplay(Root: string = SftWorkspaceRoot): ToolReplay {
  return {
    Registry: SynthToolRegistry(),
    Context: (Doc: ChatDocument) => SynthToolContext(Doc.Meta.ClockMs ?? DefaultSynthClockMs, Root),
  };
}
