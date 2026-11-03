// Data-kind taxonomy (Phase 9): every training document belongs to exactly ONE kind, and each kind
// lives in its OWN table (documents_<kind>) so data types stay physically separate. This lets models
// be trained in different directions — pure code, pure conversation, pure knowledge, or a MIX with a
// chosen size from each — instead of one blended blob. Extensible: add a kind here + a provider mapping
// and the store/migration/training pick it up. code/conversation/knowledge are in use; the rest are
// reserved (their tables exist and stay empty until a provider fills them).

export const DataKinds = ["code", "conversation", "knowledge", "books", "web", "instruction"] as const;
export type DataKind = (typeof DataKinds)[number];

export function IsDataKind(Value: string): Value is DataKind {
  return (DataKinds as readonly string[]).includes(Value);
}

/** The per-kind table name. One table per kind keeps the types cleanly separated. */
export function TableForKind(Kind: DataKind): string {
  return `documents_${Kind}`;
}

/** The kind a "folder" collect lands in when the user picked none (books/articles are the common case). */
export const DefaultFolderKind: DataKind = "books";

/**
 * Which kind table a collect source / provider writes to — the single source of truth for routing.
 * Accepts BOTH the dashboard's source ids (LearnSettings.Source: "github" | "local" | "both" | "oasst" |
 * "oasst2" | "wikipedia" | "wikidump" | "gsm8k" | "folder" | "stackexchange") and the providers' own
 * Name values ("github-repo", "local-repo", "local-folder", "web-search", …):
 *   oasst / oasst2 / stackexchange      -> conversation
 *   wikipedia / wikidump                -> knowledge
 *   gsm8k                               -> instruction
 *   folder / local-folder               -> FolderKind (the user's choice), else books
 *   web-search                          -> web (Raw-only general web)
 *   github / github-repo / local / local-repo / both / anything else -> code
 * The trailing default matches the dashboard's historical routing, where every non-dataset source is a
 * code source.
 */
export function KindForProvider(ProviderName: string, FolderKind?: DataKind): DataKind {
  switch (ProviderName) {
    case "oasst":
    case "oasst2":
    case "stackexchange":
      return "conversation";
    case "wikipedia":
    case "wikidump":
      return "knowledge";
    case "gsm8k":
      return "instruction";
    case "folder":
    case "local-folder":
      return FolderKind ?? DefaultFolderKind;
    case "web-search":
      return "web";
    default:
      return "code";
  }
}
