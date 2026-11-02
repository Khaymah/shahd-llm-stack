# Shahd (شهد)

A **from-scratch, fully-owned LLM stack in TypeScript** (Bun) with native Go/C compute kernels —
data engine → tokenizer → training → SFT → agent serving, all built here. No open-source model
weights, no fine-tuning of someone else's brain: the autograd engine, transformer, tokenizer,
training loop, safety, tools, and reasoning are owned and meant to be evolved by their owner.
Helper libraries (Zod, etc.) are fine; the *stack* is owned. The released models are the proof it
works end-to-end — see [Docs/POSITIONING.md](Docs/POSITIONING.md) for what exists around it and
why this category is nearly empty.

> **Status (honest):** the full pipeline is proven end-to-end and released as
> [**v0.1.0**](https://github.com/sabrydawood/shahd-llm-stack/releases/tag/v0.1.0) — four trained
> checkpoints, flagship **MicroChat (6.56M params)**: it greets, calls its calculator tool with
> correct arguments, answers identity questions, holds short multi-turn exchanges, and shows its
> full reasoning (think → tool → answer) live in the dashboard, persisted with every reply. These
> are tiny CPU-trained models — they master their trained distribution, not open-ended fluency;
> real capability needs scale (see [Docs/MODEL-SCALING.md](Docs/MODEL-SCALING.md)).

![Chat with the live reasoning trace](Docs/screenshots/shahd-chat-reasoning.png)

## Quickstart

```bash
bun install
bun run ci          # typecheck + length + naming + lint + gradcheck + all tests
bun test            # tests only
bun run gradcheck   # finite-difference gradient check (the numerical oracle)

bun run train       # train a char-level model on a built-in code sample, then sample
bun run sample      # sample from a saved checkpoint
bun run Scripts/TrainOnCorpus.ts     # train on the permissive seed corpus
bun run Scripts/PhaseSevenDemo.ts    # capstone: corpus -> tools agent -> speculative -> safety
bun run Scripts/ComputeSpike.ts 512  # benchmark TS vs Go FFI matmul

bun run foundry:dashboard            # control plane: collect data -> train -> chat (http://localhost:8090)
```

## Trained models (v0.1.0)

Downloadable from the [release page](https://github.com/sabrydawood/shahd-llm-stack/releases/tag/v0.1.0).
Each `.ckpt` is the **full training state** (weights + AdamW moments + RNG + tokenizer + config,
checksummed), so a downloaded model can be run — `LoadRunnableModel(path)` from
`Brain/Checkpoint/LoadRunnableModel.ts` — and also resumed (trained further) bit-identically.

| Model | Params | Architecture | Kind |
|-------|--------|--------------|------|
| **MicroChat** | 6.56M | emb 256 · 6L · 4h · ctx 512 | chat — tools, thinking, multi-turn |
| NanoChat2 | 2.55M | emb 192 · 4L · 4h · ctx 256 | chat — balanced mix |
| NanoChat | 2.55M | emb 192 · 4L · 4h · ctx 256 | chat — first mix (comparison) |
| foundry | 1.12M | emb 128 · 4L · 4h · ctx 256 | base — autocomplete |

<p>
<img src="Docs/screenshots/shahd-models.png" width="49%" alt="Models view">
<img src="Docs/screenshots/shahd-train.png" width="49%" alt="Train panel">
</p>

The full acceptance conversation (greeting → calculator tool call → identity, one conversation,
per-reply persisted reasoning): [Docs/screenshots/shahd-chat-full-conversation.png](Docs/screenshots/shahd-chat-full-conversation.png).

## Train a model automatically (GitHub Actions)

The **Train model** workflow (`.github/workflows/train-model.yml`) runs the whole pipeline on a GitHub
runner — collect the CI corpus from `Data/Spec.ci.json` (explicit permissive repos, `GITHUB_TOKEN`),
train the tokenizer, pretrain under the epoch cap, SFT, evaluate — and publishes the checkpoints,
tokenizer, eval reports and a model card as a workflow artifact (about an hour for the 3M-parameter
CI tier). Start it from **Actions → Train model → Run workflow** (inputs: epochs, embed, layers, SFT
steps, name), or push a `model-v*` tag to also publish a GitHub Release. Then:

```bash
bun run serve --Checkpoint=Checkpoints/CiChat.ckpt      # OpenAI-compatible API on 127.0.0.1:8080
bun run eval  --Checkpoint=Checkpoints/CiChat.ckpt      # the frozen evaluation suite
```

Locally the same recipe is `bun run foundry:ingest-spec` → `bun run train:bpe` →
`bun run Scripts/TrainOnFoundry.ts --Spec=Data/Spec.json --Epochs=2 …` → `Scripts/TrainSftChat.ts --From=…`
(see [Docs/MODEL-SCALING.md](Docs/MODEL-SCALING.md), "Recipe v2").

## Architecture (layered, `Brain/`)

| Layer                     | What it does                                                                                              |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `Config/`               | Zod-validated config;`Constants.ts` is the single source of defaults; derive + hash + freeze            |
| `Tensor/` `Autograd/` | Float64 tensors + reverse-mode autograd (topological backward);`GradCheck` is the oracle                |
| `Ops/`                  | Core differentiable ops (matmul, softmax, norms, RoPE, SwiGLU, cross-entropy…)                           |
| `Nn/`                   | `Shahd` transformer: embeddings, multi-head attention (+RoPE/GQA), MLP, blocks, weight tying            |
| `Optim/`                | Adam/AdamW, grad clipping, LR schedule                                                                    |
| `Tokenizer/`            | Char + byte-level BPE (no-OOV), code-aware pretokenization, special tokens                                |
| `Data/`                 | Corpus pipeline: license allowlist, MinHash dedup, quality filter, decontamination, FIM,`CorpusBuilder` |
| `Training/`             | Training step, gradient accumulation, eval loop (bits-per-byte), train loop                               |
| `Sampling/`             | Temperature/top-k/top-p/min-p sampling, generation,**KV-cache** (numerically exact)                 |
| `Checkpoint/`           | Self-describing checkpoints (weights + optimizer + RNG + config + hash)                                   |
| `Safety/`               | Controllable content-safety filter + resource limits +`GuardedGenerate` (absolute priority)             |
| `Sft/`                  | Chat template + loss masking, task taxonomy, tool-use exemplars                                           |
| `Eval/` `Rl/`         | pass@k, resource-capped code runner (not a sandbox), RLVR rejection sampling                                                  |
| `Serving/`              | Tool protocol,**rich tool system** (`Tools/`), agent loop, OpenAI-compatible server               |
| `Reasoning/`            | Speculative decoding, self-consistency, tree-of-thoughts, thinking-mode                                   |
| `ComputeBackend/`       | Pluggable numeric backend seam: TS, Go subprocess, Go FFI (in-process, 2–8× on CPU)                     |

## Conventions (enforced by CI)

1. **PascalCase** everything we declare (`check:naming` + ESLint).
2. **≤ 600 lines** per file (`check:length`).
3. **DRY-strict** — no duplicated function/constant; check before writing.
4. **Central constants**, config validated by Zod in one place.
5. **Acyclic imports**, explicit named barrels (no `index.ts`).

Full rules: [Docs/CONVENTIONS.md](Docs/CONVENTIONS.md). Design: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md).
File-by-file layout: [Docs/STRUCTURE.md](Docs/STRUCTURE.md). Where it's going: [Docs/ROADMAP.md](Docs/ROADMAP.md).
Model sizes, hardware, and the scaling path: [Docs/MODEL-SCALING.md](Docs/MODEL-SCALING.md).
Positioning + the competitive landscape: [Docs/POSITIONING.md](Docs/POSITIONING.md).

## Data

The **Data Foundry** is the tiered, inspectable dataset layer and the collection engine that fills it:
multiple licensed sources (GitHub code, OASST/Stack Exchange dialogue, Wikipedia knowledge, GSM8K
reasoning, local folders of books), each routed to its own per-kind table, with honest dedup
accounting and resumable collection. Drive it from the dashboard (`bun run foundry:dashboard`).
See [Docs/DATA-FOUNDRY.md](Docs/DATA-FOUNDRY.md).

## Safety

Safety and performance are absolute priorities living in dedicated, controllable places
(`Brain/Safety/`, `Config.Safety`, `Config.Limits`, `Config.Tools`). The model must never assist harm;
the content filter and capability gates can be strengthened or disabled centrally via config.

## Community

Want to take part — ideas, data sources, testing, or just following along? Join the Discord:
**<https://discord.gg/v4ACAn5CKf>**

## License

Proprietary — © Sabry. All rights reserved. See [LICENSE](LICENSE).
