# Model Scaling Reference

A practical reference for model sizes: how many parameters, how much data, and what hardware each
needs to **train** and to **run** (inference). Two parts: the in-repo model family (small, CPU-to-modest-GPU)
and a real-world reference for large production-scale models. Figures for large models are
order-of-magnitude references under stated assumptions, not exact guarantees.

Three things are always separate — never one file:

- **Data** — the training corpus (input). Lives in storage; read in mini-batches during training.
- **Training** — the compute process that turns data into weights. Runs on CPU/GPU; owns neither.
- **Model (checkpoint)** — the trained weights (output). Loaded once to run.

---

## 1. The in-repo model family

Parameter counts verified by building each config: `params ≈ (Vocab × Embed) + 16 × Layers × Embed²`.
Head dimension is 64 (standard) except the two smallest tiers. **The columns match the dashboard Train
panel exactly**, so each row is a ready preset. `Batch` and `Steps` are the Chinchilla ~20-tokens/param
**floor** (`steps = tokens ÷ (batch × context)`) — a lower bound; real training uses several times
more. `Corpus MB` is a practical starting size — we currently have ~60 MB collected, so the larger
tiers need much more data (see §3).

| Tier | Embed | Layers | Heads | Context | Vocab | Batch | Steps | Corpus MB | Workers | Precision | Params | Trains on |
|------|-------|--------|-------|---------|-------|-------|-------|-----------|---------|-----------|--------|-----------|
| Seed | 96 | 3 | 4 | 96 | 512 | 16 | ~6,000 | 2 | 8 | F64 | 0.49M | CPU |
| Nano | 128 | 4 | 4 | 256 | 512 | 16 | ~5,000 | 3 | 8 | F64 | 1.1M | CPU |
| Micro | 256 | 6 | 4 | 512 | 1,024 | 16 | ~16,000 | 8 | 8–16 | **F32** | 6.6M | CPU (slow) |
| Mini | 512 | 8 | 8 | 1,024 | 4,096 | 32 | ~22,000 | 30 | — (GPU) | F32 | 36M | small GPU |
| Small | 768 | 12 | 12 | 2,048 | 16,384 | 64 | ~19,000 | 80 | — (GPU) | F32/mixed | 126M | 8–12 GB GPU |
| Base | 1,024 | 24 | 16 | 4,096 | 32,000 | 128 | ~17,000 | 200 | — (GPU) | F32/mixed | 435M | 16–24 GB GPU |
| Large | 2,048 | 32 | 32 | 8,192 | 50,000 | 256 | ~22,000 | 500 | — (GPU) | F32/mixed | 2.25B | 40–80 GB GPU |

Rough equivalents by size: Small ≈ GPT-2 small (124M), Base ≈ GPT-2 medium/large, Large ≈ a small
modern LLM.

Training memory ≈ 4× the weights (weights + gradients + optimizer m/v), plus activations.

### Run knobs — what each one buys and how to pick it

These are the Train-panel fields that are about the RUN, not the architecture. They never change what
the model can learn — they change speed, memory, and (for `From base`) what the weights start from.

| Knob | What it does | How to pick |
|------|--------------|-------------|
| **Workers** | Fans the batch's sequences across that many worker threads — each runs a full forward/backward on its own tape, gradients are reduced deterministically. Parallelizes the WHOLE step (autograd + elementwise + matmuls), which per-kernel threading cannot. Works for pretrain AND chat/SFT. `0` = sequential. | Capped by `Batch` (each worker needs ≥ 1 sequence). **8 is the default sweet spot**; more workers = more concurrent tapes = more RAM, so raise it only with free memory (F32 halves per-worker tape cost, so F32 runs afford more). Drop to 4 if the machine swaps. Measured: 8 workers ≈ 3.8–4.6× over sequential. |
| **Precision** | Storage width of weights/grads/tape. **F64** (8 B): exact, gradient-checkable — the default. **F32** (4 B): HALF the memory everywhere (weights, tape, every worker's slabs) and the 8-lane f32 SIMD kernels (kernel ~1.7×; whole step ~1.15× today — the serial TS share dominates). AdamW moments stay f64 and checkpoints keep one f64 encoding either way, so nothing else changes. | **Seed/Nano: F64** (memory is trivial; keep exactness). **Micro and up: F32** — memory is the binding constraint there, and F32 is what makes wide worker pools + bigger tapes fit. Resume/warm-start IGNORE this field and keep the checkpoint's width (changing width mid-lineage would reinterpret the weights). |
| **From base** (Chat mode) | Seeds the chat run's weights from a finished base checkpoint (the pretrain→SFT bridge) instead of random init. Reuses the base tokenizer verbatim; optimizer/schedule start fresh. | **Always set it for a real chat model** — same-tier base, identical Embed/Layers/Heads/Context (hard requirement). Leave "from scratch" only for quick format experiments where the model's language doesn't matter. |
| **code MB / knowledge MB** (Pretrain) | How many megabytes of each corpus kind feed the base. Code-only → a code model; add knowledge (Wikipedia) for general language. | Scale with the tier (`Corpus MB` column ≈ code + knowledge total). A useful split is ~75% code / 25% knowledge when the chat stage should talk about non-code topics. |
| **conversation / code samples** (Chat) | How many real dialogues (documents_conversation) and code documents (documents_code) feed SFT, on top of the owned synthetic mix (persona/tools/thinking — CLI knobs with sane defaults). | Conversations drive "talks well" — use the chat table's column below. Code samples ground the language-ID task; 4,000 is a good default, 0 for a pure-chat model. |

### Steps vs tokens

`Steps` is not intrinsic to a model — it is `tokens ÷ (batch × context)`, and the real measure of "how
much training" is tokens seen (`steps × batch × context`). The table's Steps are the Chinchilla minimum
(~20 tokens/param) at each tier's batch. The step counts stay modest for the big tiers only because
their batch is large — at a tiny CPU batch the same token budget needs 10–20× more steps, which is
exactly why the larger tiers need a GPU: big batches cut the step count and each step runs far faster.

### Recipe v2 — spec-driven data, one tokenizer, ≤ 4 epochs, WSD

The v1 Micro run saw its ~8 MB corpus ~50 times (it memorised). Recipe v2 changes four things, all
driven by the Data Spec (`Data/Spec.json`, see `Data/Spec.md`):

1. **One persisted tokenizer, trained on the full mix.** `bun run train:bpe` samples `--SampleMb`
   (default 30) from the Filtered tier, split across the spec's slices by `Weight` (default
   `TargetMb`; each slice's `Langs` select its documents; held-out documents excluded; ascending-id
   order, so the sample is deterministic) and learns `VocabSize − 256 − specials` merges (16,384 total
   vocab → 16,114 merges). It writes `Data/Tokenizer.v1.json` and never overwrites one without
   `--Force`. Every training run then reuses it (`--Tokenizer`, default = that file when present);
   a resumed or warm-started run keeps its checkpoint's own merges and refuses an explicit
   `--Tokenizer` that disagrees. The trainer is incremental (16K merges over ~27 MB of code: ~3 s on
   CPU) and reproduces the original trainer merge-for-merge.
2. **The epoch guard.** With `--Spec`, `TrainOnFoundry` builds the pretraining mix per slice and,
   before touching any weight, prints `epochs = Steps × Batch × Context / unique training tokens`. Above
   `Spec.Training.MaxEpochs` (4) it refuses and prints the largest `--Steps` that fits; more data is
   the fix, `--AllowEpochs=<n>` the deliberate override. Without `--Spec` the number is reported only.
3. **WSD schedule** (`Schedule.Kind: "Wsd"`, the default under `--Spec`): linear warmup → constant peak
   → linear decay to `MinLrRatio × peak` over the last `DecayFraction` (default 0.1) of `MaxSteps`. It is
   a pure function of the step, and the flat middle means a run can be extended with `--Resume` (higher
   `--Steps`) before its decay window without an LR jump. `--Schedule=Cosine` keeps the old curve.
4. **SFT respects the split.** `TrainSftChat` drops every code/conversation document on the validation
   side of `Brain/Data/DocumentSplit.ts` (the same rule pretraining and the eval suite's held-out BPB
   use) and logs how many it held out — chat tuning can no longer contaminate held-out BPB.

```bash
bun run train:bpe                                            # Data/Spec.json -> Data/Tokenizer.v1.json
bun run Scripts/TrainOnFoundry.ts --Spec=Data/Spec.json --Name=MicroV2 \
  --EmbedDim=256 --Layers=6 --Heads=4 --Batch=16 --Workers=8 --Precision=F32 --Steps=20000
# refused? it prints the max --Steps for the data you have; add data or lower Steps
bun run Scripts/TrainSftChat.ts --Spec=Data/Spec.json --Name=MicroV2Chat --From=MicroV2 \
  --EmbedDim=256 --Layers=6 --Heads=4 --Steps=3000
```

`--Spec` also sets the context default (`Spec.Training.ContextTokens`, 1,024) for both scripts — pass
the same `--Block` to SFT as the base used, since `--From` requires an identical architecture.

---

## 2. From a base model to a chat model (SFT)

A base (pretrained) model only **autocompletes**. Making it one that **replies and follows a
conversation** is a second stage — **SFT (instruction/chat tuning)** — plus optional tool-use and
thinking on top. Three stages, each reading a different **data kind** (kept in separate tables):

| Stage | What it adds | Data kind(s) | How (dashboard) |
|-------|--------------|--------------|-----------------|
| Pretrain | language + code patterns | code (+ knowledge) | Train ▸ Mode **Pretrain** |
| **SFT (chat)** | reply in the chat format, call tools, think, then stop | **conversation** (+ code) | Train ▸ Mode **Chat / SFT** |
| RL (optional) | prefer better answers | conversation | rejection sampling |

**Warm start (the pretrain→SFT bridge).** A chat run can seed its weights from a finished base
checkpoint instead of random init: pick it in **`From base`** (Chat mode) / pass `--From=<name>`.
The base's tokenizer is reused verbatim and its architecture must match the chat run exactly
(embed/layers/heads/context); precision is inherited from the base. This is what gives the chat model
the base's language patterns underneath its chat behavior — SFT from scratch only ever learns the
format. Pretraining reserves the chat/tool special tokens in the base vocab up front (their
embeddings stay at init until SFT trains them), which is what makes the vocabularies identical across
the two stages.

**One prompt, thinking everywhere (the unified SFT recipe).** The owned SFT mix trains a single
system prompt across every conversation type, and every owned assistant turn opens with a
`<|think|>…<|endthink|>` scratchpad — so think-then-answer is the model's default behavior under the
exact prompt serving presents, and the chat view's reasoning trace shows a trained signal rather than
depending on a prompt variant the model never sees at inference. A tiny model keys behavior on the
literal prompt prefix; splitting the data across near-identical prompt wordings splits its behavior.

**The single most important input for "talks well" is conversation data** (OASST/OASST2 dialogue),
used in the SFT stage. More + more diverse dialogue → better conversational behavior, up to the
model's scale ceiling.

### Chat-model recipe by tier

SFT **keeps the base architecture** — a chat model is a base model plus an instruction stage — so the
config columns mirror the base-family table (§1) and the dashboard Train panel (Chat mode). `SFT steps`
and `Conversation examples` are the extra chat knobs. Note on vocab: the ~17 **special tokens**
(`<|user|>`, `<|assistant|>`, `<|tool_call|>`, `<|think|>`, EOS/FIM, …) sit on top of the `Vocab`
shown, and since the warm-start bridge they are reserved by the PRETRAIN stage too — base and chat
share one vocab layout, so the model's true vocab is `Vocab + specials` at both stages.

`Precision` has no chat column: a warm-started run inherits the BASE's width, and a resumed run its
own checkpoint's — the panel field only matters for a from-scratch chat run (match the tier's §1 row).

| Chat tier | Embed | Layers | Heads | Context | Vocab | Batch | SFT steps | From base | Workers | Conversation examples | Code samples | Realistically expect |
|-----------|-------|--------|-------|---------|-------|-------|-----------|-----------|---------|-----------------------|--------------|----------------------|
| Seed-chat | 96 | 3 | 4 | 96 | 512 | 16 | ~500–800 | Seed | 8 | 1k–5k | 1,000 | learns the FORMAT (replies + stops); output mostly incoherent |
| Nano-chat | 128 | 4 | 4 | 256 | 512 | 16 | ~800–1.5k | Nano | 8 | 3k–10k | 4,000 | replies + calls tools on seen patterns; still incoherent |
| Micro-chat | 256 | 6 | 4 | 512 | 1,024 | 16 | ~2k–4k | Micro | 8–16 | 10k–50k | 4,000–8,000 | short on-topic replies on seen patterns; frequent errors |
| Mini-chat | 512 | 8 | 8 | 1,024 | 4,096 | 32 | ~8k–15k | Mini | — (GPU) | 50k–200k | 10,000+ | simple coherent Q&A + tool calls; not fluent |
| Small-chat | 768 | 12 | 12 | 2,048 | 16,384 | 64 | ~20k+ | Small | — (GPU) | 200k–1M | 20,000+ | basic assistant on narrow tasks (GPT-2-class); needs a GPU |
| *fluent + senior-level code* | *2,048+* | *32+* | *32+* | *8,192+* | *50,000+* | *256+* | *100k+* | — | — | *millions* | — | *emergent at scale — not reachable from scratch on modest hardware* |

`From base` names the SAME-TIER finished base checkpoint (whatever you named it when pretraining that
tier) — the architecture columns must match it exactly. "From scratch" is only for format experiments.

### The data mix (per kind), set from the dashboard

Data types live in separate tables, so a run picks how much of each:

- **Pretrain**: `Code MB` (documents_code) + `Knowledge MB` (documents_knowledge). Code-only for a
  code base; add knowledge for general language.
- **Chat (SFT)**: `Conversations` (documents_conversation) + `Code samples` (documents_code). Set
  Code samples to 0 for a pure-chat model; set Conversations to 0 for a pure-code assistant.

### Honest ceiling

At the tiers that run on modest hardware (Seed–Mini), SFT teaches the **format + the tool/thinking
mechanism** — the model replies, stops, and can call tools — but it will **not** be fluent or write
senior-level code; that is emergent at billions of parameters + trillions of tokens (see §3). The
architecture is complete, so under-training is acceptable; to improve conversation within the ceiling,
collect **more + more diverse conversation data** and raise SFT steps + `Conversations`.

---

## 3. Real-world large models

Inference weight memory is exact math: `bytes-per-parameter × parameters`. FP16 = 2 B, INT8 = 1 B,
INT4 ≈ 0.5 B. Training data uses the Chinchilla compute-optimal rule (~20 tokens/parameter); modern
models are often "over-trained" on far more (Llama-2 used ~2T tokens at every size; Llama-3.1 405B used
~15T). Raw text is roughly ~4 bytes/token, but is usually curated from 10–100× more crawled data.

| Params | Train tokens (opt. → modern) | Weights FP16 | Weights INT4 | Inference GPU (FP16) | Training hardware (reference) |
|--------|------------------------------|--------------|--------------|----------------------|-------------------------------|
| **1.5B** | 30B → ~0.3–1T | 3 GB | ~1 GB | 1× 8 GB | 1× A100, days |
| **7B** | 140B → 1–15T | 14 GB | ~4 GB | 1× 16–24 GB | ~256× A100 · ~30 d (~184k A100-hrs) |
| **13B** | 260B → ~2T | 26 GB | ~7 GB | 1× 48 GB or 2× 24 GB | ~2× the 7B cost (~369k A100-hrs) |
| **34B** | 680B → ~2T | 68 GB | ~17 GB | 2× 48 GB or 1× 80 GB | ~5–8× the 7B cost |
| **70B** | 1.4T → 2–15T | 140 GB | ~35 GB | 2× 80 GB | ~2000× A100 · ~35 d (~1.7M A100-hrs) |
| **175B** | 3.5T → ~0.3–1T | 350 GB | ~88 GB | 8× 80 GB | ~1000s of A100 · weeks · ~$5–12M |
| **405B** | 8.1T → ~15T | 810 GB | ~200 GB | 8–16× 80 GB | ~16k× H100 · ~2 months |
| **1T+** | ~20T+ | ~2 TB | ~500 GB | 16–32× 80 GB (MoE lowers active cost) | large clusters · months |

**Training needs far more hardware than inference.** Mixed-precision training holds ~16–18 bytes per
parameter (weights + grads + Adam states + master copy), so a 7B model needs ~120 GB just for state —
sharded across GPUs (ZeRO / FSDP) — even though its FP16 weights (14 GB) run inference on one card.
4-bit quantization collapses the inference bar: a 7B runs on a laptop, a 70B on ~2× 24 GB.

---

## 4. Context length and the KV-cache

Long context (256k, 1M) is mostly a **KV-cache memory** problem, not a weights problem. During
generation the model caches keys+values for every token:

```
KV-cache bytes ≈ 2 (K,V) × Layers × KVdim × Context × bytes-per-value
```

Example for a 7B-class model (32 layers, hidden 4096, FP16). With standard multi-head attention
(KVdim = 4096) it is ~0.5 MB per token:

| Context | Full MHA (KVdim 4096) | GQA (KVdim 1024) | GQA + INT8 KV |
|---------|-----------------------|------------------|---------------|
| 8k | 4 GB | 1 GB | 0.5 GB |
| 32k | 16 GB | 4 GB | 2 GB |
| 128k | 64 GB | 16 GB | 8 GB |
| 256k | 128 GB | 32 GB | 16 GB |
| 1M | 512 GB | 128 GB | 64 GB |

This is why long-context models rely on **GQA** (fewer KV heads), **KV-cache quantization**, **paged
attention**, and **sliding-window / sparse attention** — otherwise a single 1M-token request would need
more memory than the model weights themselves.

---

## 5. Rules of thumb

```
Weights (inference)   = bytes-per-param × params          (FP16 = 2 B, INT4 ≈ 0.5 B)
Training state memory ≈ 16–18 bytes × params              (weights + grads + Adam + master)
Compute-optimal data  ≈ 20 tokens per parameter           (Chinchilla; modern models use much more)
Tokens seen           = steps × batch × context           (steps = tokens ÷ (batch × context))
Capacity (smartness)  = width (embed) × depth (layers)    → needs GPU beyond ~100M params
Long-context cost      = KV-cache, not weights             → mitigated by GQA + quantization
```

Practical path for this project: stay small on CPU (Seed–Micro) with **F32 storage from Micro up**
(available today via the `Precision` knob — half the memory, 8-lane SIMD kernels), pretrain a base
then **warm-start the chat stage from it** (`From base`), and move to a GPU for Mini and beyond.
