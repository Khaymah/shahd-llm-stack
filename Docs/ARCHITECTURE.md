# Architecture

Shahd is a decoder-only transformer language model with a from-scratch automatic-differentiation
engine. Everything runs on the CPU today in TypeScript (Bun), with an optional Go compute backend for
hot numeric kernels. The design goal is a small, readable, fully-owned engine that can be scaled and
evolved without depending on any external model.

## Layered design

The layers form an acyclic dependency graph, lowest to highest:

1. **Config** (`Brain/Config/`) — a single Zod-validated configuration object. `Constants.ts` is the
   only source of defaults; values are validated, derived (head dim, attention scale, MLP hidden
   size…), hashed, and deep-frozen. The frozen config is threaded everywhere and embedded verbatim in
   every checkpoint.
2. **Tensor + Autograd** (`Brain/Tensor/`, `Brain/Autograd/`) — a `Tensor` is a flat `Float64Array`
   of data and gradient with a shape and a backward closure. Reverse-mode autograd walks the tape in
   topological order and accumulates gradients. `GradCheck` (finite differences) is the numerical
   oracle that every op is verified against.
3. **Ops** (`Brain/Ops/`) — the differentiable primitives: matmul, add, bias, scale, transpose,
   softmax, cross-entropy (masked and unmasked), layer/RMS norm, GELU/SiLU, rotary embeddings,
   element-wise multiply. Each has an explicit forward and a `_backward` closure.
4. **Neural network** (`Brain/Nn/`) — the `Shahd` model: token/position embeddings, causal
   multi-head self-attention (with optional rotary positions and grouped-query attention), a gated or
   plain MLP, pre-norm residual blocks, a final norm, and a language-model head tied to the token
   embedding.
5. **Optimization + Training** (`Brain/Optim/`, `Brain/Training/`) — Adam/AdamW with decoupled weight
   decay, global-norm gradient clipping, warmup+cosine schedules; a training step, gradient
   accumulation (the accumulated gradient is divided by the batch count, never the learning rate), an
   eval loop reporting bits-per-byte, and the top-level train loop.
6. **Tokenizer + Data** (`Brain/Tokenizer/`, `Brain/Data/`) — a character tokenizer and a byte-level
   BPE (no out-of-vocabulary by construction) with code-aware pretokenization and atomic special
   tokens; and a corpus pipeline (license allowlist, MinHash near-dedup, quality filter, eval
   decontamination, fill-in-the-middle reformatting) assembled by `CorpusBuilder`.
7. **Sampling + Checkpoints** (`Brain/Sampling/`, `Brain/Checkpoint/`) — temperature/top-k/top-p/min-p
   sampling, autoregressive generation, a KV-cache for every architecture (Learned/RoPE, LayerNorm/RMSNorm,
   ReLU/SwiGLU, MHA/GQA) that is numerically identical to the full forward pass, and self-describing checkpoints (weights + optimizer state + RNG state + config + hash, with
   a shape-mismatch hard-fail on load).
8. **Safety** (`Brain/Safety/`) — a deterministic, controllable content-safety filter and resource
   limits, wrapped by `GuardedGenerate`, the single safe entry point for product generation.
9. **SFT, Eval, RL** (`Brain/Sft/`, `Brain/Eval/`, `Brain/Rl/`) — a chat template with loss masking,
   a task taxonomy and tool-use exemplars, pass@k with a resource-capped code runner (not a sandbox; nonce-verified tests), and RLVR via
   rejection sampling.
10. **Serving + Reasoning** (`Brain/Serving/`, `Brain/Reasoning/`) — a tool-calling protocol, a rich
    tool system with a central capability gate, a multi-step agent loop, an OpenAI-compatible server,
    and reasoning utilities (speculative decoding, self-consistency, tree-of-thoughts, thinking-mode).
11. **Compute backend** (`Brain/ComputeBackend/`) — a seam over flat numeric buffers with zero
    knowledge of autograd. Only hot-op numeric bodies may be routed to a backend (TypeScript today; an
    in-process Go FFI kernel that is 2–8× faster on CPU; a subprocess variant). The autograd tape
    always stays in TypeScript.

## Key design choices

- **Multi-head attention uses per-head weight matrices summed**, which is mathematically equivalent to
  the concatenate-then-project formulation but avoids slice/concat ops on the tape.
- **Weight tying** shares the token embedding with the language-model head (`MatMulNT`, no transpose copy), reducing
  parameters and coupling input/output representations.
- **Scaled residual initialization** shrinks residual-projection weights by `1/sqrt(2·layers)` for
  stable deep training.
- **Config-selectable modern stack**: learned vs rotary positions, LayerNorm vs RMSNorm, ReLU vs
  SwiGLU/GeGLU MLP, and grouped-query attention — all behind config, with cross-field invariants
  enforced in one place.
- **Safety and capability gates are central and controllable.** Content safety, resource limits, and
  tool capabilities (filesystem/exec/network) each live in a dedicated config section and are
  deny-by-default for anything risky.

## Testing philosophy

Correctness is anchored on gradient checking: every op and the full model are verified against finite
differences. Behavioral tests confirm mechanisms end-to-end (a model trained on a known pattern
reproduces it; the KV-cache matches the full forward exactly; checkpoints round-trip). The continuous
gate runs type-checking, the 600-line and PascalCase checks, linting, gradient checking, and the full
test suite.
