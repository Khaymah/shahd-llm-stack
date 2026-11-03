// Causal multi-head self-attention via PER-HEAD weight matrices whose projected outputs are
// SUMMED (equivalent to concat-then-Wo, verified in REVIEW.md), reusing the 2D ops with NO new
// slice/concat primitives. Supports Grouped-Query Attention: NumHeads query heads share KvHeads
// key/value heads (GroupSize = NumHeads/KvHeads query heads per KV head). MHA when KvHeads =
// NumHeads. Attention scale comes from DeriveConfig (1/sqrt(HeadDim)); RoPE optional.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, MatMulNT, CausalSoftmax, Add, ApplyRope } from "../Ops/OpsBarrel.ts";
import { InitWeight } from "./InitPolicy.ts";

/** Receives each KV head's attention inputs as they are built: K AFTER RoPE and V, both [T, HeadDim].
 *  The KV-cached decoder uses it to fill its cache from one batched forward (Sampling/CachedForward). */
export type KvSink = (KvIndex: number, K: Tensor, V: Tensor) => void;

export class MultiHeadAttention {
  WqHeads: Tensor[] = []; // [NumHeads] of [EmbedDim, HeadDim]
  WoHeads: Tensor[] = []; // [NumHeads] of [HeadDim, EmbedDim] (residual projection)
  WkHeads: Tensor[] = []; // [KvHeads] of [EmbedDim, HeadDim] (shared across a query group)
  WvHeads: Tensor[] = []; // [KvHeads] of [EmbedDim, HeadDim]
  NumHeads: number;
  KvHeads: number;
  GroupSize: number;
  AttentionScale: number;
  UseRope: boolean;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim, NumHeads } = Config.Model;
    const { HeadDim, AttentionScale, KvHeads } = Config.Derived;
    this.NumHeads = NumHeads;
    this.KvHeads = KvHeads;
    this.GroupSize = NumHeads / KvHeads;
    this.AttentionScale = AttentionScale;
    this.UseRope = Config.Model.PositionScheme === "Rope";
    for (let H = 0; H < NumHeads; H++) {
      this.WqHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WoHeads.push(InitWeight(HeadDim, EmbedDim, Rng, Config, true)); // scaled residual init
    }
    for (let Kv = 0; Kv < KvHeads; Kv++) {
      this.WkHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WvHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
    }
  }

  Forward(X: Tensor, Sink?: KvSink): Tensor {
    // Project each KV head once; it is shared by its whole query group (GQA). Scores use MatMulNT
    // (Q @ Kᵀ with K read row-wise), so no transposed copy of K is ever built — under GQA every
    // query head of the group accumulates its dK straight into the shared K node.
    const Ks: Tensor[] = [];
    const Vs: Tensor[] = [];
    for (let Kv = 0; Kv < this.KvHeads; Kv++) {
      let K = MatMul(X, this.WkHeads[Kv]);
      if (this.UseRope) K = ApplyRope(K, 0);
      const V = MatMul(X, this.WvHeads[Kv]);
      Ks.push(K);
      Vs.push(V);
      Sink?.(Kv, K, V);
    }

    let Output: Tensor | null = null;
    for (let H = 0; H < this.NumHeads; H++) {
      const KvIndex = Math.floor(H / this.GroupSize);
      let Q = MatMul(X, this.WqHeads[H]); // [T, HeadDim]
      if (this.UseRope) Q = ApplyRope(Q, 0);
      const Scores = MatMulNT(Q, Ks[KvIndex]); // [T, T] = Q @ Kᵀ
      const Weights = CausalSoftmax(Scores, this.AttentionScale); // softmax(scale·scores) over j <= i
      const HeadOut = MatMul(Weights, Vs[KvIndex]); // [T, HeadDim]
      const Projected = MatMul(HeadOut, this.WoHeads[H]); // [T, EmbedDim]
      Output = Output === null ? Projected : Add(Output, Projected);
    }
    if (Output === null) throw new Error("MultiHeadAttention: NumHeads must be >= 1");
    return Output;
  }

  Parameters(): Tensor[] {
    return [...this.WqHeads, ...this.WkHeads, ...this.WvHeads, ...this.WoHeads];
  }
}
