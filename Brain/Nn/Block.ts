// A transformer block: pre-norm + residual around attention, then pre-norm + residual around
// the MLP (GPT-2 style). Matches nano-gpt.ts's block structure.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { Add } from "../Ops/OpsBarrel.ts";
import { NormLayer } from "./NormLayer.ts";
import { MultiHeadAttention } from "./MultiHeadAttention.ts";
import type { KvSink } from "./MultiHeadAttention.ts";
import { Mlp } from "./Mlp.ts";

export class Block {
  Ln1: NormLayer;
  Attn: MultiHeadAttention;
  Ln2: NormLayer;
  Mlp: Mlp;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim } = Config.Model;
    this.Ln1 = new NormLayer(EmbedDim, Config);
    this.Attn = new MultiHeadAttention(Config, Rng);
    this.Ln2 = new NormLayer(EmbedDim, Config);
    this.Mlp = new Mlp(Config, Rng);
  }

  /** Sink (optional) observes this block's per-KV-head K/V — see MultiHeadAttention.KvSink. */
  Forward(X: Tensor, Sink?: KvSink): Tensor {
    let Y = Add(X, this.Attn.Forward(this.Ln1.Forward(X), Sink)); // pre-norm + residual
    Y = Add(Y, this.Mlp.Forward(this.Ln2.Forward(Y)));
    return Y;
  }

  Parameters(): Tensor[] {
    return [
      ...this.Ln1.Parameters(),
      ...this.Attn.Parameters(),
      ...this.Ln2.Parameters(),
      ...this.Mlp.Parameters(),
    ];
  }
}
