// Explicit named barrel for the Ops layer (rule: no index.ts anywhere — a folder's public
// surface is a named file). Consumers import ops from here.
//
// Hot-loop convention for every op: `.Data`, `.Grad`, and `.Size` are read ONCE into locals before
// the loops (`.Grad`/`.Size` are getters — Grad materializes lazily — so a per-element read cost a
// call + branch every iteration). Backward closures hoist INSIDE the closure body, never at forward
// time: capturing a parent's buffer when the node is built would keep it alive after Backward()
// releases it (Tensor.ReleaseBuffers), defeating release-on-consume.

export { MatMul } from "./MatMul.ts";
export { MatMulNT } from "./MatMulNT.ts";
export { Add } from "./Add.ts";
export { AddBias } from "./AddBias.ts";
export { Scale } from "./Scale.ts";
export { ReLU } from "./ReLU.ts";
export { Transpose } from "./Transpose.ts";
export { CausalMask } from "./CausalMask.ts";
export { SoftmaxRows } from "./SoftmaxRows.ts";
export { CausalSoftmax } from "./CausalSoftmax.ts";
export { LayerNorm } from "./LayerNorm.ts";
export { EmbeddingLookup } from "./EmbeddingLookup.ts";
export { CrossEntropy } from "./CrossEntropy.ts";
export { MaskedCrossEntropy } from "./MaskedCrossEntropy.ts";
export { ApplyRope, DefaultRopeBase, RopeFrequencies, RopeCosSin, RotateRow } from "./RotaryEmbedding.ts";
export { RmsNorm } from "./RmsNorm.ts";
export { Gelu, GeluValue } from "./Gelu.ts";
export { Silu, SiluValue } from "./Silu.ts";
export { Mul } from "./Mul.ts";
