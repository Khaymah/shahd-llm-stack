// Key/Value cache for incremental decoding: stores, per layer and per KV head, the K and V vectors
// of every token processed so far (flat, preallocated for up to Capacity tokens). Lets the cached
// forward attend a new token over all past tokens without recomputing their K/V.
//
// One cache per KV head, not per query head: under GQA (KvHeads < NumHeads) a whole query group
// reads the same K/V, so the cache is NumHeads/KvHeads times smaller than an MHA cache would be.
// Keys are stored AFTER RoPE (rotated at their own absolute position), exactly as attention uses
// them — a cached key never needs re-rotating when later tokens arrive.
//
// Positions in the cache are positions in the CURRENT context window (0 = window start), matching
// the uncached forward, which always starts its window at position 0. Capacity is BlockSize: that
// bounds learned position embeddings (Wpe has BlockSize rows) and is the window Generate slides
// over, so when the cache is full the decoder rebuilds it from the trailing window
// (CachedForward.ts). Buffers are f64 regardless of the run's storage precision — inference math
// in doubles is free accuracy at this size.

import type { Shahd } from "../Nn/Shahd.ts";
import type { NumArray } from "../Tensor/Tensor.ts";

export class KvCache {
  readonly NumLayers: number;
  readonly KvHeads: number;
  readonly HeadDim: number;
  readonly Capacity: number;
  private Keys: Float64Array[]; // index = Layer*KvHeads + KvHead; buffer of Capacity*HeadDim
  private Values: Float64Array[];
  private Lengths: number[]; // tokens appended so far, per (layer, kv head)

  constructor(NumLayers: number, KvHeads: number, HeadDim: number, Capacity: number) {
    this.NumLayers = NumLayers;
    this.KvHeads = KvHeads;
    this.HeadDim = HeadDim;
    this.Capacity = Capacity;
    this.Keys = [];
    this.Values = [];
    this.Lengths = [];
    for (let I = 0; I < NumLayers * KvHeads; I++) {
      this.Keys.push(new Float64Array(Capacity * HeadDim));
      this.Values.push(new Float64Array(Capacity * HeadDim));
      this.Lengths.push(0);
    }
  }

  /** A cache shaped for Model: one slot per (layer, KV head), Capacity = BlockSize. */
  static ForModel(Model: Shahd): KvCache {
    const Cfg = Model.Config;
    return new KvCache(Model.Blocks.length, Cfg.Derived.KvHeads, Cfg.Derived.HeadDim, Cfg.Model.BlockSize);
  }

  /** Tokens fully cached. Every (layer, head) holds the same count between decode steps; the last
   *  slot is the last one a step fills, so it is the conservative answer mid-step too. */
  get Length(): number {
    return this.Lengths[this.Lengths.length - 1] ?? 0;
  }

  get IsFull(): boolean {
    return this.Length >= this.Capacity;
  }

  /** Forget every cached token (buffers are kept and overwritten). */
  Reset(): void {
    this.Lengths.fill(0);
  }

  Append(Layer: number, KvHead: number, K: NumArray, V: NumArray): void {
    this.AppendRows(Layer, KvHead, K, V, 1);
  }

  /** Append Rows consecutive tokens' K/V, each row-major [Rows, HeadDim]. */
  AppendRows(Layer: number, KvHead: number, K: NumArray, V: NumArray, Rows: number): void {
    const Idx = this.SlotIndex(Layer, KvHead);
    const Pos = this.Lengths[Idx];
    if (Pos + Rows > this.Capacity) {
      throw new Error(`KvCache.AppendRows: ${Pos}+${Rows} tokens exceed capacity ${this.Capacity} for (layer ${Layer}, kv head ${KvHead})`);
    }
    const Count = Rows * this.HeadDim;
    if (K.length < Count || V.length < Count) {
      throw new Error(`KvCache.AppendRows: K/V hold ${K.length}/${V.length} values, need ${Count}`);
    }
    const Base = Pos * this.HeadDim;
    const KArr = this.Keys[Idx];
    const VArr = this.Values[Idx];
    for (let D = 0; D < Count; D++) {
      KArr[Base + D] = K[D];
      VArr[Base + D] = V[D];
    }
    this.Lengths[Idx] = Pos + Rows;
  }

  GetKeys(Layer: number, KvHead: number): Float64Array {
    const Idx = this.SlotIndex(Layer, KvHead);
    return this.Keys[Idx].subarray(0, this.Lengths[Idx] * this.HeadDim);
  }

  GetValues(Layer: number, KvHead: number): Float64Array {
    const Idx = this.SlotIndex(Layer, KvHead);
    return this.Values[Idx].subarray(0, this.Lengths[Idx] * this.HeadDim);
  }

  private SlotIndex(Layer: number, KvHead: number): number {
    if (Layer < 0 || Layer >= this.NumLayers || KvHead < 0 || KvHead >= this.KvHeads) {
      throw new Error(`KvCache: (layer ${Layer}, kv head ${KvHead}) outside [${this.NumLayers}, ${this.KvHeads}]`);
    }
    return Layer * this.KvHeads + KvHead;
  }
}
