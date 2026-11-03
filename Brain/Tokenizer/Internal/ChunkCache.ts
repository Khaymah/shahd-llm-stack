// A bounded, approximately-least-recently-used cache from a pretoken chunk to its BPE ids. Source code
// repeats the same chunks constantly (keywords, common identifiers, indentation runs), so most chunks
// after warm-up skip the merge loop entirely.
//
// Two generations instead of a strict LRU list: lookups hit the Young map first, then the Old one (an
// Old hit is promoted to Young). When Young fills half the capacity, Old is dropped wholesale and Young
// becomes Old. Anything used within the last generation survives; anything unused for two generations
// is evicted. Measured: a strict LRU via Map delete + re-insert per hit cost more than the merge loop it
// saves (~1.5 µs per hit); a plain Map.get is ~30 ns. Cached arrays are never handed out mutably —
// callers copy the ids into their own output.

// Chunks retained per encoder. ~64K distinct chunks covers the working set of a code corpus while
// bounding memory to a few MB (each entry is a short string + a short id array).
export const DefaultChunkCacheCapacity = 65_536;
const Generations = 2;

export class ChunkCache {
  private Young = new Map<string, readonly number[]>();
  private Old = new Map<string, readonly number[]>();
  private readonly GenerationSize: number;

  constructor(Capacity = DefaultChunkCacheCapacity) {
    if (!Number.isInteger(Capacity) || Capacity < Generations) throw new Error(`ChunkCache: capacity must be an integer >= ${Generations}, got ${Capacity}`);
    this.GenerationSize = Math.floor(Capacity / Generations);
  }

  /** Entries currently held (never more than the capacity). */
  get Size(): number {
    return this.Young.size + this.Old.size;
  }

  Get(Chunk: string): readonly number[] | undefined {
    const Hot = this.Young.get(Chunk);
    if (Hot !== undefined) return Hot;
    const Cold = this.Old.get(Chunk);
    if (Cold !== undefined) this.Set(Chunk, Cold);
    return Cold;
  }

  Set(Chunk: string, Ids: readonly number[]): void {
    if (this.Young.size >= this.GenerationSize) {
      this.Old = this.Young;
      this.Young = new Map();
    }
    this.Young.set(Chunk, Ids);
  }
}
