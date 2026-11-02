// A binary max-heap of (Count, Order, Key) entries for incremental BPE training. Ordering: higher
// Count first; on equal Count, LOWER Order first (Order = the pair's first-occurrence position, which is
// the reference trainer's tie-break). The heap is "lazy": the trainer never updates or removes an entry
// in place — it pushes a fresh entry whenever a pair's statistics change, and discards popped entries
// whose (Count, Order) no longer match the live values. Storage is struct-of-arrays Float64Arrays
// (no per-entry object), grown by doubling.

const InitialCapacity = 1024;
const GrowthFactor = 2;

export class PairHeap {
  Size = 0;
  private Counts: Float64Array;
  private Orders: Float64Array;
  private Keys: Float64Array;

  constructor(Capacity = InitialCapacity) {
    const Cap = Math.max(1, Capacity);
    this.Counts = new Float64Array(Cap);
    this.Orders = new Float64Array(Cap);
    this.Keys = new Float64Array(Cap);
  }

  /** Count of the top entry (undefined behavior when empty — check Size first). */
  TopCount(): number {
    return this.Counts[0];
  }

  TopOrder(): number {
    return this.Orders[0];
  }

  TopKey(): number {
    return this.Keys[0];
  }

  Clear(): void {
    this.Size = 0;
  }

  Push(Count: number, Order: number, Key: number): void {
    if (this.Size === this.Counts.length) this.Grow();
    let I = this.Size++;
    // Sift up: move parents down until the new entry's slot is found, then write it once.
    while (I > 0) {
      const Parent = (I - 1) >> 1;
      if (!this.Beats(Count, Order, this.Counts[Parent], this.Orders[Parent])) break;
      this.Move(Parent, I);
      I = Parent;
    }
    this.Counts[I] = Count;
    this.Orders[I] = Order;
    this.Keys[I] = Key;
  }

  /** Remove the top entry. */
  Pop(): void {
    if (this.Size === 0) return;
    const Last = --this.Size;
    if (Last === 0) return;
    const Count = this.Counts[Last];
    const Order = this.Orders[Last];
    const Key = this.Keys[Last];
    let I = 0;
    // Sift down the former last entry from the root.
    for (;;) {
      const Left = 2 * I + 1;
      if (Left >= Last) break;
      const Right = Left + 1;
      let Child = Left;
      if (Right < Last && this.Beats(this.Counts[Right], this.Orders[Right], this.Counts[Left], this.Orders[Left])) Child = Right;
      if (!this.Beats(this.Counts[Child], this.Orders[Child], Count, Order)) break;
      this.Move(Child, I);
      I = Child;
    }
    this.Counts[I] = Count;
    this.Orders[I] = Order;
    this.Keys[I] = Key;
  }

  /** True when entry (CountA, OrderA) must sit above (CountB, OrderB). */
  private Beats(CountA: number, OrderA: number, CountB: number, OrderB: number): boolean {
    return CountA > CountB || (CountA === CountB && OrderA < OrderB);
  }

  private Move(From: number, To: number): void {
    this.Counts[To] = this.Counts[From];
    this.Orders[To] = this.Orders[From];
    this.Keys[To] = this.Keys[From];
  }

  private Grow(): void {
    const Cap = this.Counts.length * GrowthFactor;
    const Counts = new Float64Array(Cap);
    const Orders = new Float64Array(Cap);
    const Keys = new Float64Array(Cap);
    Counts.set(this.Counts);
    Orders.set(this.Orders);
    Keys.set(this.Keys);
    this.Counts = Counts;
    this.Orders = Orders;
    this.Keys = Keys;
  }
}
