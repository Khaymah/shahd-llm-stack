// Embedding lookup: select rows of Table[V,C] by Ids -> [T,C]. Backward scatters gradient back
// to the looked-up rows with += so repeated ids across positions accumulate correctly.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function EmbeddingLookup(Table: Tensor, Ids: number[]): Tensor {
  const T = Ids.length;
  const C = Table.Cols;
  const Out = new Tensor(T, C, undefined, [Table]);
  const TableData = Table.Data;
  const OutData = Out.Data;
  for (let I = 0; I < T; I++) {
    const Row = Ids[I];
    // Validate before indexing: an out-of-range id would read past the backing Float64Array (undefined
    // -> NaN) and silently poison the whole forward/backward pass. A clear throw makes a tokenizer/vocab
    // off-by-one trivial to trace instead of chasing a downstream NaN.
    if (Row < 0 || Row >= Table.Rows || !Number.isInteger(Row)) {
      throw new Error(`EmbeddingLookup: id ${Row} at position ${I} out of range [0,${Table.Rows}) — likely a tokenizer/vocab mismatch`);
    }
    for (let J = 0; J < C; J++) OutData[I * C + J] = TableData[Row * C + J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const TableGrad = Table.Grad;
      for (let I = 0; I < T; I++) {
        const Row = Ids[I];
        for (let J = 0; J < C; J++) TableGrad[Row * C + J] += OutGrad[I * C + J];
      }
    };
  }
  return Out;
}
