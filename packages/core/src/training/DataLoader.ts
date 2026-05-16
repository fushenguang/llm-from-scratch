import * as tf from "@tensorflow/tfjs";

export interface DataBatch {
  x: tf.Tensor2D;
  y: tf.Tensor2D;
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

/**
 * Randomly samples token windows for next-token prediction training.
 */
export class DataLoader {
  private readonly maxStartIndex: number;

  /**
   * @param tokens Encoded dataset tokens.
   * @param batchSize Number of training sequences per batch.
   * @param blockSize Sequence length of each training example.
   */
  public constructor(
    private readonly tokens: Uint16Array,
    private readonly batchSize: number,
    private readonly blockSize: number
  ) {
    assertCondition(tokens instanceof Uint16Array, "DataLoader tokens must be a Uint16Array.");
    assertCondition(Number.isInteger(batchSize), "DataLoader batchSize must be an integer.");
    assertCondition(batchSize > 0, "DataLoader batchSize must be greater than 0.");
    assertCondition(Number.isInteger(blockSize), "DataLoader blockSize must be an integer.");
    assertCondition(blockSize > 0, "DataLoader blockSize must be greater than 0.");
    assertCondition(
      tokens.length >= blockSize + 1,
      "DataLoader dataset must contain at least blockSize + 1 tokens."
    );

    this.maxStartIndex = tokens.length - blockSize - 1;
  }

  /**
   * Samples a random batch of input and target token windows.
   *
   * Callers own the returned tensors and must dispose them when finished.
   *
   * @returns Input and target tensors with shape `[batchSize, blockSize]`.
   */
  public nextBatch(): DataBatch {
    const elementCount = this.batchSize * this.blockSize;
    const xValues = new Int32Array(elementCount);
    const yValues = new Int32Array(elementCount);

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex += 1) {
      const start = Math.floor(Math.random() * (this.maxStartIndex + 1));
      const offset = batchIndex * this.blockSize;
      const xSlice = this.tokens.subarray(start, start + this.blockSize);
      const ySlice = this.tokens.subarray(start + 1, start + this.blockSize + 1);

      xValues.set(xSlice, offset);
      yValues.set(ySlice, offset);
    }

    const x = tf.tensor2d(xValues, [this.batchSize, this.blockSize], "int32"); // shape: [B, T]
    const y = tf.tensor2d(yValues, [this.batchSize, this.blockSize], "int32"); // shape: [B, T]

    return { x, y };
  }
}
