import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class Embedding {
  public readonly embeddings: VariableLike;

  public constructor(
    public readonly numEmbeddings: number,
    public readonly embeddingDim: number,
    private readonly tf: TensorflowLike,
    name: string
  ) {
    const initialEmbeddings = tf.randomNormal(
      [numEmbeddings, embeddingDim],
      0,
      0.02,
      "float32"
    );

    this.embeddings = tf.variable(initialEmbeddings, true, `${name}/embeddings`);
    initialEmbeddings.dispose();
  }

  public apply(indices: TensorLike): TensorLike {
    return this.tf.gather(this.embeddings, indices, 0);
  }

  public trainableVariables(): VariableLike[] {
    return [this.embeddings];
  }

  public dispose(): void {
    this.embeddings.dispose();
  }
}
