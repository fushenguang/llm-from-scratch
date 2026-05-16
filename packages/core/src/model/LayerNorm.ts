import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class LayerNorm {
  public readonly gamma: VariableLike;
  public readonly beta: VariableLike;

  public constructor(
    public readonly dimension: number,
    private readonly tf: TensorflowLike,
    name: string,
    private readonly epsilon = 1e-5
  ) {
    const gammaInit = tf.ones([dimension], "float32");
    const betaInit = tf.zeros([dimension], "float32");

    this.gamma = tf.variable(gammaInit, true, `${name}/gamma`);
    this.beta = tf.variable(betaInit, true, `${name}/beta`);

    gammaInit.dispose();
    betaInit.dispose();
  }

  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const lastDim = x.shape[x.shape.length - 1];

      if (lastDim === undefined) {
        throw new Error("LayerNorm received a tensor without a trailing dimension.");
      }
      if (lastDim !== this.dimension) {
        throw new Error(
          `LayerNorm expected last dimension ${this.dimension}, received ${lastDim}.`
        );
      }

      const mean = this.tf.mean(x, -1, true);
      const centered = this.tf.sub(x, mean);
      const variance = this.tf.mean(this.tf.square(centered), -1, true);
      const normalized = this.tf.mul(
        centered,
        this.tf.rsqrt(this.tf.add(variance, this.epsilon))
      );

      return this.tf.add(this.tf.mul(normalized, this.gamma), this.beta);
    });
  }

  public trainableVariables(): VariableLike[] {
    return [this.gamma, this.beta];
  }

  public dispose(): void {
    this.gamma.dispose();
    this.beta.dispose();
  }
}
