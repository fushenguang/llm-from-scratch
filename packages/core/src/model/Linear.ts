import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

function createVariable(
  tf: TensorflowLike,
  initialValue: TensorLike,
  name: string
): VariableLike {
  const variable = tf.variable(initialValue, true, name);

  initialValue.dispose();

  return variable;
}

export class Linear {
  public readonly kernel: VariableLike;
  public readonly bias: VariableLike | null;

  public constructor(
    private readonly inFeatures: number,
    private readonly outFeatures: number,
    private readonly tf: TensorflowLike,
    name: string,
    useBias = true
  ) {
    this.kernel = createVariable(
      tf,
      tf.randomNormal([inFeatures, outFeatures], 0, 0.02, "float32"),
      `${name}/kernel`
    );
    this.bias = useBias
      ? createVariable(tf, tf.zeros([outFeatures], "float32"), `${name}/bias`)
      : null;
  }

  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const inputShape = [...x.shape];
      const inputWidth = inputShape[inputShape.length - 1];

      if (inputWidth === undefined) {
        throw new Error("Linear received a tensor without a trailing dimension.");
      }
      if (inputWidth !== this.inFeatures) {
        throw new Error(
          `Linear expected last dimension ${this.inFeatures}, received ${inputWidth}.`
        );
      }

      const flattened = this.tf.reshape(x, [-1, this.inFeatures]);
      let output = this.tf.matMul(flattened, this.kernel);

      if (this.bias !== null) {
        output = this.tf.add(output, this.bias);
      }

      return this.tf.reshape(output, [...inputShape.slice(0, -1), this.outFeatures]);
    });
  }

  public trainableVariables(): VariableLike[] {
    return this.bias === null ? [this.kernel] : [this.kernel, this.bias];
  }

  public dispose(): void {
    this.kernel.dispose();
    this.bias?.dispose();
  }
}
