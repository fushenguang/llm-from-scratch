import type { GPTConfig } from "./GPTConfig.js";
import { Linear } from "./Linear.js";
import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class MLP {
  private readonly cFc: Linear;
  private readonly cProj: Linear;

  /**
   * Creates the feed-forward network used inside a transformer block.
   *
   * @param config Embedding width for the input and output projections.
   * @param tf TensorFlow-like backend injected by the runtime package.
   * @param name Variable name prefix for this module.
   */
  public constructor(
    config: Pick<GPTConfig, "nEmbd">,
    private readonly tf: TensorflowLike,
    name = "mlp"
  ) {
    this.cFc = new Linear(config.nEmbd, config.nEmbd * 4, tf, `${name}/c_fc`);
    this.cProj = new Linear(config.nEmbd * 4, config.nEmbd, tf, `${name}/c_proj`);
  }

  /**
   * Applies the GPT-style feed-forward stack `c_fc -> GELU -> c_proj`.
   *
   * @param x Input tensor with trailing width `nEmbd`.
   * @returns Output tensor with the same leading shape as the input.
   */
  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const hidden = this.cFc.apply(x); // shape: [*, 4C]
      const activated = this.gelu(hidden); // shape: [*, 4C]

      return this.cProj.apply(activated); // shape: [*, C]
    });
  }

  /**
   * Returns the MLP module's trainable variables.
   *
   * @returns Projection variables for the expansion and contraction layers.
   */
  public trainableVariables(): VariableLike[] {
    return [...this.cFc.trainableVariables(), ...this.cProj.trainableVariables()];
  }

  /**
   * Releases tensors owned by this module.
   */
  public dispose(): void {
    this.cFc.dispose();
    this.cProj.dispose();
  }

  private gelu(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const xCubed = this.tf.pow(x, 3); // shape: [*, 4C]
      const inner = this.tf.add(x, this.tf.mul(0.044715, xCubed)); // shape: [*, 4C]
      const tanhTerm = this.tf.tanh(this.tf.mul(Math.sqrt(2 / Math.PI), inner)); // shape: [*, 4C]
      const cdf = this.tf.mul(0.5, this.tf.add(1, tanhTerm)); // shape: [*, 4C]

      return this.tf.mul(x, cdf); // shape: [*, 4C]
    });
  }
}
