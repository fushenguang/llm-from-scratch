import type { GPTConfig } from "./GPTConfig.js";
import { Linear } from "./Linear.js";
import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class MLP {
  private readonly cFc: Linear;
  private readonly cProj: Linear;

  public constructor(
    config: Pick<GPTConfig, "nEmbd">,
    private readonly tf: TensorflowLike,
    name = "mlp"
  ) {
    this.cFc = new Linear(config.nEmbd, config.nEmbd * 4, tf, `${name}/c_fc`);
    this.cProj = new Linear(config.nEmbd * 4, config.nEmbd, tf, `${name}/c_proj`);
  }

  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const hidden = this.cFc.apply(x);
      const activated = this.gelu(hidden);

      return this.cProj.apply(activated);
    });
  }

  public trainableVariables(): VariableLike[] {
    return [...this.cFc.trainableVariables(), ...this.cProj.trainableVariables()];
  }

  public dispose(): void {
    this.cFc.dispose();
    this.cProj.dispose();
  }

  private gelu(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const xCubed = this.tf.pow(x, 3);
      const inner = this.tf.add(x, this.tf.mul(0.044715, xCubed));
      const tanhTerm = this.tf.tanh(this.tf.mul(Math.sqrt(2 / Math.PI), inner));
      const cdf = this.tf.mul(0.5, this.tf.add(1, tanhTerm));

      return this.tf.mul(x, cdf);
    });
  }
}
