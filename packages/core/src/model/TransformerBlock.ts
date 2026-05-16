import type { GPTConfig } from "./GPTConfig.js";
import { CausalSelfAttention } from "./CausalSelfAttention.js";
import { LayerNorm } from "./LayerNorm.js";
import { MLP } from "./MLP.js";
import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class TransformerBlock {
  private readonly ln1: LayerNorm;
  private readonly ln2: LayerNorm;
  private readonly attn: CausalSelfAttention;
  private readonly mlp: MLP;

  public constructor(
    config: Pick<GPTConfig, "blockSize" | "nHead" | "nEmbd">,
    private readonly tf: TensorflowLike,
    name = "block"
  ) {
    this.ln1 = new LayerNorm(config.nEmbd, tf, `${name}/ln_1`);
    this.ln2 = new LayerNorm(config.nEmbd, tf, `${name}/ln_2`);
    this.attn = new CausalSelfAttention(config, tf, `${name}/attn`);
    this.mlp = new MLP(config, tf, `${name}/mlp`);
  }

  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const attnOutput = this.attn.apply(this.ln1.apply(x));
      const residualAfterAttention = this.tf.add(x, attnOutput);
      const mlpOutput = this.mlp.apply(this.ln2.apply(residualAfterAttention));

      return this.tf.add(residualAfterAttention, mlpOutput);
    });
  }

  public trainableVariables(): VariableLike[] {
    return [
      ...this.ln1.trainableVariables(),
      ...this.attn.trainableVariables(),
      ...this.ln2.trainableVariables(),
      ...this.mlp.trainableVariables()
    ];
  }

  public dispose(): void {
    this.ln1.dispose();
    this.attn.dispose();
    this.ln2.dispose();
    this.mlp.dispose();
  }
}
