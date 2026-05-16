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

  /**
   * Creates a pre-norm transformer block with residual attention and MLP paths.
   *
   * @param config Transformer dimensions and context length.
   * @param tf TensorFlow-like backend injected by the runtime package.
   * @param name Variable name prefix for this block.
   */
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

  /**
   * Applies the block's pre-norm residual attention and MLP layers.
   *
   * @param x Input tensor with shape `[B, T, C]`.
   * @returns Output tensor with shape `[B, T, C]`.
   */
  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      const attnOutput = this.attn.apply(this.ln1.apply(x)); // shape: [B, T, C]
      const residualAfterAttention = this.tf.add(x, attnOutput); // shape: [B, T, C]
      const mlpOutput = this.mlp.apply(this.ln2.apply(residualAfterAttention)); // shape: [B, T, C]

      return this.tf.add(residualAfterAttention, mlpOutput); // shape: [B, T, C]
    });
  }

  /**
   * Returns the block's trainable variables.
   *
   * @returns Layer norm, attention, and MLP variables.
   */
  public trainableVariables(): VariableLike[] {
    return [
      ...this.ln1.trainableVariables(),
      ...this.attn.trainableVariables(),
      ...this.ln2.trainableVariables(),
      ...this.mlp.trainableVariables()
    ];
  }

  /**
   * Releases tensors owned by this block.
   */
  public dispose(): void {
    this.ln1.dispose();
    this.attn.dispose();
    this.ln2.dispose();
    this.mlp.dispose();
  }
}
