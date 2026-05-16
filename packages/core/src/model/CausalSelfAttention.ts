import type { GPTConfig } from "./GPTConfig.js";
import { Linear } from "./Linear.js";
import { validateTransformerWidth } from "./ModelAssertions.js";
import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";

export class CausalSelfAttention {
  private readonly nHead: number;
  private readonly nEmbd: number;
  private readonly headDim: number;
  private readonly cAttn: Linear;
  private readonly cProj: Linear;
  private readonly causalMask: TensorLike;

  public constructor(
    config: Pick<GPTConfig, "blockSize" | "nHead" | "nEmbd">,
    private readonly tf: TensorflowLike,
    name = "attn"
  ) {
    validateTransformerWidth(config);

    this.nHead = config.nHead;
    this.nEmbd = config.nEmbd;
    this.headDim = config.nEmbd / config.nHead;
    this.cAttn = new Linear(this.nEmbd, this.nEmbd * 3, tf, `${name}/c_attn`);
    this.cProj = new Linear(this.nEmbd, this.nEmbd, tf, `${name}/c_proj`);
    this.causalMask = tf.tidy(() => {
      const lowerTriangle = tf.linalg.bandPart(
        tf.ones([config.blockSize, config.blockSize], "float32"),
        -1,
        0
      );
      const upperTriangle = tf.sub(1, lowerTriangle);
      const mask2D = tf.mul(upperTriangle, -1e9);

      return tf.reshape(mask2D, [1, 1, config.blockSize, config.blockSize]);
    });
  }

  public apply(x: TensorLike): TensorLike {
    return this.tf.tidy(() => {
      if (x.shape.length !== 3) {
        throw new Error(
          `CausalSelfAttention expects input shape [B, T, C], received [${x.shape.join(", ")}].`
        );
      }

      const batchSize = x.shape[0];
      const sequenceLength = x.shape[1];
      const channels = x.shape[2];
      const blockSize = this.causalMask.shape[2];

      if (
        batchSize === undefined ||
        sequenceLength === undefined ||
        channels === undefined ||
        blockSize === undefined
      ) {
        throw new Error("CausalSelfAttention received an invalid tensor shape.");
      }

      if (sequenceLength > blockSize) {
        throw new Error(
          `Sequence length ${sequenceLength} exceeds block size ${blockSize}.`
        );
      }
      if (channels !== this.nEmbd) {
        throw new Error(
          `CausalSelfAttention expected embedding width ${this.nEmbd}, received ${channels}.`
        );
      }

      const qkv = this.cAttn.apply(x); // shape: [B, T, 3C]
      const qkvSplit = this.tf.split(qkv, 3, -1);

      if (qkvSplit.length !== 3) {
        throw new Error(`Expected qkv split into 3 tensors, received ${qkvSplit.length}.`);
      }

      const q = qkvSplit[0];
      const k = qkvSplit[1];
      const v = qkvSplit[2];

      if (q === undefined || k === undefined || v === undefined) {
        throw new Error("Failed to split qkv tensor into query, key, and value tensors.");
      }

      const qHeads = this.tf.transpose(
        this.tf.reshape(q, [batchSize, sequenceLength, this.nHead, this.headDim]),
        [0, 2, 1, 3]
      ); // shape: [B, nHead, T, headDim]
      const kHeads = this.tf.transpose(
        this.tf.reshape(k, [batchSize, sequenceLength, this.nHead, this.headDim]),
        [0, 2, 1, 3]
      ); // shape: [B, nHead, T, headDim]
      const vHeads = this.tf.transpose(
        this.tf.reshape(v, [batchSize, sequenceLength, this.nHead, this.headDim]),
        [0, 2, 1, 3]
      ); // shape: [B, nHead, T, headDim]
      const scaledScores = this.tf.mul(
        this.tf.matMul(qHeads, kHeads, false, true),
        1 / Math.sqrt(this.headDim)
      ); // shape: [B, nHead, T, T]
      const mask = this.tf.slice(
        this.causalMask,
        [0, 0, 0, 0],
        [1, 1, sequenceLength, sequenceLength]
      ); // shape: [1, 1, T, T]
      const maskedScores = this.tf.add(scaledScores, mask); // shape: [B, nHead, T, T]
      const weights = this.tf.softmax(maskedScores, -1); // shape: [B, nHead, T, T]
      const attended = this.tf.matMul(weights, vHeads); // shape: [B, nHead, T, headDim]
      const merged = this.tf.reshape(
        this.tf.transpose(attended, [0, 2, 1, 3]),
        [batchSize, sequenceLength, channels]
      ); // shape: [B, T, C]

      return this.cProj.apply(merged);
    });
  }

  public trainableVariables(): VariableLike[] {
    return [...this.cAttn.trainableVariables(), ...this.cProj.trainableVariables()];
  }

  public dispose(): void {
    this.cAttn.dispose();
    this.cProj.dispose();
    this.causalMask.dispose();
  }
}
