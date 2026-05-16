import { Embedding } from "./Embedding.js";
import type { GPTConfig } from "./GPTConfig.js";
import { LayerNorm } from "./LayerNorm.js";
import { validateGPTConfig } from "./ModelAssertions.js";
import type { TensorLike, TensorflowLike, VariableLike } from "./TensorTypes.js";
import { TransformerBlock } from "./TransformerBlock.js";

export interface GPTForwardResult {
  logits: TensorLike;
  loss: TensorLike | null;
}

export interface GPTModel {
  readonly config: GPTConfig;
  forward(idx: TensorLike, targets?: TensorLike | null): GPTForwardResult;
  trainableVariables(): VariableLike[];
  dispose(): void;
}

export class GPT implements GPTModel {
  private readonly wte: Embedding;
  private readonly wpe: Embedding;
  private readonly blocks: TransformerBlock[];
  private readonly lnF: LayerNorm;

  public constructor(
    public readonly config: GPTConfig,
    private readonly tf: TensorflowLike
  ) {
    validateGPTConfig(config);
    this.wte = new Embedding(config.vocabSize, config.nEmbd, tf, "wte");
    this.wpe = new Embedding(config.blockSize, config.nEmbd, tf, "wpe");
    this.blocks = Array.from({ length: config.nLayer }, (_, index) => {
      return new TransformerBlock(config, tf, `blocks/${index}`);
    });
    this.lnF = new LayerNorm(config.nEmbd, tf, "ln_f");
  }

  public forward(idx: TensorLike, targets: TensorLike | null = null): GPTForwardResult {
    if (idx.shape.length !== 2) {
      throw new Error(`GPT.forward expects idx shape [B, T], received [${idx.shape.join(", ")}].`);
    }

    const batchSize = idx.shape[0];
    const sequenceLength = idx.shape[1];

    if (batchSize === undefined || sequenceLength === undefined) {
      throw new Error("GPT.forward received an invalid idx shape.");
    }

    if (sequenceLength > this.config.blockSize) {
      throw new Error(
        `Sequence length ${sequenceLength} exceeds block size ${this.config.blockSize}.`
      );
    }

    if (targets !== null) {
      if (targets.shape.length !== 2) {
        throw new Error(
          `GPT.forward expects targets shape [B, T], received [${targets.shape.join(", ")}].`
        );
      }
      const targetBatchSize = targets.shape[0];
      const targetSequenceLength = targets.shape[1];

      if (targetBatchSize === undefined || targetSequenceLength === undefined) {
        throw new Error("GPT.forward received an invalid targets shape.");
      }
      if (
        targetBatchSize !== batchSize ||
        targetSequenceLength !== sequenceLength
      ) {
        throw new Error(
          `Targets shape [${targets.shape.join(", ")}] must match idx shape [${idx.shape.join(", ")}].`
        );
      }
    }

    return this.tf.tidy(() => {
      const positions = this.tf.range(0, sequenceLength, 1, "int32"); // shape: [T]
      const tokenEmbeddings = this.wte.apply(idx); // shape: [B, T, nEmbd]
      const positionEmbeddings = this.wpe.apply(positions); // shape: [T, nEmbd]
      let x = this.tf.add(tokenEmbeddings, positionEmbeddings); // shape: [B, T, nEmbd]

      for (const block of this.blocks) {
        x = block.apply(x);
      }

      x = this.lnF.apply(x); // shape: [B, T, nEmbd]
      // TF.js produces an incorrect broadcasted gradient for rank-3 x rank-2 weight tying,
      // so flatten to rank-2 before the shared projection and then restore [B, T, vocabSize].
      const flatHidden = this.tf.reshape(x, [batchSize * sequenceLength, this.config.nEmbd]); // shape: [B*T, nEmbd]
      const flatLogits = this.tf.matMul(flatHidden, this.wte.embeddings, false, true); // shape: [B*T, vocabSize]
      const logits = this.tf.reshape(flatLogits, [
        batchSize,
        sequenceLength,
        this.config.vocabSize
      ]); // shape: [B, T, vocabSize]

      if (targets === null) {
        return { logits, loss: null };
      }

      const flatTargets = this.tf.reshape(targets, [-1]);
      const targetOneHot = this.tf.oneHot(flatTargets, this.config.vocabSize);
      const loss = this.tf.losses.softmaxCrossEntropy(targetOneHot, flatLogits);

      return { logits, loss };
    });
  }

  public trainableVariables(): VariableLike[] {
    return [
      ...this.wte.trainableVariables(),
      ...this.wpe.trainableVariables(),
      ...this.blocks.flatMap((block) => block.trainableVariables()),
      ...this.lnF.trainableVariables()
    ];
  }

  public dispose(): void {
    this.wte.dispose();
    this.wpe.dispose();
    for (const block of this.blocks) {
      block.dispose();
    }
    this.lnF.dispose();
  }
}
