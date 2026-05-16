import * as tf from "@tensorflow/tfjs";

import type { GPTModel } from "../model/GPT.js";
import type { TensorLike } from "../model/TensorTypes.js";
import type { Tokenizer } from "../tokenizer/Tokenizer.js";

export interface SamplingConfig {
  maxNewTokens: number;
  /** Temperature: < 1 is more conservative, > 1 is more random. */
  temperature: number;
  /** Top-k sampling: sample only from the top-k logits, 0 disables filtering. */
  topK: number;
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function asTensor3D(tensor: TensorLike): tf.Tensor3D {
  return tensor as unknown as tf.Tensor3D;
}

function sampleFromProbabilities(probabilities: readonly number[]): number {
  const threshold = Math.random();
  let cumulative = 0;

  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index] ?? 0;
    if (threshold <= cumulative) {
      return index;
    }
  }

  return probabilities.length - 1;
}

function buildProbabilityDistribution(
  logits: readonly number[],
  temperature: number,
  topK: number
): number[] {
  const scaledLogits = logits.map((logit) => logit / temperature);

  if (topK > 0 && topK < scaledLogits.length) {
    const threshold =
      [...scaledLogits].sort((left, right) => right - left)[topK - 1] ?? Number.NEGATIVE_INFINITY;
    for (let index = 0; index < scaledLogits.length; index += 1) {
      if ((scaledLogits[index] ?? Number.NEGATIVE_INFINITY) < threshold) {
        scaledLogits[index] = Number.NEGATIVE_INFINITY;
      }
    }
  }

  const maxLogit = Math.max(...scaledLogits);
  const exponentials = scaledLogits.map((logit) =>
    Number.isFinite(logit) ? Math.exp(logit - maxLogit) : 0
  );
  const sum = exponentials.reduce((total, value) => total + value, 0);

  assertCondition(sum > 0, "Sampling probabilities must sum to a positive value.");

  return exponentials.map((value) => value / sum);
}

/**
 * Autoregressively generates text from a model and tokenizer.
 *
 * @param model GPT model used for next-token prediction.
 * @param context Initial token sequence with shape `[1, T]`.
 * @param config Sampling configuration.
 * @param tokenizer Tokenizer used to decode the generated tokens.
 * @returns The generated text, excluding the initial context tokens.
 */
export async function generate(
  model: GPTModel,
  context: number[],
  config: SamplingConfig,
  tokenizer: Tokenizer
): Promise<string> {
  assertCondition(context.length > 0, "Sampling context must contain at least one token.");
  assertCondition(Number.isInteger(config.maxNewTokens), "maxNewTokens must be an integer.");
  assertCondition(config.maxNewTokens > 0, "maxNewTokens must be greater than 0.");
  assertCondition(Number.isFinite(config.temperature), "temperature must be finite.");
  assertCondition(config.temperature > 0, "temperature must be greater than 0.");
  assertCondition(Number.isInteger(config.topK), "topK must be an integer.");
  assertCondition(config.topK >= 0, "topK must be non-negative.");

  const generatedTokens: number[] = [];
  const runningContext = [...context];

  for (let tokenIndex = 0; tokenIndex < config.maxNewTokens; tokenIndex += 1) {
    const nextToken = tf.tidy(() => {
      const modelContext = runningContext.slice(-model.config.blockSize);
      const idx = tf.tensor2d(modelContext, [1, modelContext.length], "int32"); // shape: [1, T]
      const { logits } = model.forward(idx, null);
      const lastLogits = asTensor3D(logits)
        .slice([0, modelContext.length - 1, 0], [1, 1, model.config.vocabSize]) // shape: [1, 1, vocabSize]
        .reshape([model.config.vocabSize]); // shape: [vocabSize]
      const probabilities = buildProbabilityDistribution(
        Array.from(lastLogits.dataSync()),
        config.temperature,
        config.topK
      );

      logits.dispose();

      return sampleFromProbabilities(probabilities);
    });

    runningContext.push(nextToken);
    generatedTokens.push(nextToken);
  }

  return tokenizer.decode(generatedTokens);
}
