import * as tf from "@tensorflow/tfjs";

import { generate, type SamplingConfig } from "../generation/Sampler.js";
import type { GPTModel } from "../model/GPT.js";
import type { TensorLike, VariableLike } from "../model/TensorTypes.js";
import { assertTokenizerVocabSize, type Tokenizer } from "../tokenizer/Tokenizer.js";
import type { TrainingEvent } from "../types/TrainingEvent.js";
import { DataLoader } from "./DataLoader.js";
import { getLR } from "./LRScheduler.js";

export interface TrainerSamplingConfig extends SamplingConfig {
  tokenizer: Tokenizer;
  prompt?: string;
}

export interface TrainerConfig {
  maxSteps: number;
  batchSize: number;
  learningRate: number;
  /** AdamW weight decay. */
  weightDecay: number;
  /** Global gradient norm clipping threshold. */
  gradClipNorm: number;
  /** Warmup steps for the learning rate schedule. */
  warmupSteps: number;
  /** Run validation every N steps. */
  evalInterval: number;
  /** Generate a sample every N steps when sampling is configured. */
  sampleInterval: number;
  /** Emit a checkpoint event every N steps. */
  checkpointInterval: number;
  checkpointDir: string;
  /** Optional sampling configuration required to emit sample events. */
  sampling?: TrainerSamplingConfig;
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function assertPositiveInteger(value: number, name: string): void {
  assertCondition(Number.isInteger(value), `${name} must be an integer.`);
  assertCondition(value > 0, `${name} must be greater than 0.`);
}

function assertNonNegativeNumber(value: number, name: string): void {
  assertCondition(Number.isFinite(value), `${name} must be finite.`);
  assertCondition(value >= 0, `${name} must be non-negative.`);
}

function assertPositiveNumber(value: number, name: string): void {
  assertCondition(Number.isFinite(value), `${name} must be finite.`);
  assertCondition(value > 0, `${name} must be greater than 0.`);
}

function isAssignableVariable(variable: VariableLike): variable is VariableLike & { assign(value: tf.Tensor): void } {
  return typeof Reflect.get(variable, "assign") === "function";
}

function asTensor(tensor: TensorLike): tf.Tensor {
  return tensor as unknown as tf.Tensor;
}

function asScalar(tensor: TensorLike): tf.Scalar {
  return tensor as unknown as tf.Scalar;
}

function asErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function disposeGradientMap(gradients: tf.NamedTensorMap): void {
  for (const gradient of Object.values(gradients)) {
    gradient?.dispose();
  }
}

interface TrainerOptimizer {
  optimizer: tf.Optimizer;
  appliesDecoupledWeightDecay: boolean;
}

function createTrainerOptimizer(learningRate: number, weightDecay: number): TrainerOptimizer {
  const maybeAdamW = Reflect.get(tf.train as object, "adamW");

  if (typeof maybeAdamW === "function") {
    return {
      optimizer: maybeAdamW(learningRate, 0.9, 0.95, 1e-8, weightDecay) as tf.Optimizer,
      appliesDecoupledWeightDecay: true
    };
  }

  return {
    optimizer: tf.train.adam(learningRate, 0.9, 0.95, 1e-8),
    appliesDecoupledWeightDecay: false
  };
}

function applyDecoupledWeightDecay(
  variables: readonly (VariableLike & { assign(value: tf.Tensor): void })[],
  learningRate: number,
  weightDecay: number
): void {
  if (weightDecay === 0) {
    return;
  }

  for (const variable of variables) {
    const decayedValue = tf.tidy(() =>
      tf.sub(asTensor(variable), tf.mul(asTensor(variable), learningRate * weightDecay))
    );
    variable.assign(decayedValue);
    decayedValue.dispose();
  }
}

function clipGradients(gradients: tf.NamedTensorMap, maxNorm: number): tf.NamedTensorMap {
  const entries = Object.entries(gradients);

  assertCondition(entries.length > 0, "Gradient map must not be empty.");

  const totalNormSquared = tf.tidy(() => {
    const squaredNorms = entries.map(([, gradient]) => gradient.square().sum()); // shape: [[]...]
    const sum = tf.addN(squaredNorms); // shape: []
    squaredNorms.forEach((tensor) => tensor.dispose());
    return sum;
  });
  const totalNorm = Math.sqrt(totalNormSquared.dataSync()[0] ?? 0);
  totalNormSquared.dispose();

  if (totalNorm <= maxNorm || totalNorm === 0) {
    return gradients;
  }

  const scale = maxNorm / totalNorm;

  return Object.fromEntries(
    entries.map(([name, gradient]) => [name, gradient.mul(scale)])
  );
}

function evaluateLoss(model: GPTModel, loader: DataLoader): number {
  return tf.tidy(() => {
    const batch = loader.nextBatch();
    const { logits, loss } = model.forward(batch.x, batch.y);

    assertCondition(loss !== null, "Validation loss must not be null.");

    const lossValue = loss.dataSync()[0];
    logits.dispose();

    return lossValue ?? 0;
  });
}

function normalizeTrainableVariables(
  variables: readonly VariableLike[]
): (VariableLike & { assign(value: tf.Tensor): void })[] {
  return variables.map((variable) => {
    assertCondition(
      isAssignableVariable(variable),
      `Trainable variable ${variable.name} does not support assignment.`
    );
    return variable;
  });
}

function setOptimizerLearningRate(optimizer: tf.Optimizer, learningRate: number): void {
  const updated = Reflect.set(optimizer, "learningRate", learningRate);
  assertCondition(updated, "Failed to update optimizer learning rate.");
}

/**
 * Core training loop that streams structured events for each training milestone.
 */
export class Trainer {
  /**
   * Trains a model on tokenized data and yields streaming progress events.
   *
   * @param model GPT model to train.
   * @param trainData Training token IDs.
   * @param valData Validation token IDs.
   * @param config Training configuration.
   * @returns An async stream of training events.
   */
  public async *train(
    model: GPTModel,
    trainData: Uint16Array,
    valData: Uint16Array,
    config: TrainerConfig
  ): AsyncIterable<TrainingEvent> {
    assertPositiveInteger(config.maxSteps, "maxSteps");
    assertPositiveInteger(config.batchSize, "batchSize");
    assertPositiveNumber(config.learningRate, "learningRate");
    assertNonNegativeNumber(config.weightDecay, "weightDecay");
    assertPositiveNumber(config.gradClipNorm, "gradClipNorm");
    assertPositiveInteger(config.evalInterval, "evalInterval");
    assertPositiveInteger(config.sampleInterval, "sampleInterval");
    assertPositiveInteger(config.checkpointInterval, "checkpointInterval");
    assertCondition(config.warmupSteps >= 0, "warmupSteps must be non-negative.");
    assertCondition(Number.isInteger(config.warmupSteps), "warmupSteps must be an integer.");
    assertCondition(
      config.warmupSteps < config.maxSteps,
      "warmupSteps must be less than maxSteps."
    );
    assertCondition(config.checkpointDir.length > 0, "checkpointDir must not be empty.");

    if (config.sampling !== undefined) {
      assertTokenizerVocabSize(config.sampling.tokenizer, model.config);
      if (config.sampling.prompt !== undefined) {
        config.sampling.tokenizer.encode(config.sampling.prompt);
      }
    }

    const trainLoader = new DataLoader(trainData, config.batchSize, model.config.blockSize);
    const valLoader = new DataLoader(valData, config.batchSize, model.config.blockSize);
    const trainableVariables = normalizeTrainableVariables(model.trainableVariables());
    let currentLearningRate = getLR(0, {
      maxLr: config.learningRate,
      warmupSteps: config.warmupSteps,
      maxSteps: config.maxSteps
    });
    const { optimizer, appliesDecoupledWeightDecay } = createTrainerOptimizer(
      currentLearningRate,
      config.weightDecay
    );
    const startedAt = performance.now();
    let latestTrainLoss = 0;
    let latestValLoss = Number.NaN;
    let steadyStateTensorCount: number | null = null;

    try {
      yield { type: "start", config: model.config, totalSteps: config.maxSteps };

      for (let step = 1; step <= config.maxSteps; step += 1) {
        const stepStartedAt = performance.now();
        const loss = tf.tidy(() => {
          const batch = trainLoader.nextBatch();
          const { value: lossTensor, grads } = tf.variableGrads(() => {
            const { logits, loss: batchLoss } = model.forward(batch.x, batch.y);

            logits.dispose();
            assertCondition(batchLoss !== null, "Training loss must not be null.");

            return asScalar(batchLoss);
          });
          const lossValue = lossTensor.dataSync()[0] ?? 0;
          const clippedGrads = clipGradients(grads, config.gradClipNorm);

          if (!appliesDecoupledWeightDecay) {
            applyDecoupledWeightDecay(trainableVariables, currentLearningRate, config.weightDecay);
          }
          optimizer.applyGradients(clippedGrads);

          disposeGradientMap(grads);
          if (clippedGrads !== grads) {
            disposeGradientMap(clippedGrads);
          }
          lossTensor.dispose();

          return lossValue;
        });

        const elapsedSeconds = Math.max((performance.now() - stepStartedAt) / 1_000, 1e-9);
        const tokensPerSec = (config.batchSize * model.config.blockSize) / elapsedSeconds;

        latestTrainLoss = loss;
        yield { type: "step", step, loss, lr: currentLearningRate, tokensPerSec };

        if (step % config.evalInterval === 0) {
          latestValLoss = evaluateLoss(model, valLoader);
          yield {
            type: "eval",
            step,
            trainLoss: latestTrainLoss,
            valLoss: latestValLoss
          };
        }

        if (config.sampling !== undefined && step % config.sampleInterval === 0) {
          const promptTokens =
            config.sampling.prompt !== undefined
              ? config.sampling.tokenizer.encode(config.sampling.prompt)
              : [trainData[0] ?? 0];
          const text = await generate(
            model,
            promptTokens,
            {
              maxNewTokens: config.sampling.maxNewTokens,
              temperature: config.sampling.temperature,
              topK: config.sampling.topK
            },
            config.sampling.tokenizer
          );

          yield { type: "sample", step, text };
        }

        if (step % config.checkpointInterval === 0) {
          yield {
            type: "checkpoint",
            step,
            path: `${config.checkpointDir.replace(/\/$/, "")}/step-${step}.json`
          };
        }

        if (step < config.maxSteps) {
          currentLearningRate = getLR(step, {
            maxLr: config.learningRate,
            warmupSteps: config.warmupSteps,
            maxSteps: config.maxSteps
          });
          setOptimizerLearningRate(optimizer, currentLearningRate);
        }

        const tensorCountAfterStep = tf.memory().numTensors;
        if (steadyStateTensorCount === null) {
          steadyStateTensorCount = tensorCountAfterStep;
        } else {
          assertCondition(
            tensorCountAfterStep === steadyStateTensorCount,
            `Tensor leak detected during training: expected ${steadyStateTensorCount} tensors after cleanup, received ${tensorCountAfterStep}.`
          );
        }
      }

      if (Number.isNaN(latestValLoss)) {
        latestValLoss = evaluateLoss(model, valLoader);
      }

      yield {
        type: "done",
        finalValLoss: latestValLoss,
        totalTime: (performance.now() - startedAt) / 1_000
      };
    } catch (error) {
      yield { type: "error", message: asErrorMessage(error) };
    } finally {
      optimizer.dispose();
    }
  }
}
