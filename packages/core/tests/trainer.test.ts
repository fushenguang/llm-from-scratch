import * as tf from "@tensorflow/tfjs";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { generate } from "../src/generation/Sampler.js";
import { GPT, type GPTModel } from "../src/model/index.js";
import { CharTokenizer } from "../src/tokenizer/index.js";
import { Trainer, type TrainerConfig } from "../src/training/index.js";
import type { TrainingEvent } from "../src/types/index.js";

function buildTrainingText(): string {
  const alphabet = Array.from({ length: 65 }, (_, index) => String.fromCharCode(32 + index)).join("");
  return alphabet.repeat(240);
}

describe("Trainer", () => {
  const trainingText = buildTrainingText();
  const tokenizer = new CharTokenizer(trainingText);
  const data = new Uint16Array(tokenizer.encode(trainingText));
  const splitIndex = Math.floor(data.length * 0.9);
  const trainData = data.slice(0, splitIndex);
  const valData = data.slice(splitIndex);
  const modelConfig = {
    vocabSize: tokenizer.vocabSize,
    blockSize: 16,
    nLayer: 1,
    nHead: 1,
    nEmbd: 16,
    dropout: 0
  };

  let model: GPTModel;

  beforeEach(() => {
    model = new GPT(modelConfig, tf);
  });

  afterEach(() => {
    model.dispose();
  });

  test("runs a smoke training loop and emits progress events", async () => {
    const trainer = new Trainer();
    const trainerConfig: TrainerConfig = {
      maxSteps: 10,
      batchSize: 2,
      learningRate: 1e-3,
      weightDecay: 0.1,
      gradClipNorm: 1.0,
      warmupSteps: 2,
      evalInterval: 5,
      sampleInterval: 10,
      checkpointInterval: 100,
      checkpointDir: "/tmp/test-checkpoints",
      sampling: {
        tokenizer,
        prompt: trainingText.slice(0, 8),
        maxNewTokens: 16,
        temperature: 1,
        topK: 0
      }
    };
    const events: TrainingEvent[] = [];

    for await (const event of trainer.train(model, trainData, valData, trainerConfig)) {
      events.push(event);
    }

    expect(events[0]).toEqual({
      type: "start",
      config: modelConfig,
      totalSteps: 10
    });
    expect(events.filter((event) => event.type === "step")).toHaveLength(10);
    expect(events.filter((event) => event.type === "eval")).toHaveLength(2);
    expect(events.filter((event) => event.type === "sample")).toHaveLength(1);
    expect(events.at(-1)?.type).toBe("done");
    expect(
      events.find((event): event is Extract<TrainingEvent, { type: "sample" }> => event.type === "sample")
        ?.text.length
    ).toBe(16);
  }, 60_000);

  test("emits checkpoint events and does not leak tensors across training", async () => {
    const trainer = new Trainer();
    const trainerConfig: TrainerConfig = {
      maxSteps: 4,
      batchSize: 2,
      learningRate: 1e-3,
      weightDecay: 0.1,
      gradClipNorm: 1.0,
      warmupSteps: 1,
      evalInterval: 2,
      sampleInterval: 4,
      checkpointInterval: 2,
      checkpointDir: "/tmp/test-checkpoints/",
      sampling: {
        tokenizer,
        prompt: trainingText.slice(0, 8),
        maxNewTokens: 8,
        temperature: 1,
        topK: 0
      }
    };
    const before = tf.memory().numTensors;
    const events: TrainingEvent[] = [];

    for await (const event of trainer.train(model, trainData, valData, trainerConfig)) {
      events.push(event);
    }

    const checkpoints = events.filter(
      (event): event is Extract<TrainingEvent, { type: "checkpoint" }> => event.type === "checkpoint"
    );

    expect(checkpoints).toEqual([
      { type: "checkpoint", step: 2, path: "/tmp/test-checkpoints/step-2.json" },
      { type: "checkpoint", step: 4, path: "/tmp/test-checkpoints/step-4.json" }
    ]);
    expect(tf.memory().numTensors).toBe(before);
  }, 60_000);

  test("generate returns the requested number of decoded characters", async () => {
    const text = await generate(
      model,
      tokenizer.encode(trainingText.slice(0, 4)),
      {
        maxNewTokens: 12,
        temperature: 1,
        topK: 5
      },
      tokenizer
    );

    expect(text).toHaveLength(12);
  });
});
