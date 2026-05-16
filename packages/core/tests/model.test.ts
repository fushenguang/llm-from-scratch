import * as tf from "@tensorflow/tfjs";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { CONFIGS, type GPTConfig } from "../src/model/GPTConfig.js";
import { CausalSelfAttention, GPT, type GPTModel } from "../src/model/index.js";

describe("GPT Model", () => {
  const config = CONFIGS.tiny;
  let model: GPTModel;

  beforeEach(() => {
    model = new GPT(config, tf);
  });

  afterEach(() => {
    model.dispose();
  });

  test("parameter count is approximately correct", () => {
    const params = model.trainableVariables().reduce((sum, variable) => sum + variable.size, 0);

    expect(params).toBeGreaterThan(400_000);
    expect(params).toBeLessThan(600_000);
  });

  test("forward pass output shape is correct", () => {
    const batchSize = 2;
    const sequenceLength = 16;
    const idx = tf.randomUniform(
      [batchSize, sequenceLength],
      0,
      config.vocabSize,
      "int32"
    ) as tf.Tensor2D;
    const { logits, loss } = model.forward(idx, null);

    expect(logits.shape).toEqual([batchSize, sequenceLength, config.vocabSize]);
    expect(loss).toBeNull();

    idx.dispose();
    logits.dispose();
  });

  test("forward with targets produces scalar loss", () => {
    const batchSize = 2;
    const sequenceLength = 16;
    const idx = tf.randomUniform(
      [batchSize, sequenceLength],
      0,
      config.vocabSize,
      "int32"
    ) as tf.Tensor2D;
    const targets = tf.randomUniform(
      [batchSize, sequenceLength],
      0,
      config.vocabSize,
      "int32"
    ) as tf.Tensor2D;
    const { logits, loss } = model.forward(idx, targets);

    expect(loss).not.toBeNull();
    expect(loss?.shape).toEqual([]);

    const lossValue = loss?.dataSync()[0];

    expect(lossValue).toBeGreaterThan(3.5);
    expect(lossValue).toBeLessThan(6.0);

    idx.dispose();
    targets.dispose();
    logits.dispose();
    loss?.dispose();
  });

  test("weight tying: lm_head shares weights with wte", () => {
    const variableNames = model.trainableVariables().map((variable) => variable.name);

    expect(variableNames.some((name) => name.includes("lm_head"))).toBe(false);
  });

  test("causal self-attention does not leak future tokens into earlier positions", () => {
    const attn = new CausalSelfAttention({ blockSize: 4, nHead: 2, nEmbd: 4 }, tf, "test_attn");
    const base = tf.tensor3d(
      [
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
        ]
      ],
      [1, 4, 4]
    );
    const changedFuture = tf.tensor3d(
      [
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [50, -50, 25, -25]
        ]
      ],
      [1, 4, 4]
    );
    const outputA = attn.apply(base) as tf.Tensor3D;
    const outputB = attn.apply(changedFuture) as tf.Tensor3D;
    const prefixA = outputA.slice([0, 0, 0], [1, 3, 4]);
    const prefixB = outputB.slice([0, 0, 0], [1, 3, 4]);
    const maxDifference = tf.max(tf.abs(tf.sub(prefixA, prefixB))).dataSync()[0];

    expect(maxDifference).toBeLessThan(1e-5);

    base.dispose();
    changedFuture.dispose();
    outputA.dispose();
    outputB.dispose();
    prefixA.dispose();
    prefixB.dispose();
    attn.dispose();
  });

  test("no tensor leak after forward", () => {
    const before = tf.memory().numTensors;

    tf.tidy(() => {
      const idx = tf.zeros([2, 16], "int32") as tf.Tensor2D;
      const { logits } = model.forward(idx, null);

      return logits;
    }).dispose();

    const after = tf.memory().numTensors;

    expect(after).toBe(before);
  });

  test("invalid GPT configs fail fast", () => {
    const invalidConfig: GPTConfig = {
      ...config,
      nHead: 3
    };

    expect(() => new GPT(invalidConfig, tf)).toThrow(/nEmbd .* divisible by nHead/i);
  });

  test("forward rejects sequences longer than block size", () => {
    const idx = tf.zeros([1, config.blockSize + 1], "int32") as tf.Tensor2D;

    expect(() => model.forward(idx, null)).toThrow(
      new RegExp(`Sequence length ${config.blockSize + 1} exceeds block size ${config.blockSize}`)
    );

    idx.dispose();
  });

  test("forward rejects targets whose shape differs from idx", () => {
    const idx = tf.zeros([2, 16], "int32") as tf.Tensor2D;
    const targets = tf.zeros([2, 15], "int32") as tf.Tensor2D;

    expect(() => model.forward(idx, targets)).toThrow(/must match idx shape/i);

    idx.dispose();
    targets.dispose();
  });
});
