import type { GPTConfig } from "./GPTConfig.js";

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function assertPositiveInteger(value: number, name: string): void {
  assertCondition(Number.isInteger(value), `${name} must be an integer.`);
  assertCondition(value > 0, `${name} must be greater than 0.`);
}

function assertDropout(value: number): void {
  assertCondition(Number.isFinite(value), "dropout must be finite.");
  assertCondition(value >= 0, "dropout must be non-negative.");
  assertCondition(value < 1, "dropout must be less than 1.");
}

export function validateGPTConfig(config: GPTConfig): void {
  assertPositiveInteger(config.vocabSize, "vocabSize");
  assertPositiveInteger(config.blockSize, "blockSize");
  assertPositiveInteger(config.nLayer, "nLayer");
  assertPositiveInteger(config.nHead, "nHead");
  assertPositiveInteger(config.nEmbd, "nEmbd");
  assertDropout(config.dropout);
  assertCondition(
    config.nEmbd % config.nHead === 0,
    `nEmbd (${config.nEmbd}) must be divisible by nHead (${config.nHead}).`
  );
}

export function validateTransformerWidth(
  config: Pick<GPTConfig, "nHead" | "nEmbd">
): void {
  assertPositiveInteger(config.nHead, "nHead");
  assertPositiveInteger(config.nEmbd, "nEmbd");
  assertCondition(
    config.nEmbd % config.nHead === 0,
    `nEmbd (${config.nEmbd}) must be divisible by nHead (${config.nHead}).`
  );
}
