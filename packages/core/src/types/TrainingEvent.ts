import type { GPTConfig } from "../model/GPTConfig.js";

/**
 * Serializable training events shared across training and UI layers.
 */
export type TrainingEvent =
  | { type: "start"; config: GPTConfig; totalSteps: number }
  | { type: "step"; step: number; loss: number; lr: number; tokensPerSec: number }
  | { type: "eval"; step: number; trainLoss: number; valLoss: number }
  | { type: "sample"; step: number; text: string }
  | { type: "checkpoint"; step: number; path: string }
  | { type: "done"; finalValLoss: number; totalTime: number }
  | { type: "error"; message: string };
