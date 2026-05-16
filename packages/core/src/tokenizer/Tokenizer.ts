import type { GPTConfig } from "../model/GPTConfig.js";

export interface Tokenizer {
  readonly vocabSize: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  serialize(): TokenizerState;
}

export interface TokenizerState {
  type: "char" | "bpe";
  vocabSize: number;
  vocab?: Record<string, number>;
}

export function assertTokenizerVocabSize(
  tokenizer: Pick<Tokenizer, "vocabSize">,
  config: Pick<GPTConfig, "vocabSize">
): void {
  if (tokenizer.vocabSize !== config.vocabSize) {
    throw new Error(
      `Tokenizer vocab size ${tokenizer.vocabSize} does not match GPTConfig vocab size ${config.vocabSize}.`
    );
  }
}
