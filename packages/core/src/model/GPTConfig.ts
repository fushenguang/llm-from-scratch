export interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dropout: number;
}

export const CONFIGS = {
  tiny: {
    vocabSize: 65,
    blockSize: 256,
    nLayer: 2,
    nHead: 2,
    nEmbd: 128,
    dropout: 0
  },
  small: {
    vocabSize: 65,
    blockSize: 256,
    nLayer: 4,
    nHead: 4,
    nEmbd: 256,
    dropout: 0
  },
  medium: {
    vocabSize: 65,
    blockSize: 256,
    nLayer: 6,
    nHead: 6,
    nEmbd: 384,
    dropout: 0
  }
} satisfies Record<string, GPTConfig>;
