---
schema_version: v1
doc_type: interfaces
last_reviewed: 2026-05-15
---

# Interfaces

## `@llm-from-scratch/core`

### `GPTConfig`

```ts
interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dropout: number;
}
```

### `Tokenizer`

```ts
interface Tokenizer {
  readonly vocabSize: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  serialize(): TokenizerState;
}

interface TokenizerState {
  type: "char" | "bpe";
  vocabSize: number;
  vocab?: Record<string, number>;
}
```

### `getLR`

```ts
interface LRSchedulerConfig {
  maxLr: number;
  warmupSteps: number;
  maxSteps: number;
}

function getLR(step: number, config: LRSchedulerConfig): number;
```

### `TrainingEvent`

```ts
type TrainingEvent =
  | { type: "start"; config: GPTConfig; totalSteps: number }
  | { type: "step"; step: number; loss: number; lr: number; tokensPerSec: number }
  | { type: "eval"; step: number; trainLoss: number; valLoss: number }
  | { type: "sample"; step: number; text: string }
  | { type: "checkpoint"; step: number; path: string }
  | { type: "done"; finalValLoss: number; totalTime: number }
  | { type: "error"; message: string };
```

### `SamplingConfig` / `generate`

```ts
interface SamplingConfig {
  maxNewTokens: number;
  temperature: number;
  topK: number;
}

function generate(
  model: GPTModel,
  context: number[],
  config: SamplingConfig,
  tokenizer: Tokenizer
): Promise<string>;
```

### `Trainer`

```ts
interface TrainerSamplingConfig extends SamplingConfig {
  tokenizer: Tokenizer;
  prompt?: string;
}

interface TrainerConfig {
  maxSteps: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  gradClipNorm: number;
  warmupSteps: number;
  evalInterval: number;
  sampleInterval: number;
  checkpointInterval: number;
  checkpointDir: string;
  sampling?: TrainerSamplingConfig;
}

class Trainer {
  train(
    model: GPTModel,
    trainData: Uint16Array,
    valData: Uint16Array,
    config: TrainerConfig
  ): AsyncIterable<TrainingEvent>;
}
```
