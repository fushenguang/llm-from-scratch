---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-6-module-6-trainer
---

## Module 7: Sampler（文本生成）

**文件**: `packages/core/src/generation/Sampler.ts`

**实现要求**:

```typescript
export interface SamplingConfig {
  maxNewTokens: number;
  /** 温度：< 1 更保守，> 1 更随机，= 1 原始分布 */
  temperature: number;
  /** Top-k 采样：仅从概率最高的 k 个 token 中采样，0 = 禁用 */
  topK: number;
}

/**
 * 自回归生成
 * @param model GPT 模型
 * @param context 初始 token 序列 — shape: [1, T]，T >= 1
 * @param config 采样参数
 * @param tokenizer 用于解码输出
 * @returns 生成的文本（不含初始 context）
 */
export async function generate(
  model: GPTModel,
  context: number[],
  config: SamplingConfig,
  tokenizer: Tokenizer
): Promise<string>
```

**采样逻辑**:

```
for i in range(maxNewTokens):
  1. 截断 context 到最后 blockSize 个 token（防止超出 context window）
  2. forward(context, null) → logits — [1, T, vocabSize]
  3. 取最后一个时间步: logits[:, -1, :] — [1, vocabSize]
  4. 除以 temperature
  5. 若 topK > 0:
       找到第 topK 大的值作为阈值
       将低于阈值的 logits 设为 -1e9
  6. softmax → 概率分布
  7. tf.multinomial(probs, 1) → 采样下一个 token
  8. 追加到 context
```

***
