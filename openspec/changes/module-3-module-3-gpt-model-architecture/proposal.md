---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-2-module-2-chartokenizer
---

## Module 3: GPT Model Architecture

**文件**: `packages/core/src/model/` 下各文件

**重要**: 所有 tensor 操作使用 `tf` 命名空间（由上层 package 注入），core 包**不直接 import** `@tensorflow/tfjs-node-gpu` 或 `@tensorflow/tfjs`。通过依赖注入传入 `tf` 对象。

> 这是架构的关键设计：core 包是环境无关的，trainer-node 注入 `tfjs-node-gpu`，web 注入 `tfjs` + WebGPU backend。

**3.1 CausalSelfAttention**

```
输入: x — shape: [B, T, C]，其中 C = nEmbd
输出: shape: [B, T, C]

实现步骤（每步标注 shape）：
1. c_attn: Linear(C → 3C)，得到 qkv — [B, T, 3C]
2. split qkv → q, k, v，各 [B, T, C]
3. reshape 为多头: [B, T, nHead, headDim] → transpose → [B, nHead, T, headDim]
4. 计算 attn scores: tf.matMul(q, k, false, true) / sqrt(headDim) — [B, nHead, T, T]
5. 因果掩码: 上三角设为 -1e9（不得使用 -Infinity，会导致 NaN）
6. softmax(scores, axis=-1) — [B, nHead, T, T]
7. tf.matMul(weights, v) — [B, nHead, T, headDim]
8. transpose + reshape → [B, T, C]
9. c_proj: Linear(C → C)
```

**3.2 MLP**

```
输入: x — [B, T, C]
实现: c_fc(C→4C) → GELU → c_proj(4C→C)
GELU 近似公式: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
输出: [B, T, C]
```

**3.3 TransformerBlock**

```
Pre-norm 结构（不是 Post-norm）:
x = x + attn(layerNorm1(x))   // residual connection
x = x + mlp(layerNorm2(x))    // residual connection
```

**3.4 GPT**

```
组件:
- wte: Embedding(vocabSize, nEmbd)       — token embedding
- wpe: Embedding(blockSize, nEmbd)       — position embedding
- blocks: TransformerBlock × nLayer
- ln_f: LayerNorm(nEmbd)                 — final layer norm
- lm_head: Linear(nEmbd → vocabSize, bias=false)

权重绑定（weight tying）:
lm_head 的权重矩阵 = wte 的权重矩阵（转置使用）
在 TF.js 中：lm_head 不创建独立 Variable，
forward 时用 tf.matMul(x, wte.embeddings, false, true) 替代 lm_head

forward 实现:
1. pos = tf.range(0, T)                             // [T]
2. tok_emb = wte.apply(idx)                         // [B, T, nEmbd]
3. pos_emb = wpe.apply(pos)                         // [T, nEmbd]，广播加到 tok_emb
4. x = tok_emb + pos_emb                            // [B, T, nEmbd]
5. for each block: x = block.apply(x)
6. x = ln_f.apply(x)                                // [B, T, nEmbd]
7. logits = tf.matMul(x, wte.embeddings, false, true) // [B, T, vocabSize]（weight tying）
8. if targets != null:
     loss = tf.losses.softmaxCrossEntropy(
       tf.oneHot(targets.flatten(), vocabSize),
       logits.reshape([-1, vocabSize])
     )
```

**验收**: 见 §5 Module 3 测试

***
