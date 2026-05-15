---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-3-module-3-gpt-model-architecture
---

## Module 4: DataLoader

**文件**: `packages/core/src/training/DataLoader.ts`

**实现要求**:

* 构造函数接受 `Uint16Array`（编码后的完整数据集）和 `batchSize`, `blockSize`
* `nextBatch()`: 返回 `{ x: tf.Tensor2D, y: tf.Tensor2D }`，shape 均为 `[batchSize, blockSize]`
* 随机采样（不是顺序遍历）：每次随机选 `batchSize` 个起始位置
* `x[i]` = tokens\[start : start+blockSize]，`y[i]` = tokens\[start+1 : start+blockSize+1]
* 调用方负责 `.dispose()` 返回的 tensor

***
