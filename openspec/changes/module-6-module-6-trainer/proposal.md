---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-5-module-5-lr-scheduler
---

## Module 6: Trainer（核心训练循环）

**文件**: `packages/core/src/training/Trainer.ts`

**实现要求**:

```
1. 优化器: tf.train.adamW(lr, beta1=0.9, beta2=0.95, epsilon=1e-8, weightDecay)
   注意: AdamW 在 TF.js 中可能需要手动实现 weight decay（decoupled），
   若 tf.train.adamW 不支持 decoupled weight decay，使用 AdamW 的正确实现：
     θ = θ - lr * weightDecay * θ  （先衰减）
     θ = θ - lr * adam_gradient    （再更新）

2. 训练步（每步必须在 tf.tidy() 中执行）:
   a. 采样 batch: { x, y } = dataLoader.nextBatch()
   b. 使用 tf.variableGrads() 计算梯度
   c. 梯度裁剪（global norm clipping）:
      totalNorm = sqrt(sum(||grad_i||^2 for all i))
      if totalNorm > gradClipNorm:
          scale = gradClipNorm / totalNorm
          scaled_grads = {k: g * scale for k, g in grads}
   d. optimizer.applyGradients(scaledGrads)
   e. 更新 optimizer 的 lr（TF.js optimizer lr 是 variable，直接赋值）
   f. 记录 step 耗时，计算 tokensPerSec = batchSize * blockSize / elapsedSeconds

3. 产出 TrainingEvent（通过 yield）:
   - 每步: yield { type: 'step', step, loss, lr, tokensPerSec }
   - 每 evalInterval 步: 在 val 集上计算 loss（不更新梯度），yield { type: 'eval', ... }
   - 每 sampleInterval 步: 调用 Sampler 生成 200 个字符，yield { type: 'sample', ... }
   - 每 checkpointInterval 步: 保存权重（仅在 trainer-node 中实现），yield { type: 'checkpoint', ... }

4. 内存管理警告:
   - tf.variableGrads 的回调函数内，loss 必须作为返回值，不得提前 dispose
   - 每个 step 结束后，检查 tf.memory().numTensors 不持续增长
```

***
