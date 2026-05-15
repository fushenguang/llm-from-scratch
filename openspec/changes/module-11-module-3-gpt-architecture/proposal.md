---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-10-module-2-chartokenizer
---

## Module 3 测试（GPT Architecture）

```typescript
// tests/model.test.ts
// 使用 @tensorflow/tfjs（CPU backend）运行，不需要 GPU

describe('GPT Model', () => {
  const config: GPTConfig = CONFIGS.tiny; // vocabSize=65, blockSize=256, nLayer=2, nHead=2, nEmbd=128
  let model: GPTModel;

  beforeEach(() => { model = new GPT(config, tf); });
  afterEach(() => { model.dispose(); });

  test('parameter count is approximately correct', () => {
    const params = model.trainableVariables()
      .reduce((sum, v) => sum + v.size, 0);
    // tiny config 约 0.5M 参数，允许 ±10% 误差
    expect(params).toBeGreaterThan(400_000);
    expect(params).toBeLessThan(600_000);
  });

  test('forward pass output shape is correct', () => {
    const B = 2, T = 16;
    const idx = tf.randomUniform([B, T], 0, config.vocabSize, 'int32') as tf.Tensor2D;
    const { logits, loss } = model.forward(idx, null);
    expect(logits.shape).toEqual([B, T, config.vocabSize]);
    expect(loss).toBeNull();
    idx.dispose(); logits.dispose();
  });

  test('forward with targets produces scalar loss', () => {
    const B = 2, T = 16;
    const idx = tf.randomUniform([B, T], 0, config.vocabSize, 'int32') as tf.Tensor2D;
    const targets = tf.randomUniform([B, T], 0, config.vocabSize, 'int32') as tf.Tensor2D;
    const { logits, loss } = model.forward(idx, targets);
    expect(loss).not.toBeNull();
    expect(loss!.shape).toEqual([]);
    const lossVal = loss!.dataSync()[0];
    // 初始 loss 应接近 -log(1/vocabSize) = log(65) ≈ 4.17
    expect(lossVal).toBeGreaterThan(3.5);
    expect(lossVal).toBeLessThan(6.0);
    [idx, targets, logits, loss!].forEach(t => t.dispose());
  });

  test('weight tying: lm_head shares weights with wte', () => {
    // 通过验证参数数量：有 weight tying 时，参数数应少于无 weight tying
    // 具体实现：检查 trainableVariables() 中没有名为 lm_head/kernel 的变量
    const varNames = model.trainableVariables().map(v => v.name);
    expect(varNames.some(n => n.includes('lm_head'))).toBe(false);
  });

  test('no tensor leak after forward', () => {
    const before = tf.memory().numTensors;
    tf.tidy(() => {
      const idx = tf.zeros([2, 16], 'int32') as tf.Tensor2D;
      const { logits } = model.forward(idx, null);
      return logits; // tidy 会清理除返回值外的所有 tensor
    }).dispose();
    const after = tf.memory().numTensors;
    expect(after).toBe(before);
  });
});
```
