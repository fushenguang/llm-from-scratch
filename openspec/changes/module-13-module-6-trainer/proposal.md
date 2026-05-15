---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-12-module-5-lr-scheduler
---

## Module 6 集成测试（Trainer 冒烟测试）

```typescript
// 仅验证训练循环可以正常跑 10 步，不验证 loss 收敛
describe('Trainer smoke test', () => {
  test('runs 10 steps without error', async () => {
    const config = CONFIGS.tiny;
    const tokenizer = new CharTokenizer(shakespeareText.slice(0, 10000));
    const data = new Uint16Array(tokenizer.encode(shakespeareText.slice(0, 10000)));
    const trainData = data.slice(0, Math.floor(data.length * 0.9));
    const valData = data.slice(Math.floor(data.length * 0.9));

    const model = new GPT(config, tf);
    const trainer = new Trainer();
    const trainerConfig: TrainerConfig = {
      maxSteps: 10,
      batchSize: 2,
      learningRate: 1e-3,
      weightDecay: 0.1,
      gradClipNorm: 1.0,
      warmupSteps: 2,
      evalInterval: 5,
      sampleInterval: 10,
      checkpointInterval: 100,
      checkpointDir: '/tmp/test-checkpoints',
    };

    const events: TrainingEvent[] = [];
    for await (const event of trainer.train(model, trainData, valData, trainerConfig)) {
      events.push(event);
    }

    expect(events.filter(e => e.type === 'step')).toHaveLength(10);
    expect(events.filter(e => e.type === 'eval')).toHaveLength(2); // step 5 和 step 10
    expect(events.at(-1)?.type).toBe('done');

    model.dispose();
  }, 60_000); // 给 60 秒超时
});
```

***
