---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-11-module-3-gpt-architecture
---

## Module 5 测试（LR Scheduler）

```typescript
describe('LRScheduler', () => {
  const config = { maxLr: 1e-3, warmupSteps: 100, maxSteps: 1000 };

  test('lr starts at 0 at step 0', () => {
    expect(getLR(0, config)).toBeCloseTo(0);
  });

  test('lr reaches maxLr at end of warmup', () => {
    expect(getLR(100, config)).toBeCloseTo(1e-3);
  });

  test('lr decays after warmup', () => {
    expect(getLR(500, config)).toBeLessThan(1e-3);
    expect(getLR(500, config)).toBeGreaterThan(1e-4);
  });

  test('lr reaches minLr at end of training', () => {
    expect(getLR(1000, config)).toBeCloseTo(1e-4); // minLr = maxLr * 0.1
  });
});
```
