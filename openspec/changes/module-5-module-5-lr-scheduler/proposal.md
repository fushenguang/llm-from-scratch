---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-4-module-4-dataloader
---

## Module 5: LR Scheduler

**文件**: `packages/core/src/training/LRScheduler.ts`

**实现要求（Cosine Decay with Warmup）**:

```
if step < warmupSteps:
    lr = maxLr * (step / warmupSteps)
else:
    progress = (step - warmupSteps) / (maxSteps - warmupSteps)
    lr = minLr + 0.5 * (maxLr - minLr) * (1 + cos(π * progress))

其中 minLr = maxLr * 0.1（固定比例）
```

**接口**:

```typescript
export function getLR(step: number, config: {
  maxLr: number;
  warmupSteps: number;
  maxSteps: number;
}): number
```

***
