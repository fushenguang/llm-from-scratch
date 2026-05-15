---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-7-module-7-sampler
---

## Module 8: trainer-node 主程序

**文件**: `packages/trainer-node/src/main.ts`

**实现要求**:

* CLI 入口，使用 `process.argv` 或轻量级 CLI 库（如 `minimist`）解析参数
* 支持参数: `--config tiny|small|medium`，`--steps N`，`--data-dir PATH`，`--checkpoint-dir PATH`，`--port N`（SSE 服务端口，默认 3001）
* 启动流程:
  1. 加载 `@tensorflow/tfjs-node-gpu`，打印 GPU 信息
  2. 读取训练数据，用 `CharTokenizer` 编码，90/10 分割 train/val
  3. 初始化 `GPT` 模型，打印参数量
  4. 启动 SSE HTTP 服务（见 `SseServer.ts`）
  5. 调用 `trainer.train()`，`for await...of` 消费事件，广播给所有 SSE 客户端

**CheckpointManager**:

```typescript
// 保存: 将 model.getWeights() 序列化为 JSON，存入 {checkpointDir}/step-{N}.json
// 加载: 从 JSON 恢复，调用 model.setWeights()
// 最多保留最新 3 个 checkpoint（自动清理旧的）
```

**SseServer**:

```typescript
// 使用 express 实现
// GET /events → text/event-stream，持久连接
// POST /generate { prompt: string, config: SamplingConfig } → application/json，返回生成文本
// GET /status → 返回当前训练状态（当前 step，当前 loss，模型 config）
```

***
