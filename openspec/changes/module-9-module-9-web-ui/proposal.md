---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-8-module-8-trainer-node
---

## Module 9: Web 可视化 UI

**文件**: `packages/web/` 下各文件

#### 9.1 训练仪表盘 (`app/page.tsx`)

必须展示以下内容：

* **实时 Loss 曲线**: 折线图，显示 trainLoss 和 valLoss 随 step 变化，使用 `recharts`
* **训练状态**: 当前 step / total steps 进度条，tokens/sec，当前 LR
* **最新生成样本**: 每次收到 `sample` 事件后更新显示
* **连接状态指示**: SSE 连接状态（连接中 / 已连接 / 断开）

SSE 消费方式:

```typescript
// 使用 EventSource API，连接到 Next.js Route Handler
// Route Handler 代理到 trainer-node 的 SSE 服务（避免 CORS）
const source = new EventSource('/api/train-stream');
source.onmessage = (e) => {
  const event: TrainingEvent = JSON.parse(e.data);
  // dispatch to state
};
```

#### 9.2 文本生成 Playground (`app/playground/page.tsx`)

功能：

* Prompt 输入框
* 滑块控制 `temperature`（0.1 ~ 2.0）和 `topK`（0 ~ 100）
* 生成按钮，调用 `/api/generate`
* 流式显示生成结果（字符逐个出现效果）
* 模型选择：从浏览器加载已保存的 checkpoint 文件（file picker），或连接 trainer-node

#### 9.3 Attention 可视化 (`components/AttentionVisualizer.tsx`)

* 输入一段文本，显示每个 attention head 的注意力矩阵（热力图）
* 使用 `canvas` 绘制，行 = query token，列 = key token，颜色深浅表示 attention weight
* 支持选择 layer 和 head

**注意**: Attention 可视化需要模型在 forward 时暴露中间的 attention weights。需要为 GPT model 添加 `forwardWithAttention()` 方法，返回 `{ logits, attentionWeights: tf.Tensor4D[] }`（每层一个，shape: `[nHead, T, T]`）

***
