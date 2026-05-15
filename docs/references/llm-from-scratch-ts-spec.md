---
doc_type: change-request-greenfield
schema_version: v1
---
# LLM From Scratch — TypeScript Implementation Spec
**面向 AI Coding Agent 的完备指令文档**

> **版本**: v1.0  
> **目标读者**: GitHub Copilot / Claude Sonnet 4.6 等 AI Coding Agent  
> **参考原项目**: https://github.com/angelos-p/llm-from-scratch（Python/PyTorch）  
> **实现语言**: TypeScript (strict mode)  
> **训练环境**: Ubuntu 24.x + NVIDIA RTX 3060 (CUDA)  
> **三大用途**: AI 教育自学 / Web 可视化演示 / 未来应用于生产（用户专属小模型训练）

---

## 0. 阅读本文档的方式

本文档是**规格说明（Spec）**，不是实现教程。AI Agent 应当：

1. **先读完整个文档**，理解全局依赖关系，再开始写任何代码
2. 严格按照 §4 的**模块实现顺序**逐步推进，不得跳跃
3. 每个模块完成后，执行 §5 中对应的**验收测试命令**，通过后再进入下一个
4. 遇到类型定义不确定的情况，**优先参考 §3 的接口定义**，不得自行发明
5. **不得使用任何 Python 风格的动态类型处理**，所有 tensor shape 必须在注释中明确标注

---

## 1. 项目全局约束

### 1.1 技术栈锁定

| 层次 | 技术 | 版本约束 | 说明 |
|---|---|---|---|
| 语言 | TypeScript | `>=5.4` | strict mode，无 `any` |
| 张量运算 | `@tensorflow/tfjs-node-gpu` | `>=4.20` | Node.js 训练，CUDA 后端 |
| Web 可视化 | `@tensorflow/tfjs` + WebGPU backend | `>=4.20` | 浏览器演示 |
| Web 框架 | Next.js 14+ (App Router) | `>=14.2` | 可视化 UI |
| UI 组件 | shadcn/ui + Tailwind CSS v3 | latest | — |
| 实时通信 | Server-Sent Events (SSE) | — | 训练进度推流 |
| 数据格式 | JSON + Float32Array binary | — | 模型权重序列化 |
| 包管理 | pnpm | `>=9` | monorepo workspace |
| 测试 | Vitest | `>=1.6` | 单元测试 |
| 代码规范 | ESLint + Prettier | — | 项目根配置 |

### 1.2 禁止事项（AI Agent 必须遵守）

- ❌ 禁止使用 `any` 类型，用 `unknown` + type guard 替代
- ❌ 禁止在模型代码中使用同步文件 IO（`fs.readFileSync`），一律 async
- ❌ 禁止在浏览器包中引入 `@tensorflow/tfjs-node-gpu`
- ❌ 禁止硬编码训练超参数，所有参数必须通过 `GPTConfig` 传入
- ❌ 禁止在 tensor 操作后不调用 `.dispose()`（内存泄漏）
- ❌ 禁止跳过 §5 的验收测试直接进入下一模块

### 1.3 必须事项

- ✅ 每个 tensor 操作后必须在注释中标注 shape，格式：`// shape: [B, T, C]`
- ✅ 每个公共函数必须有 JSDoc，包含参数说明和返回值
- ✅ 所有数值超参数必须有合理范围的运行时断言（`assert`）
- ✅ 训练循环中必须使用 `tf.tidy()` 管理中间 tensor 生命周期

---

## 2. 仓库结构

```
llm-from-scratch-ts/
├── packages/
│   ├── core/                        # 纯 TS，与运行环境无关
│   │   ├── src/
│   │   │   ├── tokenizer/
│   │   │   │   ├── CharTokenizer.ts
│   │   │   │   └── index.ts
│   │   │   ├── model/
│   │   │   │   ├── GPTConfig.ts
│   │   │   │   ├── CausalSelfAttention.ts
│   │   │   │   ├── MLP.ts
│   │   │   │   ├── TransformerBlock.ts
│   │   │   │   ├── GPT.ts
│   │   │   │   └── index.ts
│   │   │   ├── training/
│   │   │   │   ├── DataLoader.ts
│   │   │   │   ├── Trainer.ts
│   │   │   │   ├── LRScheduler.ts
│   │   │   │   └── index.ts
│   │   │   ├── generation/
│   │   │   │   ├── Sampler.ts
│   │   │   │   └── index.ts
│   │   │   └── types/
│   │   │       ├── TrainingEvent.ts  # SSE 事件类型定义（共享）
│   │   │       └── index.ts
│   │   ├── tests/
│   │   └── package.json
│   │
│   ├── trainer-node/                # Node.js 训练进程（CUDA）
│   │   ├── src/
│   │   │   ├── main.ts              # 入口：CLI 训练脚本
│   │   │   ├── SseServer.ts         # HTTP SSE 服务，推送训练事件
│   │   │   └── CheckpointManager.ts
│   │   ├── data/
│   │   │   └── shakespeare.txt
│   │   └── package.json
│   │
│   └── web/                         # Next.js 可视化 UI
│       ├── app/
│       │   ├── page.tsx             # 主页：训练仪表盘
│       │   ├── playground/
│       │   │   └── page.tsx         # 文本生成 Playground
│       │   └── api/
│       │       └── train-stream/
│       │           └── route.ts     # Next.js Route Handler → 代理 SSE
│       ├── components/
│       │   ├── TrainingDashboard.tsx
│       │   ├── LossChart.tsx
│       │   ├── AttentionVisualizer.tsx
│       │   └── TextPlayground.tsx
│       └── package.json
│
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── vitest.config.ts
└── README.md
```

---

## 3. 核心类型与接口定义

> AI Agent 实现时必须严格遵守以下接口，不得修改签名。

### 3.1 GPTConfig

```typescript
// packages/core/src/model/GPTConfig.ts

export interface GPTConfig {
  /** 词表大小。字符级：65（Shakespeare），BPE：50257 */
  vocabSize: number;
  /** 最大序列长度（context window） */
  blockSize: number;
  /** Transformer Block 层数 */
  nLayer: number;
  /** 多头注意力头数，必须整除 nEmbd */
  nHead: number;
  /** 嵌入维度 */
  nEmbd: number;
  /** Dropout 率，训练时用，推理时设为 0 */
  dropout: number;
}

/** 预设配置 */
export const CONFIGS = {
  tiny:   { vocabSize: 65, blockSize: 256, nLayer: 2, nHead: 2, nEmbd: 128, dropout: 0.0 },
  small:  { vocabSize: 65, blockSize: 256, nLayer: 4, nHead: 4, nEmbd: 256, dropout: 0.0 },
  medium: { vocabSize: 65, blockSize: 256, nLayer: 6, nHead: 6, nEmbd: 384, dropout: 0.0 },
} satisfies Record<string, GPTConfig>;
```

### 3.2 Tokenizer 接口

```typescript
// packages/core/src/tokenizer/index.ts

export interface Tokenizer {
  readonly vocabSize: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  /** 序列化为 JSON，用于保存至 checkpoint */
  serialize(): TokenizerState;
  /** 从序列化状态恢复 */
}

export interface TokenizerState {
  type: 'char' | 'bpe';
  vocabSize: number;
  /** char tokenizer: char→index 映射 */
  vocab?: Record<string, number>;
}
```

### 3.3 训练事件（SSE 共享类型）

```typescript
// packages/core/src/types/TrainingEvent.ts
// 这些类型在 trainer-node 和 web 之间通过 SSE 共享，必须可 JSON 序列化

export type TrainingEvent =
  | { type: 'start';    config: GPTConfig; totalSteps: number }
  | { type: 'step';     step: number; loss: number; lr: number; tokensPerSec: number }
  | { type: 'eval';     step: number; trainLoss: number; valLoss: number }
  | { type: 'sample';   step: number; text: string }
  | { type: 'checkpoint'; step: number; path: string }
  | { type: 'done';     finalValLoss: number; totalTime: number }
  | { type: 'error';    message: string };
```

### 3.4 GPT Model 接口

```typescript
// packages/core/src/model/GPT.ts

import * as tf from '@tensorflow/tfjs';

export interface GPTModel {
  readonly config: GPTConfig;
  /** 
   * 前向传播
   * @param idx  token IDs — shape: [B, T]
   * @param targets  下一个 token 的标签 — shape: [B, T]，推理时为 null
   * @returns { logits: shape [B, T, vocabSize], loss: scalar | null }
   */
  forward(idx: tf.Tensor2D, targets: tf.Tensor2D | null): { logits: tf.Tensor3D; loss: tf.Scalar | null };
  
  /** 返回所有可训练变量，供 optimizer 使用 */
  trainableVariables(): tf.Variable[];
  
  /** 导出权重为可序列化格式 */
  getWeights(): tf.NamedTensorMap;
  
  /** 从权重恢复模型 */
  setWeights(weights: tf.NamedTensorMap): void;
  
  /** 释放所有 tensor 内存 */
  dispose(): void;
}
```

### 3.5 Trainer 接口

```typescript
// packages/core/src/training/Trainer.ts

export interface TrainerConfig {
  maxSteps: number;
  batchSize: number;
  learningRate: number;
  /** AdamW weight decay */
  weightDecay: number;
  /** 梯度裁剪阈值 */
  gradClipNorm: number;
  /** warmup steps for LR schedule */
  warmupSteps: number;
  /** 每隔多少步评估一次 val loss */
  evalInterval: number;
  /** 每隔多少步采样一次生成文本 */
  sampleInterval: number;
  /** 每隔多少步保存一次 checkpoint */
  checkpointInterval: number;
  checkpointDir: string;
}

export interface Trainer {
  /** 
   * 开始训练，通过 AsyncIterable 流式产出训练事件
   * 调用方用 for await...of 消费事件
   */
  train(
    model: GPTModel,
    trainData: Uint16Array,
    valData: Uint16Array,
    config: TrainerConfig
  ): AsyncIterable<TrainingEvent>;
}
```

---

## 4. 模块实现规格（按顺序）

### Module 1: 项目脚手架与 Monorepo 配置

**目标**: 建立可运行的空项目骨架

**任务清单**:
1. 在项目根创建 `pnpm-workspace.yaml`，声明 `packages/*` 为 workspace
2. 创建 `tsconfig.base.json`（见下方配置）
3. 创建三个 package 的 `package.json`，核心依赖如下：
   - `packages/core`: 无运行时依赖，仅 devDependencies（`@tensorflow/tfjs` 用于类型，不在 core 中 import 实现）
   - `packages/trainer-node`: `@tensorflow/tfjs-node-gpu`, `express` (SSE server)
   - `packages/web`: `next`, `@tensorflow/tfjs`, `@tensorflow/tfjs-backend-webgpu`
4. 创建 `vitest.config.ts`，配置测试根目录

**`tsconfig.base.json` 内容**:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "lib": ["ES2022"],
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist"
  }
}
```

**验收**: `pnpm install` 无报错，`pnpm -r build` 可执行（即使输出为空）

---

### Module 2: CharTokenizer

**文件**: `packages/core/src/tokenizer/CharTokenizer.ts`

**实现要求**:
- 构造函数接受训练文本 `string`，自动构建 char→index 和 index→char 双向映射
- `vocabSize` = 唯一字符数量
- `encode(text)`: 返回 `number[]`，遇到未知字符抛出 `Error`，不得静默跳过
- `decode(tokens)`: 返回 `string`
- `serialize()` / 静态方法 `CharTokenizer.fromState(state)` 实现双向转换

**关键实现细节**:
```
字符排序必须确定性（sort()），确保同样的训练文本每次产生相同的 vocabSize 和映射
vocabSize 必须与 GPTConfig.vocabSize 一致，训练开始前必须断言
```

**验收**: 见 §5 Module 2 测试

---

### Module 3: GPT Model Architecture

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

---

### Module 4: DataLoader

**文件**: `packages/core/src/training/DataLoader.ts`

**实现要求**:
- 构造函数接受 `Uint16Array`（编码后的完整数据集）和 `batchSize`, `blockSize`
- `nextBatch()`: 返回 `{ x: tf.Tensor2D, y: tf.Tensor2D }`，shape 均为 `[batchSize, blockSize]`
- 随机采样（不是顺序遍历）：每次随机选 `batchSize` 个起始位置
- `x[i]` = tokens[start : start+blockSize]，`y[i]` = tokens[start+1 : start+blockSize+1]
- 调用方负责 `.dispose()` 返回的 tensor

---

### Module 5: LR Scheduler

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

---

### Module 6: Trainer（核心训练循环）

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

---

### Module 7: Sampler（文本生成）

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

---

### Module 8: trainer-node 主程序

**文件**: `packages/trainer-node/src/main.ts`

**实现要求**:
- CLI 入口，使用 `process.argv` 或轻量级 CLI 库（如 `minimist`）解析参数
- 支持参数: `--config tiny|small|medium`，`--steps N`，`--data-dir PATH`，`--checkpoint-dir PATH`，`--port N`（SSE 服务端口，默认 3001）
- 启动流程:
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

---

### Module 9: Web 可视化 UI

**文件**: `packages/web/` 下各文件

#### 9.1 训练仪表盘 (`app/page.tsx`)

必须展示以下内容：
- **实时 Loss 曲线**: 折线图，显示 trainLoss 和 valLoss 随 step 变化，使用 `recharts`
- **训练状态**: 当前 step / total steps 进度条，tokens/sec，当前 LR
- **最新生成样本**: 每次收到 `sample` 事件后更新显示
- **连接状态指示**: SSE 连接状态（连接中 / 已连接 / 断开）

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
- Prompt 输入框
- 滑块控制 `temperature`（0.1 ~ 2.0）和 `topK`（0 ~ 100）
- 生成按钮，调用 `/api/generate`
- 流式显示生成结果（字符逐个出现效果）
- 模型选择：从浏览器加载已保存的 checkpoint 文件（file picker），或连接 trainer-node

#### 9.3 Attention 可视化 (`components/AttentionVisualizer.tsx`)

- 输入一段文本，显示每个 attention head 的注意力矩阵（热力图）
- 使用 `canvas` 绘制，行 = query token，列 = key token，颜色深浅表示 attention weight
- 支持选择 layer 和 head

**注意**: Attention 可视化需要模型在 forward 时暴露中间的 attention weights。需要为 GPT model 添加 `forwardWithAttention()` 方法，返回 `{ logits, attentionWeights: tf.Tensor4D[] }`（每层一个，shape: `[nHead, T, T]`）

---

## 5. 验收测试规格

> AI Agent 每完成一个 Module，必须能通过以下对应测试。测试文件放在 `packages/core/tests/` 下。

### Module 2 测试（CharTokenizer）

```typescript
// tests/tokenizer.test.ts
describe('CharTokenizer', () => {
  const text = 'hello world';
  const tokenizer = new CharTokenizer(text);

  test('encode/decode roundtrip', () => {
    const encoded = tokenizer.encode(text);
    expect(tokenizer.decode(encoded)).toBe(text);
  });

  test('vocab is deterministic', () => {
    const t2 = new CharTokenizer(text);
    expect(t2.vocabSize).toBe(tokenizer.vocabSize);
    expect(t2.encode('hello')).toEqual(tokenizer.encode('hello'));
  });

  test('unknown char throws', () => {
    expect(() => tokenizer.encode('你好')).toThrow();
  });

  test('serialize/deserialize', () => {
    const state = tokenizer.serialize();
    const restored = CharTokenizer.fromState(state);
    expect(restored.encode('hello')).toEqual(tokenizer.encode('hello'));
  });
});
```

### Module 3 测试（GPT Architecture）

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

### Module 5 测试（LR Scheduler）

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

### Module 6 集成测试（Trainer 冒烟测试）

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

---

## 6. 环境配置（Ubuntu 24.x + RTX 3060）

### 6.1 前置依赖

```bash
# CUDA 12.x + cuDNN 8.x（TF.js-node-gpu 要求）
# 验证 CUDA:
nvidia-smi
nvcc --version

# Node.js 20 LTS（通过 nvm）
nvm install 20
nvm use 20

# pnpm
npm install -g pnpm@9
```

### 6.2 TF.js GPU 验证脚本

在开始任何模型实现前，先运行以下脚本验证 GPU 可用：

```typescript
// packages/trainer-node/src/verify-gpu.ts
import * as tf from '@tensorflow/tfjs-node-gpu';

async function main() {
  console.log('TF.js version:', tf.version.tfjs);
  console.log('Backend:', tf.getBackend());
  
  // 创建一个大矩阵相乘，验证 GPU 执行
  const a = tf.randomNormal([1000, 1000]);
  const b = tf.randomNormal([1000, 1000]);
  const start = Date.now();
  const c = tf.matMul(a, b);
  await c.data(); // 等待 GPU 计算完成
  console.log(`GPU matmul 1000x1000 took ${Date.now() - start}ms`);
  console.log('GPU memory:', tf.memory());
  [a, b, c].forEach(t => t.dispose());
}

main().catch(console.error);
```

**预期输出**: Backend 为 `tensorflow`（不是 `cpu`），矩阵乘法耗时 < 100ms

### 6.3 训练启动命令

```bash
# 完整训练（medium config，约 45 分钟）
cd packages/trainer-node
npx ts-node src/main.ts --config medium --steps 5000 --port 3001

# 快速冒烟测试（tiny config，约 5 分钟）
npx ts-node src/main.ts --config tiny --steps 200 --port 3001
```

---

## 7. 架构决策记录（ADR）

> 以下是关键的架构决策，AI Agent 实现时不得推翻，如需讨论应在 PR 中提出。

### ADR-001: Core 包不依赖具体 TF.js 实现

**决策**: `packages/core` 不 import `@tensorflow/tfjs` 或 `@tensorflow/tfjs-node-gpu`。`tf` 对象通过构造函数参数注入。

**原因**: 同一份模型代码需要在 Node.js（CUDA）和浏览器（WebGPU）两个环境运行，打包工具（webpack/esbuild）无法 tree-shake 掉错误的 backend。依赖注入使环境切换为零代价。

**实现**: `GPT` 构造函数签名为 `constructor(config: GPTConfig, tf: typeof import('@tensorflow/tfjs'))`

### ADR-002: 训练进度通过 SSE 而非 WebSocket 传输

**决策**: 使用 Server-Sent Events，不用 WebSocket。

**原因**: 训练进度是单向数据流（server → client），SSE 语义更匹配，实现更简单，且 Next.js Route Handler 原生支持 SSE 响应流。

### ADR-003: 权重序列化使用 JSON + Base64

**决策**: Checkpoint 保存为 JSON 文件，Float32Array 用 Base64 编码存储。

**原因**: 便于调试（可直接查看文件内容），便于未来通过 HTTP API 传输给浏览器加载。RTX 3060 的 6GB 显存对应的模型（10M 参数 = 40MB）完全在合理范围内。

### ADR-004: 字符级 Tokenizer 作为第一阶段实现

**决策**: 第一阶段仅实现字符级 tokenizer（vocabSize=65），BPE 作为 Phase 2 扩展。

**原因**: 与原项目一致，适合 Shakespeare 数据集，减少第一阶段实现复杂度。`Tokenizer` 接口已预留 `type: 'char' | 'bpe'` 的扩展点。

---

## 8. 未来扩展路径（Phase 2+）

以下功能不在当前 Spec 范围内，但架构已预留扩展点：

1. **BPE Tokenizer**: 实现 `Tokenizer` 接口的 `BPETokenizer` 类，支持更大数据集
2. **用户专属模型训练**: 
   - 数据接入层（从应用数据库导出用户数据）
   - Fine-tuning 支持（加载预训练 checkpoint，在用户数据上继续训练）
   - 隐私保护（differential privacy，用户数据不出本地）
3. **模型导出**: 导出为 ONNX 格式，用于生产环境推理
4. **分布式训练**: 多 GPU 支持（trainer-node 的 Trainer 已使用 AsyncIterable，便于改造为 worker 模式）

---

## 9. 实现优先级与里程碑

| 里程碑 | 包含 Module | 可交付物 | 验证方式 |
|---|---|---|---|
| M1: 跑通训练 | 1, 2, 3, 4, 5, 6, 7, 8 | `trainer-node` CLI 可以在 RTX 3060 上完整训练并生成莎士比亚风格文本 | 运行 medium config 5000 步，val loss < 1.5 |
| M2: Web 可视化 | 9.1 | 浏览器实时看到 loss 曲线和生成样本 | 打开 localhost:3000 看到实时更新的 dashboard |
| M3: Playground | 9.2 | 可以在浏览器里输入 prompt 生成文本 | 输入 "To be or not to be" 得到合理续写 |
| M4: Attention 可视化 | 9.3 | 可视化每个 head 的注意力模式 | 注意力热力图正确渲染，可以切换 layer/head |

---

*文档结束。实现时如发现本文档与实际 API 有冲突，以 TF.js 官方文档为准，并在 PR 中更新本文档。*
