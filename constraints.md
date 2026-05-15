---
schema_version: v1
doc_type: constraints
last_reviewed: 2026-05-15
---

# Constraints

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

* ❌ 禁止使用 `any` 类型，用 `unknown` + type guard 替代
* ❌ 禁止在模型代码中使用同步文件 IO（`fs.readFileSync`），一律 async
* ❌ 禁止在浏览器包中引入 `@tensorflow/tfjs-node-gpu`
* ❌ 禁止硬编码训练超参数，所有参数必须通过 `GPTConfig` 传入
* ❌ 禁止在 tensor 操作后不调用 `.dispose()`（内存泄漏）
* ❌ 禁止跳过 §5 的验收测试直接进入下一模块

### 1.3 必须事项

* ✅ 每个 tensor 操作后必须在注释中标注 shape，格式：`// shape: [B, T, C]`
* ✅ 每个公共函数必须有 JSDoc，包含参数说明和返回值
* ✅ 所有数值超参数必须有合理范围的运行时断言（`assert`）
* ✅ 训练循环中必须使用 `tf.tidy()` 管理中间 tensor 生命周期

***
