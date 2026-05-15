---
schema_version: v1
doc_type: adr
last_reviewed: 2026-05-15
---

# ADR 0001: Initial Architecture

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

***
