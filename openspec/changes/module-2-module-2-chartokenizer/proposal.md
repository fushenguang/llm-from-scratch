---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-1-module-1-monorepo
---

## Module 2: CharTokenizer

**文件**: `packages/core/src/tokenizer/CharTokenizer.ts`

**实现要求**:

* 构造函数接受训练文本 `string`，自动构建 char→index 和 index→char 双向映射
* `vocabSize` = 唯一字符数量
* `encode(text)`: 返回 `number[]`，遇到未知字符抛出 `Error`，不得静默跳过
* `decode(tokens)`: 返回 `string`
* `serialize()` / 静态方法 `CharTokenizer.fromState(state)` 实现双向转换

**关键实现细节**:

```
字符排序必须确定性（sort()），确保同样的训练文本每次产生相同的 vocabSize 和映射
vocabSize 必须与 GPTConfig.vocabSize 一致，训练开始前必须断言
```

**验收**: 见 §5 Module 2 测试

***
