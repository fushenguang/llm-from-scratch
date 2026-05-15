---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on: []
---

## Module 1: 项目脚手架与 Monorepo 配置

**目标**: 建立可运行的空项目骨架

**任务清单**:

1. 在项目根创建 `pnpm-workspace.yaml`，声明 `packages/*` 为 workspace
2. 创建 `tsconfig.base.json`（见下方配置）
3. 创建三个 package 的 `package.json`，核心依赖如下：
   * `packages/core`: 无运行时依赖，仅 devDependencies（`@tensorflow/tfjs` 用于类型，不在 core 中 import 实现）
   * `packages/trainer-node`: `@tensorflow/tfjs-node-gpu`, `express` (SSE server)
   * `packages/web`: `next`, `@tensorflow/tfjs`, `@tensorflow/tfjs-backend-webgpu`
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

***
