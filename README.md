# llm-from-scratch-ts

TypeScript monorepo scaffold for the "LLM From Scratch" project.

## Workspace layout

- `packages/core`: shared model, tokenizer, training, and generation logic
- `packages/trainer-node`: Node.js training process and SSE server
- `packages/web`: Next.js App Router UI

## Tooling

- `pnpm` workspace via `pnpm-workspace.yaml`
- shared TypeScript defaults in `tsconfig.base.json`
- workspace tests configured in `vitest.config.ts`

## Commands

```bash
pnpm install
pnpm -r build
pnpm test
```
