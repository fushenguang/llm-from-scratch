---
schema_version: v1
doc_type: change-request-feature
last_reviewed: 2026-05-15
refs:
  - vision.md
  - constraints.md
  - interfaces.md
depends_on:
  - module-9-module-9-web-ui
---

## Module 2 测试（CharTokenizer）

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
