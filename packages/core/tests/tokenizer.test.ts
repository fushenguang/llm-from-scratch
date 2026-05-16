import { describe, expect, test } from "vitest";

import { CONFIGS } from "../src/model/GPTConfig.js";
import {
  CharTokenizer,
  assertTokenizerVocabSize
} from "../src/tokenizer/index.js";

describe("CharTokenizer", () => {
  const text = "hello world";
  const tokenizer = new CharTokenizer(text);
  const vocab65 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?";

  test("encode/decode roundtrip", () => {
    const encoded = tokenizer.encode(text);

    expect(tokenizer.decode(encoded)).toBe(text);
  });

  test("vocab is deterministic", () => {
    const secondTokenizer = new CharTokenizer(text);

    expect(secondTokenizer.vocabSize).toBe(tokenizer.vocabSize);
    expect(secondTokenizer.encode("hello")).toEqual(tokenizer.encode("hello"));
    expect(secondTokenizer.serialize()).toEqual(tokenizer.serialize());
  });

  test("unknown char throws", () => {
    expect(() => tokenizer.encode("你好")).toThrow(/Unknown character/);
  });

  test("serialize/deserialize", () => {
    const restored = CharTokenizer.fromState(tokenizer.serialize());

    expect(restored.encode("hello")).toEqual(tokenizer.encode("hello"));
    expect(restored.decode(tokenizer.encode(text))).toBe(text);
  });

  test("asserts tokenizer vocab size against GPT config", () => {
    const configTokenizer = new CharTokenizer(vocab65);

    expect(() => assertTokenizerVocabSize(configTokenizer, CONFIGS.tiny)).not.toThrow();
    expect(() =>
      assertTokenizerVocabSize(tokenizer, CONFIGS.tiny)
    ).toThrow(/does not match GPTConfig vocab size/);
  });
});
