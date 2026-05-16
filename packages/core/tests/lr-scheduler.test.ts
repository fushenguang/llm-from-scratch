import { describe, expect, test } from "vitest";

import { getLR } from "../src/training/index.js";

describe("LRScheduler", () => {
  const config = { maxLr: 1e-3, warmupSteps: 100, maxSteps: 1000 };

  test("lr starts at 0 at step 0", () => {
    expect(getLR(0, config)).toBeCloseTo(0);
  });

  test("lr reaches maxLr at end of warmup", () => {
    expect(getLR(100, config)).toBeCloseTo(1e-3);
  });

  test("lr decays after warmup", () => {
    expect(getLR(500, config)).toBeLessThan(1e-3);
    expect(getLR(500, config)).toBeGreaterThan(1e-4);
  });

  test("lr reaches minLr at end of training", () => {
    expect(getLR(1000, config)).toBeCloseTo(1e-4);
  });

  test("lr follows cosine decay halfway through the decay phase", () => {
    expect(getLR(550, config)).toBeCloseTo(5.5e-4);
  });

  test("lr stays at minLr after maxSteps", () => {
    expect(getLR(1500, config)).toBeCloseTo(1e-4);
  });

  test("zero warmup starts directly at maxLr", () => {
    expect(getLR(0, { maxLr: 1e-3, warmupSteps: 0, maxSteps: 1000 })).toBeCloseTo(1e-3);
  });

  test("rejects invalid scheduler configuration", () => {
    expect(() => getLR(-1, config)).toThrow(/step/);
    expect(() => getLR(0, { ...config, maxLr: 0 })).toThrow(/maxLr/);
    expect(() => getLR(0, { ...config, warmupSteps: 1000 })).toThrow(/warmupSteps/);
  });
});
