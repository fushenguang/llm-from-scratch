export interface LRSchedulerConfig {
  maxLr: number;
  warmupSteps: number;
  maxSteps: number;
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function assertFinitePositive(value: number, name: string): void {
  assertCondition(Number.isFinite(value), `${name} must be finite.`);
  assertCondition(value > 0, `${name} must be greater than 0.`);
}

function assertNonNegativeInteger(value: number, name: string): void {
  assertCondition(Number.isInteger(value), `${name} must be an integer.`);
  assertCondition(value >= 0, `${name} must be non-negative.`);
}

/**
 * Computes the learning rate for a given training step using linear warmup
 * followed by cosine decay down to 10% of the maximum learning rate.
 *
 * @param step Current zero-based training step.
 * @param config Scheduler configuration values.
 * @returns Learning rate for the provided step.
 */
export function getLR(step: number, config: LRSchedulerConfig): number {
  assertNonNegativeInteger(step, "step");
  assertFinitePositive(config.maxLr, "maxLr");
  assertNonNegativeInteger(config.warmupSteps, "warmupSteps");
  assertNonNegativeInteger(config.maxSteps, "maxSteps");
  assertCondition(config.maxSteps > 0, "maxSteps must be greater than 0.");
  assertCondition(
    config.warmupSteps < config.maxSteps,
    "warmupSteps must be less than maxSteps."
  );

  const minLr = config.maxLr * 0.1;

  if (step < config.warmupSteps) {
    return config.maxLr * (step / config.warmupSteps);
  }

  const clampedStep = Math.min(step, config.maxSteps);
  const progress =
    (clampedStep - config.warmupSteps) / (config.maxSteps - config.warmupSteps);

  return minLr + 0.5 * (config.maxLr - minLr) * (1 + Math.cos(Math.PI * progress));
}
