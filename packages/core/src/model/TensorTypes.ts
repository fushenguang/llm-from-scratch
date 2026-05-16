export type TypedArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Int16Array
  | Int8Array;

export interface TensorLike {
  readonly shape: number[];
  readonly size: number;
  dispose(): void;
  dataSync(): TypedArray;
}

export interface VariableLike extends TensorLike {
  readonly name: string;
}

export type TensorLikeInput = TensorLike | number;

export interface TensorflowLike {
  tidy<T>(fn: () => T): T;
  variable(initialValue: TensorLike, trainable?: boolean, name?: string): VariableLike;
  randomNormal(
    shape: number[],
    mean?: number,
    stdDev?: number,
    dtype?: string
  ): TensorLike;
  zeros(shape: number[], dtype?: string): TensorLike;
  ones(shape: number[], dtype?: string): TensorLike;
  range(start: number, stop: number, step?: number, dtype?: string): TensorLike;
  reshape(x: TensorLike, shape: number[]): TensorLike;
  transpose(x: TensorLike, perm?: number[]): TensorLike;
  matMul(
    a: TensorLike,
    b: TensorLike,
    transposeA?: boolean,
    transposeB?: boolean
  ): TensorLike;
  add(a: TensorLikeInput, b: TensorLikeInput): TensorLike;
  sub(a: TensorLikeInput, b: TensorLikeInput): TensorLike;
  mul(a: TensorLikeInput, b: TensorLikeInput): TensorLike;
  square(x: TensorLike): TensorLike;
  mean(x: TensorLike, axis?: number | number[], keepDims?: boolean): TensorLike;
  rsqrt(x: TensorLikeInput): TensorLike;
  tanh(x: TensorLike): TensorLike;
  pow(base: TensorLike, exp: TensorLikeInput): TensorLike;
  split(x: TensorLike, numOrSizeSplits: number | number[], axis?: number): TensorLike[];
  softmax(logits: TensorLike, dim?: number): TensorLike;
  slice(x: TensorLike, begin: number[], size: number[]): TensorLike;
  gather(x: TensorLike, indices: TensorLike, axis?: number): TensorLike;
  oneHot(indices: TensorLike, depth: number): TensorLike;
  linalg: {
    bandPart(a: TensorLike, numLower: number, numUpper: number): TensorLike;
  };
  losses: {
    softmaxCrossEntropy(oneHotLabels: TensorLike, logits: TensorLike): TensorLike;
  };
}
