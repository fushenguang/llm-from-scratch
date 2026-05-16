import { afterEach, describe, expect, test, vi } from "vitest";

import { DataLoader } from "../src/training/index.js";

describe("DataLoader", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("nextBatch returns int32 tensors with expected shape and shifted targets", () => {
    const loader = new DataLoader(Uint16Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 2, 3);
    const randomSpy = vi.spyOn(Math, "random");

    randomSpy.mockReturnValueOnce(0);
    randomSpy.mockReturnValueOnce(0.5);

    const { x, y } = loader.nextBatch();

    expect(randomSpy).toHaveBeenCalledTimes(2);
    expect(x.shape).toEqual([2, 3]);
    expect(y.shape).toEqual([2, 3]);
    expect(x.dtype).toBe("int32");
    expect(y.dtype).toBe("int32");
    expect(Array.from(x.dataSync())).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(y.dataSync())).toEqual([1, 2, 3, 4, 5, 6]);

    x.dispose();
    y.dispose();
  });

  test("samples each row from independent random start positions", () => {
    const loader = new DataLoader(Uint16Array.from([10, 11, 12, 13, 14, 15, 16, 17]), 3, 2);
    const randomSpy = vi.spyOn(Math, "random");

    randomSpy.mockReturnValueOnce(0.99);
    randomSpy.mockReturnValueOnce(0.25);
    randomSpy.mockReturnValueOnce(0.99);

    const { x, y } = loader.nextBatch();
    const xRows = Array.from(x.dataSync());
    const yRows = Array.from(y.dataSync());

    expect(xRows).toEqual([15, 16, 11, 12, 15, 16]);
    expect(yRows).toEqual([16, 17, 12, 13, 16, 17]);

    x.dispose();
    y.dispose();
  });

  test("accepts the minimum dataset length of blockSize plus one token", () => {
    const loader = new DataLoader(Uint16Array.from([4, 5, 6, 7]), 2, 3);
    const randomSpy = vi.spyOn(Math, "random").mockReturnValue(0.99);

    const { x, y } = loader.nextBatch();

    expect(randomSpy).toHaveBeenCalledTimes(2);
    expect(Array.from(x.dataSync())).toEqual([4, 5, 6, 4, 5, 6]);
    expect(Array.from(y.dataSync())).toEqual([5, 6, 7, 5, 6, 7]);

    x.dispose();
    y.dispose();
  });

  test("rejects datasets shorter than blockSize plus one token", () => {
    expect(() => new DataLoader(Uint16Array.from([1, 2, 3]), 2, 3)).toThrow(
      /blockSize \+ 1/
    );
  });

  test("rejects empty datasets with the minimum-length validation", () => {
    expect(() => new DataLoader(new Uint16Array(0), 2, 3)).toThrow(/blockSize \+ 1/);
  });

  test("rejects datasets that are not Uint16Array instances", () => {
    const invalidTokens = Int32Array.from([1, 2, 3, 4]) as unknown as Uint16Array;

    expect(() => new DataLoader(invalidTokens, 2, 2)).toThrow(/Uint16Array/);
  });

  test("rejects non-positive batch and block sizes", () => {
    expect(() => new DataLoader(Uint16Array.from([1, 2, 3, 4]), 0, 2)).toThrow(/batchSize/);
    expect(() => new DataLoader(Uint16Array.from([1, 2, 3, 4]), 2, 0)).toThrow(/blockSize/);
  });

  test("rejects non-integer batch and block sizes", () => {
    expect(() => new DataLoader(Uint16Array.from([1, 2, 3, 4]), 1.5, 2)).toThrow(/batchSize/);
    expect(() => new DataLoader(Uint16Array.from([1, 2, 3, 4]), 2, 1.5)).toThrow(/blockSize/);
  });
});
