import { describe, expect, it } from "vitest";
import { deltaToReference, interpolate } from "./delta";

describe("interpolate", () => {
  it("is exact on a line, at nodes and between them", () => {
    const xs = [0, 1, 2, 4];
    const ys = xs.map((x) => 3 * x - 1);
    const q = [0, 0.5, 1, 3, 4];
    expect(interpolate(xs, ys, q)).toEqual(q.map((x) => 3 * x - 1));
  });

  it("handles a decreasing x grid", () => {
    expect(interpolate([4, 2, 0], [8, 4, 0], [1, 3])).toEqual([2, 6]);
  });

  it("returns null outside the range (no extrapolation)", () => {
    expect(interpolate([0, 1], [0, 1], [-0.1, 1.1, Number.NaN])).toEqual([null, null, null]);
  });

  it("returns all null for a non-monotonic or repeated x grid", () => {
    expect(interpolate([0, 2, 1], [0, 1, 2], [0.5, 1.5])).toEqual([null, null]);
    expect(interpolate([0, 1, 1], [0, 1, 2], [0.5])).toEqual([null]);
    expect(interpolate([0, null, 2], [0, 1, 2], [1])).toEqual([null]);
  });

  it("does not bridge gaps in y", () => {
    expect(interpolate([0, 1, 2], [0, null, 2], [0.5, 1, 1.5])).toEqual([null, null, null]);
  });

  it("handles degenerate grids", () => {
    expect(interpolate([], [], [0])).toEqual([null]);
    expect(interpolate([1], [5], [1, 2])).toEqual([5, null]);
  });
});

describe("deltaToReference", () => {
  it("subtracts the reference on the reference grid", () => {
    const refX = [0, 1, 2];
    const refY = [0, 1, 2];
    // revision sampled on a different grid, offset +0.5 from the reference
    const out = deltaToReference(refX, refY, [0, 0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2, 2.5]);
    out.forEach((v) => expect(v).toBeCloseTo(0.5, 12));
  });

  it("is null where the revision does not cover the reference x or either side is missing", () => {
    expect(deltaToReference([0, 1, 2], [0, 1, null], [0, 1], [0, 1])).toEqual([0, 0, null]);
  });

  it("is zero for a revision identical to the reference", () => {
    const x = [-1, 0, 1];
    const y = [3, 2, 5];
    expect(deltaToReference(x, y, x, y)).toEqual([0, 0, 0]);
  });
});
