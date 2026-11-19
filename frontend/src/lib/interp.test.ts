import { describe, expect, it } from "vitest";
import { buildSpline, drawPoints } from "./interp";

const range = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

describe("buildSpline", () => {
  it("is exact at the knots", () => {
    const x = range(-3, 3, 13);
    const y = x.map((v) => Math.sin(v) + 0.1 * v * v);
    const s = buildSpline(x, y);
    x.forEach((v, i) => expect(s.evaluate(v)).toBe(y[i]));
  });

  it("keeps linear data linear (also between knots)", () => {
    const x = [0, 1, 2.5, 4, 7];
    const y = x.map((v) => 3 * v - 2);
    const s = buildSpline(x, y);
    for (const q of [0.3, 1.7, 3.1, 5.5, 6.99]) expect(s.evaluate(q)).toBeCloseTo(3 * q - 2, 12);
  });

  it("approximates a smooth cubic away from the natural ends", () => {
    const x = range(-1, 1, 21);
    const s = buildSpline(
      x,
      x.map((v) => v ** 3),
    );
    expect(s.evaluate(0.05)).toBeCloseTo(0.05 ** 3, 4);
    expect(s.evaluate(0.51)).toBeCloseTo(0.51 ** 3, 3);
  });

  it("matches a hand-computed natural spline", () => {
    // knots (0,0) (1,1) (2,0): symmetric, M1 = -3 and the ends are natural.
    // On [0, 1]: S(x) = M1 x^3 / 6 + (y1 - M1 / 6) x = -x^3 / 2 + 1.5 x.
    const s = buildSpline([0, 1, 2], [0, 1, 0]);
    expect(s.evaluate(0.5)).toBeCloseTo(-0.0625 + 0.75, 12);
    expect(s.evaluate(1.5)).toBeCloseTo(0.6875, 12);
  });

  it("returns null outside the range and does not extrapolate", () => {
    const s = buildSpline([0, 1, 2], [0, 1, 4]);
    expect(s.evaluate(-0.001)).toBeNull();
    expect(s.evaluate(2.001)).toBeNull();
    expect(s.evaluate(Number.NaN)).toBeNull();
    expect(s.domain).toEqual([0, 2]);
  });

  it("splits at gaps and never interpolates across them", () => {
    const s = buildSpline([0, 1, 2, 3, 4, 5], [0, 1, null, 3, 4, 5]);
    expect(s.segments).toBe(2);
    expect(s.evaluate(0.5)).not.toBeNull();
    expect(s.evaluate(1.5)).toBeNull();
    expect(s.evaluate(2)).toBeNull(); // the gap point itself
    expect(s.evaluate(3)).toBe(3);
    const r = s.resample(5);
    expect(r.y.filter((v) => v === null)).toHaveLength(1);
    // the drawn line is broken exactly once, between the two segments
    const k = r.y.indexOf(null);
    expect(r.x[k - 1]).toBe(1);
    expect(r.x[k + 1]).toBe(3);
  });

  it("treats a two-point segment as a line and a one-point segment as a single x", () => {
    const s = buildSpline([0, 1, 2, 5], [0, 2, null, 7]);
    expect(s.evaluate(0.25)).toBeCloseTo(0.5, 12);
    expect(s.evaluate(5)).toBe(7);
    expect(s.evaluate(5.5)).toBeNull();
    const r = s.resample(10);
    expect(r.x[r.x.length - 1]).toBe(5);
    expect(r.y[r.y.length - 1]).toBe(7);
  });

  it("sorts unordered input and drops repeated x", () => {
    const s = buildSpline([2, 0, 1, 1], [4, 0, 1, 99]);
    expect(s.evaluate(1)).toBe(1); // the first repeated x wins
    expect(s.evaluate(0)).toBe(0);
    expect(s.evaluate(2)).toBe(4);
    expect(buildSpline([], []).evaluate(0)).toBeNull();
    expect(buildSpline([], []).domain).toBeNull();
  });

  it("resamples densely, hitting the end points and the knots' values", () => {
    const x = range(0, 10, 11);
    const y = x.map((v) => Math.cos(v / 3));
    const s = buildSpline(x, y);
    const r = s.resample(drawPoints(11));
    expect(r.x).toHaveLength(200);
    expect(r.x[0]).toBe(0);
    expect(r.x[199]).toBe(10);
    expect(r.y[0]).toBe(y[0]);
    expect(r.y[199]).toBe(y[10]);
    r.x.forEach((v, i) => expect(r.y[i]).toBe(s.evaluate(v)));
    expect(drawPoints(121)).toBe(968);
  });
});
