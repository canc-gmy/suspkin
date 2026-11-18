import { describe, expect, it } from "vitest";
import type { HistoryData } from "../api/types";
import {
  convergenceTraces,
  G_CLIP,
  mapHistory,
  marginRows,
  marginTraces,
  paretoTraces,
  parcoordsTrace,
  isLogScale,
  type OptColors,
} from "./optimization";

const C: OptColors = { accent: "#00f", ink: "#000", muted: "#888", danger: "#f00", baseline: "#444" };

// rows arrive out of order: eval ids 2, 0, 1, 3
const hist: HistoryData = {
  run_id: "r",
  columns: {
    eval_id: [2, 0, 1, 3],
    feasible: [true, false, true, true],
    f_total: [5, 9, 7, null],
    f_a: [1, 3, 2, null],
    f_b: [4, 1, 2, null],
    "x_front.p": [0.2, 0.9, 0.5, 0.1],
    "x_rear.q": [0.3, 0.1, 0.8, 0.4],
  },
  meta: {},
  result: {},
  objectives: ["f_a", "f_b"],
  best_so_far: [5, null, 7, 5],
  pareto_all: [true, false, true, false],
  pareto_pairs: { "f_a|f_b": [true, false, true, false], "f_a|f_c": [false, false, true, false] },
  knee_eval_id: 2,
};

describe("mapHistory", () => {
  const v = mapHistory(hist);

  it("sorts every column by evaluation id and keeps the API analysis aligned", () => {
    expect(v.evalId).toEqual([0, 1, 2, 3]);
    expect(v.fTotal).toEqual([9, 7, 5, null]);
    expect(v.feasible).toEqual([false, true, true, true]);
    expect(v.best).toEqual([null, 7, 5, 5]);
    expect(v.paretoAll).toEqual([false, true, true, false]);
    expect(v.paretoPairs["f_a|f_c"]).toEqual([false, true, false, false]);
    expect(v.knee).toBe(2);
    expect(v.columns["f_a"]).toEqual([3, 2, 1, null]);
    expect(v.variables).toEqual(["x_front.p", "x_rear.q"]);
  });
});

describe("traces", () => {
  const v = mapHistory(hist);

  it("convergence separates infeasible points and drops rows without a value", () => {
    const [infeas, feas, best] = convergenceTraces(v, C);
    expect(infeas?.["x"]).toEqual([0]);
    expect(feas?.["x"]).toEqual([1, 2]);
    expect(best?.["x"]).toEqual([1, 2, 3]);
    expect(best?.["y"]).toEqual([7, 5, 5]);
  });

  it("pareto marks the front sorted by x and the knee", () => {
    const t = paretoTraces(v, "f_a", "f_b", C);
    const front = t.find((x) => x["name"] === "front (this pair)");
    expect(front?.["x"]).toEqual([1, 2]);
    expect(front?.["customdata"]).toEqual([2, 1]);
    const knee = t.find((x) => x["name"] === "knee");
    expect(knee?.["x"]).toEqual([1]);
    expect(t.find((x) => x["name"] === "feasible")?.["x"]).toEqual([]);
    const all = t.find((x) => x["name"] === "non-dominated (all objectives)");
    expect(all?.["customdata"]).toEqual([1, 2]);
  });

  it("draws the line of the chosen pair only, in either axis order", () => {
    const rows = { ...hist, objectives: ["f_a", "f_b", "f_c"] };
    const w = mapHistory({ ...rows, columns: { ...rows.columns, f_c: [1, 1, 1, null] } });
    const line = (x: string, y: string) => paretoTraces(w, x, y, C).find((t) => t["name"] === "front (this pair)");
    expect(line("f_a", "f_c")?.["customdata"]).toEqual([1]);
    expect(line("f_c", "f_a")?.["customdata"]).toEqual([1]);
    expect(line("f_b", "f_c")?.["customdata"]).toEqual([]); // no such pair in the response
  });

  it("parallel coordinates use feasible rows only", () => {
    const p = parcoordsTrace(v, C) as { dimensions: { label: string; values: unknown[] }[] } | null;
    expect(p?.dimensions.map((d) => d.label)).toEqual(["front.p", "rear.q"]);
    expect(p?.dimensions[0]?.values).toEqual([0.5, 0.2]);
    expect(parcoordsTrace({ ...v, feasible: v.feasible.map(() => false) }, C)).toBeNull();
  });
});

describe("margins and scales", () => {
  it("reads g of baseline and best and clips the bars", () => {
    const rows = marginRows({ baseline: { g: { a: -190, b: 0.5 } }, best: { g: { a: -1, b: null } } });
    expect(rows).toEqual([
      { name: "a", baseline: -190, best: -1 },
      { name: "b", baseline: 0.5, best: null },
    ]);
    const [base] = marginTraces(rows, C);
    expect(base?.["x"]).toEqual([-G_CLIP, 0.5]);
    expect(base?.["customdata"]).toEqual([-190, 0.5]);
    expect(marginRows({})).toEqual([]);
  });

  it("uses a log axis only for positive data spanning over two decades", () => {
    expect(isLogScale([1, 10, 1000])).toBe(true);
    expect(isLogScale([1, 10, 50])).toBe(false);
    expect(isLogScale([-1, 10, 1000])).toBe(false);
    expect(isLogScale([])).toBe(false);
  });
});
