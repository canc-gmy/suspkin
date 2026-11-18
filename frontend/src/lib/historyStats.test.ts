import { describe as suite, expect, it } from "vitest";
import type { HistoryData } from "../api/types";
import {
  F_FAIL,
  baselineRow,
  bestRow,
  constraintStats,
  countBy,
  describe,
  evolutionBands,
  failedRows,
  feasibleRows,
  mapStats,
  mean,
  objectiveStats,
  paramStats,
  quantile,
  scoreHistogram,
  std,
  summarize,
  topRows,
  windowSize,
} from "./historyStats";
import { histogramTraces, paramTraces, withAlpha } from "./historyStatsCharts";

interface Row {
  f: number;
  feasible?: boolean;
  status?: string;
  source?: string;
  seconds?: number;
  px?: number;
  g?: number;
}

function history(rows: Row[], extra: Partial<HistoryData> = {}, result: Record<string, unknown> = {}): HistoryData {
  const col = (fn: (r: Row, i: number) => number | string | boolean) => rows.map(fn);
  return {
    run_id: "r",
    columns: {
      eval_id: col((_, i) => i),
      "p_front.a_x": col((r) => (r.px ?? 0.1) * 1),
      f_one: col((r) => r.f / 2),
      f_two: col((r) => r.f / 4),
      f_total: col((r) => r.f),
      g_c: col((r) => r.g ?? -1),
      feasible: col((r) => r.feasible ?? true),
      status: col((r) => r.status ?? "ok"),
      seconds: col((r) => r.seconds ?? 1),
      source: col((r) => r.source ?? "de"),
    },
    meta: {},
    result,
    objectives: ["f_one", "f_two"],
    best_so_far: [],
    pareto_all: [],
    pareto_pairs: {},
    knee_eval_id: null,
    ...extra,
  } as HistoryData;
}

suite("basic statistics", () => {
  it("mean and sample std", () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    // sample std of 2,4,4,4,5,5,7,9 is sqrt(32/7), not the population value 2
    expect(std([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(Math.sqrt(32 / 7), 12);
    expect(std([5])).toBeNull();
    expect(mean([])).toBeNull();
  });
  it("quantiles interpolate linearly", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantile([4, 1, 3, 2], 0.25)).toBeCloseTo(1.75, 12);
    expect(quantile([7], 0.9)).toBe(7);
    expect(quantile([], 0.5)).toBeNull();
    expect(quantile([1, 2], 2)).toBe(2);
  });
  it("describe and countBy", () => {
    const d = describe([3, 1, 2]);
    expect([d.n, d.min, d.median, d.max, d.mean]).toEqual([3, 1, 2, 3, 2]);
    expect(describe([]).mean).toBeNull();
    expect(countBy(["a", "b", "a"])).toEqual([
      ["a", 2],
      ["b", 1],
    ]);
  });
});

suite("row classes", () => {
  const d = mapStats(
    history([
      { f: 5 },
      { f: 3, feasible: false },
      { f: F_FAIL, feasible: false, status: "unsolved" },
      { f: 1 },
      { f: F_FAIL, status: "ok" },
    ]),
  );
  it("flags failed rows by status and penalty", () => {
    expect(failedRows(d)).toEqual([false, false, true, false, true]);
    expect(feasibleRows(d)).toEqual([0, 3]);
  });
  it("sorts rows by eval_id", () => {
    const h = history([{ f: 1 }, { f: 2 }, { f: 3 }]);
    h.columns["eval_id"] = [2, 0, 1];
    expect(mapStats(h).fTotal).toEqual([2, 3, 1]);
  });
  it("summary counts exclude failures from the means", () => {
    const s = summarize(d);
    expect(s.total).toBe(5);
    expect(s.byStatus).toEqual([
      ["ok", 4],
      ["unsolved", 1],
    ]);
    expect(s.failed).toBe(2);
    expect(s.feasibleShare).toBeCloseTo(40, 12);
    expect(s.fFeasible.mean).toBe(3);
    expect(s.fOk.mean).toBe(3);
    expect(s.fOk.n).toBe(3);
    expect(s.seconds.mean).toBe(1);
  });
});

suite("top group", () => {
  const rows = Array.from({ length: 40 }, (_, i) => ({ f: 40 - i }));
  const d = mapStats(history(rows));
  it("takes the best 10 percent, at least 5", () => {
    const all = feasibleRows(d);
    const top = topRows(d, all) as number[];
    expect(top).toHaveLength(5); // ceil(4) = 4, raised to the minimum of 5
    expect(top.map((i) => d.fTotal[i])).toEqual([1, 2, 3, 4, 5]);
    const big = mapStats(history(Array.from({ length: 100 }, (_, i) => ({ f: i }))));
    expect(topRows(big, feasibleRows(big))).toHaveLength(10);
  });
  it("is null with fewer than 5 samples", () => {
    expect(topRows(d, [0, 1, 2, 3])).toBeNull();
  });
});

suite("histogram", () => {
  const d = mapStats(
    history([
      { f: 1 },
      { f: 2 },
      { f: 3 },
      { f: 4, feasible: false },
      { f: F_FAIL, feasible: false, status: "infeasible" },
      { f: F_FAIL, feasible: false, status: "unsolved" },
    ]),
  );
  it("excludes failed rows from the range and counts them", () => {
    const h = scoreHistogram(d, 3)!;
    expect(h.excluded).toBe(2);
    expect(h.edges[0]).toBe(1);
    expect(h.edges[3]).toBe(4);
    expect(h.feasible.reduce((a, b) => a + b, 0)).toBe(3);
    expect(h.infeasible.reduce((a, b) => a + b, 0)).toBe(1);
    expect(h.infeasible[2]).toBe(1); // the maximum lands in the last bin
    expect([h.best, h.median, h.mean]).toEqual([1, 2, 2]);
  });
  it("handles a constant score and no ok rows", () => {
    const one = scoreHistogram(mapStats(history([{ f: 2 }, { f: 2 }])))!;
    expect(one.feasible.reduce((a, b) => a + b, 0)).toBe(2);
    expect(scoreHistogram(mapStats(history([{ f: F_FAIL, status: "unsolved", feasible: false }])))).toBeNull();
  });
  it("marks best, median and mean in the legend", () => {
    const names = histogramTraces(scoreHistogram(d)!, { accent: "#fff", ink: "#000", muted: "#888", danger: "#f00", baseline: "#888" }).map(
      (t) => t["name"] as string,
    );
    expect(names.slice(0, 2)).toEqual(["feasible", "infeasible"]);
    expect(names.slice(2).map((n) => n.split(" ")[0])).toEqual(["best", "median", "mean"]);
  });
});

suite("objectives, parameters and constraints", () => {
  const rows: Row[] = [
    { f: 4, px: 0.1, g: -2 },
    { f: 2, px: 0.2, g: -1 },
    { f: 6, px: 0.3, g: 3, feasible: false },
    { f: 1, px: 0.4, g: 0 },
  ];
  const d = mapStats(history(rows, {}, { best_eval_id: 3, baseline: { f_total: 4 } }));
  it("converts p_ columns from m to mm", () => {
    expect(d.paramUnit).toBe("mm");
    expect(d.params["front.a_x"]).toEqual([100, 200, 300, 400].map((v) => expect.closeTo(v, 9)));
  });
  it("falls back to the unit cube without p_ columns", () => {
    const h = history(rows);
    h.columns["x_a"] = [0.1, 0.2, 0.3, 0.4];
    delete h.columns["p_front.a_x"];
    const m = mapStats(h);
    expect(m.paramUnit).toBe("unit cube");
    expect(m.params["a"]).toEqual([0.1, 0.2, 0.3, 0.4]);
  });
  it("finds the best and the baseline rows", () => {
    expect(bestRow(d)).toBe(3);
    expect(baselineRow(d)).toBe(0);
    expect(bestRow(mapStats(history(rows)))).toBe(3);
    expect(baselineRow(mapStats(history(rows)))).toBeNull();
  });
  it("objective statistics use feasible rows and the best design", () => {
    const o = objectiveStats(d)[0]!;
    expect(o.name).toBe("f_one");
    expect(o.values).toEqual([2, 1, 0.5]);
    expect(o.mean).toBeCloseTo(3.5 / 3, 12);
    expect(o.bestDesign).toBe(0.5);
  });
  it("parameter statistics in mm, with a top group only from 5 samples", () => {
    const p = paramStats(d)[0]!;
    expect(p.all.mean).toBeCloseTo((100 + 200 + 400) / 3, 9);
    expect(p.top).toBeNull();
    expect(p.best).toBeCloseTo(400, 9);
    expect(p.baseline).toBeCloseTo(100, 9);
    expect([p.lo, p.hi]).toEqual([expect.closeTo(100, 9), expect.closeTo(400, 9)]);
    const big = mapStats(history(Array.from({ length: 30 }, (_, i) => ({ f: 30 - i, px: 0.1 + i / 100 }))));
    const q = paramStats(big)[0]!;
    expect(q.top!.n).toBe(5);
    expect(q.top!.mean!).toBeGreaterThan(q.all.mean!); // best scores sit at high px
  });
  it("drops constant parameters", () => {
    expect(paramStats(mapStats(history([{ f: 1 }, { f: 2 }])))).toEqual([]);
  });
  it("normalises the dot chart to the explored range", () => {
    const p = paramStats(d);
    const t = paramTraces(p, { accent: "a", ink: "i", muted: "m", danger: "d", baseline: "b" });
    expect((t[2]!["x"] as number[])[0]).toBeCloseTo(1, 9); // best is at the top of the range
  });
  it("constraint pass rates count g at or below zero over ok samples", () => {
    const c = constraintStats(d)[0]!;
    expect(c.name).toBe("g_c");
    expect(c.all.n).toBe(4);
    expect(c.all.passRate).toBe(75);
    expect(c.all.maxG).toBe(3);
    expect(c.all.meanG).toBe(0);
    expect(c.top).toBeNull();
  });
});

suite("evolution", () => {
  it("window size", () => {
    expect(windowSize(50)).toBe(10);
    expect(windowSize(1000)).toBe(50);
  });
  it("band of feasible samples per window", () => {
    const rows: Row[] = Array.from({ length: 20 }, (_, i) => ({ f: i < 10 ? 10 : 20, feasible: i !== 3 }));
    const bands = evolutionBands(mapStats(history(rows)), 10);
    expect(bands).toHaveLength(2);
    expect(bands[0]).toMatchObject({ mean: 10, std: 0, n: 9 });
    expect(bands[1]).toMatchObject({ mean: 20, n: 10 });
    expect(bands[1]!.x).toBeCloseTo(14.5, 12);
  });
  it("skips windows without feasible samples and gives no std for one sample", () => {
    const rows: Row[] = [{ f: 1, feasible: false }, { f: 1, feasible: false }, { f: 5 }];
    const b = evolutionBands(mapStats(history(rows)), 2);
    expect(b).toHaveLength(1);
    expect(b[0]!.std).toBeNull();
  });
});

suite("edge cases", () => {
  it("handles an empty history and no feasible samples", () => {
    const e = mapStats(history([]));
    expect(e.n).toBe(0);
    expect(summarize(e).feasibleShare).toBeNull();
    const none = mapStats(history([{ f: 3, feasible: false }]));
    expect(summarize(none).fFeasible.mean).toBeNull();
    expect(bestRow(none)).toBeNull();
    expect(evolutionBands(none)).toEqual([]);
    expect(paramStats(none)[0]?.all.mean ?? null).toBeNull();
  });
  it("withAlpha", () => {
    expect(withAlpha("#23468F", 0.5)).toBe("rgba(35, 70, 143, 0.5)");
    expect(withAlpha("red", 0.5)).toBe("red");
  });
});
