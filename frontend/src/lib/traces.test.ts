import { describe, expect, it } from "vitest";
import type { FigureSpec, PanelSpec } from "../api/spec";
import { buildPanel, lineStyleAt, plotlyDash, type RevisionData } from "./traces";

const axis = { column: "travel", label: "wheel travel", dim: "length", factor: 1000, unit: "mm" };
const s = (column: string, label: string) => ({
  column,
  label,
  dim: "angle",
  side: null,
  corner: null,
  factor: 2,
  unit: "deg",
});
const panel = (series: ReturnType<typeof s>[], extra: Partial<PanelSpec> = {}): PanelSpec => ({
  id: "p",
  title: "P",
  y_label: "[deg]",
  y_dim: "angle",
  ic: false,
  requires: [],
  alternative_to: null,
  x: null,
  series,
  ...extra,
});
const fig = (p: PanelSpec): FigureSpec => ({ id: "f", group: "g", title: "F", study: "front_corner", columns: 1, x: axis, panels: [p] });
const rev = (id: string, letter: string, color: string, y: (number | null)[], travel = [-0.01, 0, 0.01]): RevisionData => ({
  id,
  letter,
  color,
  table: { travel, left: y, right: y.map((v) => (v === null ? null : -v)) },
});

const LS = ["-", "--", ":", "-."];

describe("buildPanel", () => {
  it("draws one line per revision and series, applying display factors", () => {
    const p = panel([s("left", "Left"), s("right", "Right")]);
    const a = rev("a", "A", "#111111", [1, 2, 3]);
    const b = rev("b", "B", "#222222", [2, 3, 4]);
    const plot = buildPanel(fig(p), p, [a, b], a, "overlay", LS, "#999999");
    const lines = plot.traces.filter((t) => t["mode"] === "lines" && t["showlegend"] === false);
    expect(lines).toHaveLength(4);
    const x = lines[0]?.["x"] as number[];
    const y = lines[0]?.["y"] as number[];
    expect(x.length).toBeGreaterThanOrEqual(200); // dense resample of the spline
    expect([x[0], x[x.length - 1]]).toEqual([-10, 10]);
    expect([y[0], y[y.length - 1]]).toEqual([2, 6]);
    // the readout evaluates the very spline that is drawn
    expect(plot.curves).toHaveLength(4);
    expect(plot.curves[0]?.at(0)).toBeCloseTo(4, 12);
    expect(plot.curves[0]?.at(11)).toBeNull();
    x.forEach((v, i) => expect(plot.curves[0]?.at(v)).toBeCloseTo(y[i] as number, 12));
    const dashes = lines.map((t) => (t["line"] as { dash: string }).dash);
    expect(dashes).toEqual(["solid", "dash", "solid", "dash"]);
    expect(plot.legend).toBe(true);
    expect(plot.traces.filter((t) => t["showlegend"] === true).map((t) => t["name"])).toEqual(["Left", "Right"]);
    expect(plot.xTitle).toBe("Wheel travel [mm]");
    expect(plot.yTitle).toBe("[deg]");
  });

  it("has no legend for a single series and puts a ring on the reference at x = 0", () => {
    const p = panel([s("left", "Left")]);
    const a = rev("a", "A", "#111111", [1, 2, 3]);
    const b = rev("b", "B", "#222222", [2, 3, 4]);
    const plot = buildPanel(fig(p), p, [a, b], a, "overlay", LS, "#999999");
    expect(plot.legend).toBe(false);
    const rings = plot.traces.filter((t) => t["mode"] === "markers");
    expect(rings).toHaveLength(1);
    expect(rings[0]?.["x"]).toEqual([0]);
    expect(rings[0]?.["y"]).toEqual([4]);
  });

  it("shows revision minus reference in delta mode and hides the reference", () => {
    const p = panel([s("left", "Left")]);
    const a = rev("a", "A", "#111111", [1, 2, 3]);
    const b = rev("b", "B", "#222222", [2, 3, 4]);
    const plot = buildPanel(fig(p), p, [a, b], a, "delta", LS, "#999999");
    const lines = plot.traces.filter((t) => t["mode"] === "lines");
    expect(lines).toHaveLength(1);
    expect(lines[0]?.["name"]).toBe("B");
    const dy = lines[0]?.["y"] as number[];
    expect(dy.length).toBeGreaterThanOrEqual(200);
    dy.forEach((v) => expect(v).toBeCloseTo(2, 12)); // (+1) * factor 2
    expect(plot.curves).toHaveLength(1);
    expect(plot.curves[0]?.at(3)).toBeCloseTo(2, 12);
    expect(plot.traces.some((t) => t["mode"] === "markers")).toBe(false);
    expect(plot.yTitle).toBe("Δ [deg]");
  });

  it("counts undefined points of an IC panel", () => {
    const p = panel([s("left", "Left")], { ic: true });
    const a = rev("a", "A", "#111111", [1, null, 3]);
    expect(buildPanel(fig(p), p, [a], a, "overlay", LS, "#999").icDropped).toBe(1);
  });
});

describe("delta with a non-monotonic panel x", () => {
  it("interpolates on the study x and plots at the reference panel x", () => {
    const p = panel([s("left", "Left")], { x: { ...axis, column: "outer", factor: 1 } });
    const f = fig(p);
    const mk = (id: string, letter: string, y: number[], outer: number[]): RevisionData => ({
      id,
      letter,
      color: "#000",
      table: { travel: [-0.01, 0, 0.01], left: y, outer },
    });
    const a = mk("a", "A", [1, 2, 3], [0, 5, 1]); // outer rises then falls
    const b = mk("b", "B", [2, 3, 4], [0, 6, 2]);
    const plot = buildPanel(f, p, [a, b], a, "delta", LS, "#999");
    const line = plot.traces.find((t) => t["mode"] === "lines");
    const x = line?.["x"] as number[];
    const y = line?.["y"] as number[];
    // drawn at the reference's panel x (parametric in the study x): from 0, up past 5, down to 1
    expect([x[0], x[x.length - 1]]).toEqual([0, 1]);
    expect(Math.max(...x)).toBeGreaterThanOrEqual(5);
    y.forEach((v) => expect(v).toBeCloseTo(2, 12));
    expect(plot.curves[0]?.at(-10)).toBeCloseTo(2, 12);
  });
});

describe("solved points and gaps", () => {
  it("adds 3 px markers only at the real points when asked", () => {
    const p = panel([s("left", "Left")]);
    const a = rev("a", "A", "#111111", [1, null, 3, 4], [-0.01, 0, 0.01, 0.02]);
    const off = buildPanel(fig(p), p, [a], null, "overlay", LS, "#999");
    expect(off.traces.some((t) => t["mode"] === "markers")).toBe(false);
    const on = buildPanel(fig(p), p, [a], null, "overlay", LS, "#999", { showPoints: true });
    const dots = on.traces.filter((t) => t["mode"] === "markers");
    expect(dots).toHaveLength(1);
    expect(dots[0]?.["x"]).toEqual([-10, 10, 20]);
    expect(dots[0]?.["y"]).toEqual([2, 6, 8]);
    expect((dots[0]?.["marker"] as { size: number }).size).toBe(3);
  });

  it("never draws or reads across a gap", () => {
    const p = panel([s("left", "Left")], { ic: true });
    const a = rev("a", "A", "#111111", [1, 2, null, 4, 5], [-0.02, -0.01, 0, 0.01, 0.02]);
    const plot = buildPanel(fig(p), p, [a], null, "overlay", LS, "#999");
    const y = plot.traces.find((t) => t["mode"] === "lines")?.["y"] as (number | null)[];
    expect(y.filter((v) => v === null)).toHaveLength(1);
    expect(plot.curves[0]?.at(0)).toBeNull();
    expect(plot.curves[0]?.at(-15)).not.toBeNull();
  });

  it("has no hover labels", () => {
    const p = panel([s("left", "Left")]);
    const a = rev("a", "A", "#111111", [1, 2, 3]);
    const plot = buildPanel(fig(p), p, [a], a, "overlay", LS, "#999");
    for (const t of plot.traces) {
      expect(t["hovertemplate"]).toBeUndefined();
      expect(t["xaxis"]).toBeUndefined();
    }
    expect(plot.traces.filter((t) => t["mode"] === "lines" && t["showlegend"] === false).every((t) => t["hoverinfo"] === "none")).toBe(true);
  });
});

describe("line styles", () => {
  it("cycles linestyles by series index and maps them to Plotly dashes", () => {
    expect(lineStyleAt(LS, 5)).toBe("--");
    expect(["-", "--", ":", "-."].map(plotlyDash)).toEqual(["solid", "dash", "dot", "dashdot"]);
  });
});
