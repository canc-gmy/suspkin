import { describe, expect, it } from "vitest";
import type { FigureSpec, PanelSpec, SeriesSpec } from "../api/spec";
import { figureColumns, figuresFor, groupLabel, groupsOf, hasAxles, seriesIn, selectPanels } from "./panels";

const ser = (column: string): SeriesSpec => ({
  column,
  label: column,
  dim: "angle",
  side: null,
  corner: null,
  factor: 1,
  unit: "deg",
});

const panel = (id: string, columns: string[], extra: Partial<PanelSpec> = {}): PanelSpec => ({
  id,
  title: id,
  y_label: "[deg]",
  y_dim: "angle",
  ic: false,
  requires: [],
  alternative_to: null,
  x: null,
  series: columns.map(ser),
  ...extra,
});

const axis = { column: "rack", label: "rack travel", dim: "length", factor: 1000, unit: "mm" };

// the steering figure: ratio panel, else the gain panel as its alternative
const steering: FigureSpec = {
  id: "steering",
  group: "steering",
  title: "Steering",
  study: "rack",
  columns: 2,
  x: axis,
  panels: [
    panel("steer_angle", ["d_left", "d_right"]),
    panel("ackermann", ["ackermann"], { x: { ...axis, column: "outer" } }),
    panel("steering_ratio", ["ratio"], { requires: ["ratio"] }),
    panel("steer_gain", ["gain"], { requires: ["gain"], alternative_to: "steering_ratio" }),
  ],
};

const cols = (...c: string[]) => new Set(c);

describe("selectPanels", () => {
  const base = ["rack", "d_left", "d_right", "outer", "ackermann", "gain"];

  it("falls back to the alternative panel when the required column is missing", () => {
    const ids = selectPanels(steering, [cols(...base)]).map((p) => p.id);
    expect(ids).toEqual(["steer_angle", "ackermann", "steer_gain"]);
  });

  it("draws the primary panel and drops its alternative when the requirement holds", () => {
    const ids = selectPanels(steering, [cols(...base, "ratio")]).map((p) => p.id);
    expect(ids).toEqual(["steer_angle", "ackermann", "steering_ratio"]);
  });

  it("needs a required column in only one dataset", () => {
    const ids = selectPanels(steering, [cols(...base), cols(...base, "ratio")]).map((p) => p.id);
    expect(ids).toContain("steering_ratio");
    expect(ids).not.toContain("steer_gain");
  });

  it("skips panels whose x column or series are missing", () => {
    expect(selectPanels(steering, [cols("rack", "d_left")]).map((p) => p.id)).toEqual(["steer_angle"]);
    // ackermann panel plots against its own x column "outer"
    expect(selectPanels(steering, [cols("rack", "ackermann")]).map((p) => p.id)).toEqual([]);
    expect(selectPanels(steering, [cols("d_left")])).toEqual([]);
    expect(selectPanels(steering, [])).toEqual([]);
  });
});

describe("seriesIn", () => {
  it("keeps the series index for the line style", () => {
    const p = panel("p", ["a", "b", "c"]);
    expect(seriesIn(p, cols("a", "c")).map((x) => x.index)).toEqual([0, 2]);
  });
});

describe("groups", () => {
  const fig = (id: string, group: string): FigureSpec => ({ ...steering, id, group });
  const figs = [fig("corner_alignment_front", "corner_alignment"), fig("corner_alignment_rear", "corner_alignment"), fig("roll", "roll")];

  it("lists groups in spec order with human labels", () => {
    expect(groupsOf(figs)).toEqual(["corner_alignment", "roll"]);
    expect(groupLabel("corner_alignment")).toBe("Alignment");
    expect(groupLabel("heave_pitch")).toBe("Heave & pitch");
    expect(groupLabel("some_new_group")).toBe("Some new group");
  });

  it("picks the figure of the chosen axle only for groups with axles", () => {
    expect(figuresFor(figs, "corner_alignment", "rear").map((f) => f.id)).toEqual(["corner_alignment_rear"]);
    expect(figuresFor(figs, "roll", "rear").map((f) => f.id)).toEqual(["roll"]);
    expect(hasAxles(figs.slice(0, 2))).toBe(true);
    expect(hasAxles(figs.slice(2))).toBe(false);
  });

  it("lays out Front, Rear and Both as columns", () => {
    const ids = (cols: FigureSpec[][]) => cols.map((c) => c.map((f) => f.id));
    expect(ids(figureColumns(figs, "corner_alignment", "front"))).toEqual([["corner_alignment_front"]]);
    expect(ids(figureColumns(figs, "corner_alignment", "both"))).toEqual([
      ["corner_alignment_front"],
      ["corner_alignment_rear"],
    ]);
    // a group without axles never splits, whatever the choice
    expect(ids(figureColumns(figs, "roll", "both"))).toEqual([["roll"]]);
    expect(ids(figureColumns(figs, "roll", "rear"))).toEqual([["roll"]]);
  });
});
