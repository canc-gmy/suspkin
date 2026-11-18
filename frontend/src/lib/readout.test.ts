import { describe, expect, it } from "vitest";
import type { PanelSpec } from "../api/spec";
import {
  buildReadout,
  clampX,
  columnTitle,
  formatX,
  nudge,
  nudgeStep,
  parseX,
  pointerToX,
  readoutDecimals,
  signed,
  type ReadoutPanel,
} from "./readout";
import type { PanelCurve } from "./traces";

const ser = (column: string, side: PanelCurve["side"], unit = "deg") => ({
  column,
  label: column,
  dim: "angle",
  side,
  corner: null,
  factor: 1,
  unit,
});
const panel: PanelSpec = {
  id: "toe",
  title: "Toe",
  y_label: "[deg]",
  y_dim: "angle",
  ic: false,
  requires: [],
  alternative_to: null,
  x: null,
  series: [ser("l", "left"), ser("r", "right")],
};
const curve = (revId: string, k: number, side: PanelCurve["side"], at: (t: number) => number | null): PanelCurve => ({
  revId,
  letter: revId.toUpperCase(),
  color: "#000",
  seriesIndex: k,
  label: k === 0 ? "l" : "r",
  side,
  unit: "deg",
  at,
});
const revs = [
  { id: "a", letter: "A", color: "#111" },
  { id: "b", letter: "B", color: "#222" },
];

describe("decimals and steps", () => {
  it("derives decimals from the unit in one place", () => {
    expect(readoutDecimals("deg")).toBe(3);
    expect(readoutDecimals("mm")).toBe(2);
    expect(readoutDecimals("%")).toBe(2);
    expect(readoutDecimals("mm/deg")).toBe(3);
  });

  it("nudges by 0.1 (lengths) or 0.01 (angles), ten times that with Shift, without float drift", () => {
    const len = { dim: "length", unit: "mm" };
    const ang = { dim: "angle", unit: "deg" };
    expect(nudgeStep(len)).toBe(0.1);
    expect(nudgeStep(ang)).toBe(0.01);
    let x = 0;
    for (let i = 0; i < 3; i++) x = nudge(x, 1, false, len);
    expect(x).toBe(0.3);
    expect(nudge(1, -1, true, len)).toBe(0);
    expect(nudge(0, 1, true, ang)).toBe(0.1);
    expect(formatX(12.345, len)).toBe("12.35");
    expect(formatX(-0.0004, ang)).toBe("0.000");
  });
});

describe("buildReadout", () => {
  const panels: ReadoutPanel[] = [
    {
      panel,
      curves: [
        curve("a", 0, "left", (t) => -0.0612 + t),
        curve("a", 1, "right", () => -0.058),
        curve("b", 0, "left", (t) => (t > 5 ? null : 0.2)),
        curve("b", 1, "right", () => 0.214),
      ],
    },
  ];

  it("lays out L/R pairs with the panel's finer decimals and '-' out of range", () => {
    const t = buildReadout(panels, revs, "a", false, 0);
    expect(t.columns[0]?.title).toBe("Toe [deg]");
    expect(t.rows[0]?.cells[0]?.parts).toEqual([
      { prefix: "L", text: "−0.061" },
      { prefix: "R", text: "−0.058" },
    ]);
    const out = buildReadout(panels, revs, "a", false, 6);
    expect(out.rows[1]?.cells[0]?.parts[0]).toEqual({ prefix: "L", text: "-" });
  });

  it("shows signed values and a ref row in delta mode", () => {
    const d: ReadoutPanel[] = [{ panel, curves: [curve("b", 0, "left", () => 0.25), curve("b", 1, "right", () => -0.5)] }];
    const t = buildReadout(d, revs, "a", true, 1);
    expect(t.columns[0]?.title).toBe("Δ Toe [deg]");
    expect(t.rows[0]?.cells[0]?.parts[0]?.text).toBe("ref");
    expect(t.rows[1]?.cells[0]?.parts.map((p) => p.text)).toEqual(["+0.250", "−0.500"]);
  });

  it("is empty without a cursor and marks a revision without the panel", () => {
    const t = buildReadout(panels, [...revs, { id: "c", letter: "C", color: "#333" }], "a", false, null);
    expect(t.rows[0]?.cells[0]?.parts.every((p) => p.text === "")).toBe(true);
    expect(t.rows[2]?.cells[0]?.parts[0]?.text).toBe("-");
  });

  it("prefixes a single-series cell with nothing, and a sideless multi-series cell with its label", () => {
    const one: ReadoutPanel[] = [{ panel, curves: [curve("a", 0, null, () => 1)] }];
    expect(buildReadout(one, revs, null, false, 0).rows[0]?.cells[0]?.parts[0]?.prefix).toBeNull();
    const two: ReadoutPanel[] = [{ panel, curves: [curve("a", 0, null, () => 1), curve("a", 1, null, () => 2)] }];
    expect(buildReadout(two, revs, null, false, 0).rows[0]?.cells[0]?.parts.map((p) => p.prefix)).toEqual(["l", "r"]);
  });

  it("formats signed numbers and column titles", () => {
    expect(signed(0.004, 2)).toBe("0.00");
    expect(signed(-0.5, 1)).toBe("−0.5");
    expect(signed(null, 1)).toBe("-");
    expect(columnTitle({ ...panel, title: "Camber [deg]" }, false)).toBe("Camber [deg]");
  });
});

describe("pointer and typed x", () => {
  const xa = { _offset: 72, _length: 400, p2d: (px: number) => -20 + (px / 400) * 40 };
  const ya = { _offset: 8, _length: 250 };

  it("converts the pointer to data x and returns null outside the plot area", () => {
    const rect = { left: 100, top: 50 };
    expect(pointerToX(100 + 72 + 200, 50 + 8 + 100, rect, xa, ya)).toBeCloseTo(0, 12);
    expect(pointerToX(100 + 72, 50 + 8, rect, xa, ya)).toBe(-20);
    expect(pointerToX(100 + 71, 50 + 100, rect, xa, ya)).toBeNull();
    expect(pointerToX(100 + 200, 50 + 7, rect, xa, ya)).toBeNull();
    expect(pointerToX(100 + 200, 50 + 8 + 251, rect, xa, ya)).toBeNull();
  });

  it("parses typed values with a decimal comma or a Unicode minus and rejects junk", () => {
    expect(parseX(" 12,5 ")).toBe(12.5);
    expect(parseX("−3.25")).toBe(-3.25);
    expect(parseX("+.5")).toBe(0.5);
    expect(parseX("")).toBeNull();
    expect(parseX("1e3")).toBeNull();
    expect(parseX("abc")).toBeNull();
    expect(clampX(50, [-30, 30])).toBe(30);
    expect(clampX(5, null)).toBe(5);
  });
});
