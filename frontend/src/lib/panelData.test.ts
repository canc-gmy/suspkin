import { describe, expect, it } from "vitest";
import type { FigureSpec, PanelSpec } from "../api/spec";
import { csvName, numText, panelRows, toCsv, toTsv } from "./panelData";
import type { RevisionData } from "./traces";

const axis = { column: "travel", label: "wheel travel", dim: "length", factor: 1000, unit: "mm" };
const ser = (column: string, label: string) => ({
  column,
  label,
  dim: "angle",
  side: null,
  corner: null,
  factor: 2, // display = SI x 2, so the unit conversion is visible
  unit: "deg",
});
const panel: PanelSpec = {
  id: "camber",
  title: "Camber",
  y_label: "[deg]",
  y_dim: "angle",
  ic: false,
  requires: [],
  alternative_to: null,
  x: null,
  series: [ser("left", "Camber left"), ser("right", "Camber right")],
};
const fig: FigureSpec = { id: "corner_alignment_front", group: "g", title: "F", study: "s", columns: 1, x: axis, panels: [panel] };
const rev = (id: string, letter: string, travel: number[], left: (number | null)[], right = left): RevisionData => ({
  id,
  letter,
  color: "#000000",
  table: { travel, left, right },
});

const A = rev("a", "A", [-0.01, 0, 0.01], [1, 2, 3]);
const B = rev("b", "B", [-0.01, 0, 0.01], [1.5, 2.5, 3.5], [0, 0, 0]);

describe("panelRows", () => {
  it("exports solved points in display units, one column per revision and series", () => {
    const rows = panelRows(fig, panel, [A, B], A, false);
    expect(rows[0]).toEqual([
      "Wheel travel [mm]",
      "A camber left [deg]",
      "A camber right [deg]",
      "B camber left [deg]",
      "B camber right [deg]",
    ]);
    expect(rows.slice(1)).toEqual([
      ["-10", "2", "2", "3", "0"],
      ["0", "4", "4", "5", "0"],
      ["10", "6", "6", "7", "0"],
    ]);
  });

  it("uses the union of x and leaves the missing cells empty", () => {
    const C = rev("c", "C", [0, 0.005], [1, null]);
    const rows = panelRows(fig, panel, [A, C], A, false);
    expect(rows.slice(1).map((r) => r[0])).toEqual(["-10", "0", "5", "10"]);
    expect(rows[1]).toEqual(["-10", "2", "2", "", ""]);
    expect(rows[2]).toEqual(["0", "4", "4", "2", "2"]);
    expect(rows[3]).toEqual(["5", "", "", "", ""]); // C has a point there but its value is missing
  });

  it("exports Δ columns at the reference x and leaves the reference out", () => {
    const rows = panelRows(fig, panel, [A, B], A, true);
    expect(rows[0]).toEqual(["Wheel travel [mm]", "B-A camber left [deg]", "B-A camber right [deg]"]);
    expect(rows.slice(1)).toEqual([
      ["-10", "1", "-2"],
      ["0", "1", "-4"],
      ["10", "1", "-6"],
    ]);
  });

  it("takes Δ from the spline when the revision has no point at a reference x", () => {
    const sparse = rev("s", "S", [-0.01, 0.01], [1, 3]); // a straight line: value 2 at x = 0
    const rows = panelRows(fig, panel, [A, sparse], A, true);
    expect(rows[2]).toEqual(["0", "0", "0"]); // 4 (S, spline) - 4 (A)
  });

  it("adds the panel x column for a panel with its own x axis", () => {
    const own: PanelSpec = { ...panel, x: { column: "px", label: "toe", dim: "angle", factor: 3, unit: "deg" } };
    const withPx: RevisionData = { ...A, table: { ...A.table, px: [0.1, 0.2, 0.3] } };
    const rows = panelRows(fig, own, [withPx], withPx, false);
    expect(rows[0]?.slice(0, 3)).toEqual(["Wheel travel [mm]", "A toe [deg]", "A camber left [deg]"]);
    expect(rows[1]?.slice(0, 3)).toEqual(["-10", "0.3", "2"]);
  });

  it("is empty in Δ mode without a reference and has only the header without data", () => {
    expect(panelRows(fig, panel, [A], null, true)).toEqual([]);
    expect(panelRows(fig, panel, [], null, false)).toEqual([["Wheel travel [mm]"]]);
  });
});

describe("text output", () => {
  it("removes float noise", () => {
    expect(numText(0.1 + 0.2)).toBe("0.3");
    expect(numText(-12.5)).toBe("-12.5");
  });

  it("writes CSV with quoting and LF ends, and TSV for the clipboard", () => {
    const rows = [["a, b", 'say "hi"'], ["1", ""]];
    expect(toCsv(rows)).toBe('"a, b","say ""hi"""\n1,\n');
    expect(toTsv(rows)).toBe('a, b\tsay "hi"\n1\t\n');
  });

  it("names the file after the figure and the panel", () => {
    expect(csvName(fig, panel)).toBe("corner_alignment_front_camber.csv");
  });
});
