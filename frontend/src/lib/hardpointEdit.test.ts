import { describe, expect, it } from "vitest";
import type { Hardpoint } from "../api/types";
import {
  buildEdits,
  cellChanged,
  changedPoints,
  editReducer,
  effectiveMm,
  indexBase,
  yamlFileName,
  type EditAction,
  type EditModel,
} from "./hardpointEdit";
import type { Vec3 } from "./hardpoints";

const hp = (point: string, side: Hardpoint["side"], x: number, y: number, z: number, overridden = false): Hardpoint => ({
  axle: "front",
  point,
  side,
  x,
  y,
  z,
  overridden,
});

// metres; lca_outer is symmetric, uca_outer has a right override
const POINTS: Hardpoint[] = [
  hp("lca_outer", "left", 0, 0.58, 0.13),
  hp("lca_outer", "right", 0, -0.58, 0.13),
  hp("uca_outer", "left", 0.01, 0.55, 0.3),
  hp("uca_outer", "right", 0.02, -0.56, 0.31, true),
];
const BASE = indexBase(POINTS);
const LCA = { axle: "front", point: "lca_outer" } as const;
const UCA = { axle: "front", point: "uca_outer" } as const;
const MM: Record<string, Vec3> = { left: [0, 580, 130], right: [0, -580, 130] };

const run = (...actions: EditAction[]): EditModel => actions.reduce<EditModel>(editReducer, null);
const set = (id: typeof LCA | typeof UCA, side: "left" | "right", axis: "x" | "y" | "z", mm: number | null, base: Vec3 = MM[side] as Vec3): EditAction => ({
  type: "set",
  ...id,
  side,
  axis,
  mm,
  base,
});

describe("editReducer", () => {
  it("ignores edits outside edit mode and starts clean", () => {
    expect(editReducer(null, set(LCA, "left", "z", 140))).toBeNull();
    expect(run({ type: "start", sourceId: "a" })).toEqual({ sourceId: "a", cells: {}, rightOnly: {} });
  });

  it("records a changed cell and drops it when typed back to the original or cleared", () => {
    const s = run({ type: "start", sourceId: "a" }, set(LCA, "left", "z", 142));
    expect(s?.cells["front|lca_outer|left"]).toEqual({ z: 142 });
    expect(run({ type: "start", sourceId: "a" }, set(LCA, "left", "z", 142), set(LCA, "left", "z", 130))?.cells).toEqual({});
    expect(run({ type: "start", sourceId: "a" }, set(LCA, "left", "z", 142), set(LCA, "left", "z", null))?.cells).toEqual({});
  });

  it("resets one point and discards everything", () => {
    const s = run(
      { type: "start", sourceId: "a" },
      set(LCA, "left", "z", 142),
      set(UCA, "left", "x", 12, [10, 550, 300]),
      { type: "rightOnly", ...LCA, on: true },
      { type: "reset", ...LCA },
    );
    expect(Object.keys(s?.cells ?? {})).toEqual(["front|uca_outer|left"]);
    expect(s?.rightOnly).toEqual({});
    expect(editReducer(s, { type: "discard" })).toBeNull();
  });

  it("unticking right-only drops the right edits (the right point mirrors again)", () => {
    const s = run(
      { type: "start", sourceId: "a" },
      { type: "rightOnly", ...LCA, on: true },
      set(LCA, "right", "z", 135),
      { type: "rightOnly", ...LCA, on: false },
    );
    expect(s?.cells).toEqual({});
    expect(s?.rightOnly["front|lca_outer"]).toBe(false);
  });
});

describe("mirroring display", () => {
  it("shows the right row as the mirror of the edited left row", () => {
    const s = run({ type: "start", sourceId: "a" }, set(LCA, "left", "y", 590))!;
    expect(effectiveMm(s, BASE, LCA, "right")).toEqual([0, -590, 130]);
    expect(cellChanged(s, BASE, LCA, "right", "y")).toBe(true);
    expect(cellChanged(s, BASE, LCA, "right", "z")).toBe(false);
  });

  it("keeps an overridden right point where it is until the user unticks right-only", () => {
    const left: Vec3 = [10, 550, 300];
    const s = run({ type: "start", sourceId: "a" }, set(UCA, "left", "z", 305, left))!;
    expect(effectiveMm(s, BASE, UCA, "right")).toEqual([20, -560, 310]);
    const off = run({ type: "start", sourceId: "a" }, { type: "rightOnly", ...UCA, on: false })!;
    expect(effectiveMm(off, BASE, UCA, "right")).toEqual([10, -550, 300]);
  });
});

describe("buildEdits", () => {
  it("is empty without edits or outside edit mode", () => {
    expect(buildEdits(null, BASE)).toEqual([]);
    expect(buildEdits(run({ type: "start", sourceId: "a" }), BASE)).toEqual([]);
  });

  it("converts mm to m and mirrors by default (side both)", () => {
    const s = run({ type: "start", sourceId: "a" }, set(LCA, "left", "z", 142.3), set(LCA, "left", "x", 5))!;
    expect(buildEdits(s, BASE)).toEqual([{ axle: "front", point: "lca_outer", side: "both", x: 0.005, z: 0.1423 }]);
    expect(changedPoints(s, BASE)).toBe(1);
  });

  it("right-only: right edit alone, and a left edit pins the right point", () => {
    const only = run({ type: "start", sourceId: "a" }, { type: "rightOnly", ...LCA, on: true }, set(LCA, "right", "z", 135))!;
    expect(buildEdits(only, BASE)).toEqual([{ axle: "front", point: "lca_outer", side: "right", z: 0.135 }]);
    const pinned = run({ type: "start", sourceId: "a" }, { type: "rightOnly", ...LCA, on: true }, set(LCA, "left", "z", 140))!;
    expect(buildEdits(pinned, BASE)).toEqual([
      { axle: "front", point: "lca_outer", side: "right", x: 0, y: -0.58, z: 0.13 },
      { axle: "front", point: "lca_outer", side: "left", z: 0.14 },
    ]);
  });

  it("a point with a right override writes left only, so the override stays", () => {
    const s = run({ type: "start", sourceId: "a" }, set(UCA, "left", "z", 305, [10, 550, 300]))!;
    expect(buildEdits(s, BASE)).toEqual([{ axle: "front", point: "uca_outer", side: "left", z: 0.305 }]);
  });

  it("unticking right-only on an override sends a full both edit to drop it", () => {
    const s = run({ type: "start", sourceId: "a" }, { type: "rightOnly", ...UCA, on: false })!;
    expect(buildEdits(s, BASE)).toEqual([{ axle: "front", point: "uca_outer", side: "both", x: 0.01, y: 0.55, z: 0.3 }]);
  });
});

describe("yamlFileName", () => {
  it("slugs the label and ends in .yaml", () => {
    expect(yamlFileName("opt best edited")).toBe("opt_best_edited.yaml");
    expect(yamlFileName("  raised lca (2) ")).toBe("raised_lca_2.yaml");
    expect(yamlFileName("///")).toBe("revision.yaml");
  });
});
