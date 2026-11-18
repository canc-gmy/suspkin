import { describe, expect, it } from "vitest";
import type { Hardpoint } from "../api/types";
import { buildHardpointRows, changedSceneNames, filterRows, sceneName } from "./hardpoints";

const hp = (axle: Hardpoint["axle"], point: string, side: Hardpoint["side"], x: number, y: number, z: number, overridden = false): Hardpoint => ({
  axle,
  point,
  side,
  x,
  y,
  z,
  overridden,
});

// metres
const REF: Hardpoint[] = [
  hp("rear", "uca_outer", "left", -1.6, 0.52, 0.3),
  hp("front", "tie_inner", "left", 0.08, 0.15, 0.14),
  hp("front", "lca_outer", "right", 0, -0.58, 0.13),
  hp("front", "lca_outer", "left", 0, 0.58, 0.13),
  hp("front", "uca_outer", "left", 0, 0.55, 0.3),
];
const MOVED: Hardpoint[] = [
  hp("rear", "uca_outer", "left", -1.6, 0.52, 0.3),
  hp("front", "tie_inner", "left", 0.08, 0.15, 0.14 + 0.00005), // 0.05 mm: below the threshold
  hp("front", "lca_outer", "right", 0, -0.58, 0.13, true),
  hp("front", "lca_outer", "left", 0, 0.58, 0.142), // +12 mm in z
  hp("front", "uca_outer", "left", 0.003, 0.55, 0.304), // 3 mm in x, 4 mm in z
];

describe("buildHardpointRows", () => {
  const rows = buildHardpointRows({ a: REF, b: MOVED }, ["a", "b"], "a");

  it("computes mm positions and the vector delta", () => {
    const uca = rows.find((r) => r.key === "front|uca_outer|left")!;
    expect(uca.values["a"]).toEqual([0, 550, 300]);
    expect(uca.deltas["b"]!.d[0]).toBeCloseTo(3, 9);
    expect(uca.deltas["b"]!.mag).toBeCloseTo(5, 9); // 3-4-5
    expect(uca.deltas["a"]).toBeUndefined(); // no delta for the reference
  });

  it("puts differing rows first inside each axle, then alphabetical, left before right", () => {
    expect(rows.map((r) => r.key)).toEqual([
      "front|lca_outer|left", // differs
      "front|uca_outer|left", // differs
      "front|lca_outer|right", // same position, only flagged overridden
      "front|tie_inner|left", // 0.05 mm: not differing
      "rear|uca_outer|left",
    ]);
    expect(rows.map((r) => r.differs)).toEqual([true, true, false, false, false]);
  });

  it("marks right points from right_overrides", () => {
    expect(rows.find((r) => r.key === "front|lca_outer|right")!.overridden).toBe(true);
    expect(rows.find((r) => r.key === "front|lca_outer|left")!.overridden).toBe(false);
  });

  it("uses null for a revision without hardpoints and never flags it as differing", () => {
    const r = buildHardpointRows({ a: REF, b: null }, ["a", "b"], "a");
    expect(r.every((x) => x.values["b"] === null && x.deltas["b"] === null && !x.differs)).toBe(true);
    expect(r).toHaveLength(REF.length);
  });

  it("counts a point missing from the reference as differing", () => {
    const r = buildHardpointRows({ a: REF.slice(1), b: REF }, ["a", "b"], "a");
    const added = r.find((x) => x.key === "rear|uca_outer|left")!;
    expect(added.differs).toBe(true);
    expect(added.values["a"]).toBeNull();
  });
});

describe("filterRows", () => {
  const rows = buildHardpointRows({ a: REF, b: MOVED }, ["a", "b"], "a");

  it("keeps only differing rows", () => {
    expect(filterRows(rows, { differingOnly: true, text: "" }).map((r) => r.key)).toEqual([
      "front|lca_outer|left",
      "front|uca_outer|left",
    ]);
  });

  it("matches the point name case-insensitively and combines with the differing filter", () => {
    expect(filterRows(rows, { differingOnly: false, text: " UCA_" }).map((r) => r.key)).toEqual([
      "front|uca_outer|left",
      "rear|uca_outer|left",
    ]);
    expect(filterRows(rows, { differingOnly: true, text: "uca" }).map((r) => r.key)).toEqual(["front|uca_outer|left"]);
    expect(filterRows(rows, { differingOnly: true, text: "nothing" })).toEqual([]);
  });
});

describe("changedSceneNames", () => {
  it("lists scene names of the points that moved more than 0.1 mm", () => {
    expect(changedSceneNames(MOVED, REF).sort()).toEqual(["FL:lca_outer", "FL:uca_outer"]);
  });

  it("is empty for the reference itself and for unreadable revisions", () => {
    expect(changedSceneNames(REF, REF)).toEqual([]);
    expect(changedSceneNames(null, REF)).toEqual([]);
    expect(changedSceneNames(MOVED, null)).toEqual([]);
  });

  it("maps axle and side to the corner id", () => {
    expect(sceneName(hp("rear", "x", "right", 0, 0, 0))).toBe("RR:x");
    expect(sceneName(hp("front", "x", "right", 0, 0, 0))).toBe("FR:x");
  });
});
