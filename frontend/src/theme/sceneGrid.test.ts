import { describe, expect, it } from "vitest";
import type { Tokens } from "./plotlyTemplate";
import { buildSceneFigure, cameraForLayout, fitCamera, sceneFrame, type Bounds } from "./scene3d";
import {
  buildGridTraces,
  buildPlaneAnnotations,
  buildPlaneTraces,
  clampPlaneOffset,
  FADE_BANDS,
  gridGeometry,
  LABEL_GAP,
  LABEL_REACH,
  niceStep,
  planeGeometry,
  PLANE_MARGIN,
  planeRange,
  worldFrame,
  WORLD_SPAN,
} from "./sceneGrid";
import { defaultPlaneState, loadPlane, PLANE_KEY, savePlane } from "../lib/planeMenu";

const TOKENS = {
  sheet: "#fff",
  sheetRaised: "#eee",
  ink: "#111",
  inkMuted: "#777",
  grid: "#ccc",
  gridMinor: "#ddd",
  rule: "#bbb",
  layoutBlue: "#00f",
  fontUi: "Barlow",
  rev: {},
} as Tokens;

const CAR: Bounds = { min: [-1.6, -0.7, -0.05], max: [0.2, 0.7, 0.45] };
const FIGURE = {
  data: [{ type: "scatter3d", mode: "lines", x: [-1.5, 0.1], y: [-0.6, 0.6], z: [0, 0.4] }],
  layout: { scene: {} },
};

describe("world box and grid", () => {
  it("is a cube concentric with the car, at least 3x its footprint", () => {
    const w = worldFrame(CAR);
    const edge = w.ranges[0]![1] - w.ranges[0]![0];
    expect(edge).toBeGreaterThanOrEqual(WORLD_SPAN * 1.8 - 1e-9);
    expect(edge).toBeGreaterThanOrEqual(WORLD_SPAN * 1.4);
    w.ranges.forEach(([lo, hi], i) => {
      expect(hi - lo).toBeCloseTo(edge);
      expect((lo + hi) / 2).toBeCloseTo((CAR.min[i]! + CAR.max[i]!) / 2);
    });
    expect(w.aspect).toEqual({ x: 1, y: 1, z: 1 });
    expect(w.scale).toBeCloseTo(1.8 / edge);
  });

  it("picks 1-2-5 steps", () => {
    expect(niceStep(0.54)).toBe(0.5);
    expect(niceStep(0.2)).toBe(0.2);
    expect(niceStep(1)).toBe(1);
  });

  it("spaces lines by the minor step and spans the world box", () => {
    const w = worldFrame(CAR);
    const g = gridGeometry(w, CAR);
    expect(g.minor).toBeCloseTo(g.major / 5);
    const all = [...g.segments.minor, ...g.segments.major].flat();
    const vertical = all.filter((s) => s.a[0] === s.b[0]);
    const xs = [...new Set(vertical.map((s) => Math.round(s.a[0] / g.minor)))].sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i++) expect(xs[i]! - xs[i - 1]!).toBe(1);
    expect(xs[0]! * g.minor).toBeGreaterThanOrEqual(w.ranges[0]![0] - 1e-9);
    expect(xs[xs.length - 1]! * g.minor).toBeLessThanOrEqual(w.ranges[0]![1] + 1e-9);
    expect(xs.length * g.minor).toBeGreaterThan(WORLD_SPAN * 1.8 * 0.9);
    // lines run the whole edge
    const ys = vertical.flatMap((s) => [s.a[1], s.b[1]]);
    expect(Math.min(...ys)).toBeCloseTo(w.ranges[1]![0]);
    expect(Math.max(...ys)).toBeCloseTo(w.ranges[1]![1]);
  });

  it("fades with distance: inner segments use the first band, far ones the last", () => {
    const g = gridGeometry(worldFrame(CAR), CAR);
    const count = (b: number) => g.segments.minor[b]!.length + g.segments.major[b]!.length;
    expect(count(0)).toBeGreaterThan(0);
    expect(count(FADE_BANDS.length - 1)).toBeGreaterThan(0);
    const opacities = FADE_BANDS.map((b) => b.opacity);
    expect([...opacities].sort((a, b) => b - a)).toEqual(opacities);
  });

  it("puts the tick labels in metres outside the car box, within the framed extent", () => {
    const g = gridGeometry(worldFrame(CAR), CAR);
    const gap = LABEL_GAP * 1.8;
    expect(g.row).toBeCloseTo(CAR.min[1] - gap);
    expect(g.column).toBeCloseTo(CAR.max[0] + gap);
    const xl = g.labels.filter((l) => l.y === g.row);
    const yl = g.labels.filter((l) => l.x === g.column);
    expect(xl.length).toBeGreaterThan(2);
    expect(yl.length).toBeGreaterThan(2);
    for (const l of g.labels) {
      const inside = l.x >= CAR.min[0] && l.x <= CAR.max[0] && l.y >= CAR.min[1] && l.y <= CAR.max[1];
      expect(inside).toBe(false);
      expect(l.text).toMatch(/^-?\d+\.\d$/);
    }
    for (const l of xl) {
      expect(l.x).toBeGreaterThanOrEqual(CAR.min[0] - LABEL_REACH * 1.8 - 1e-9);
      expect(l.x).toBeLessThanOrEqual(CAR.max[0] + LABEL_REACH * 1.8 + 1e-9);
    }
    for (const l of yl) {
      expect(l.y).toBeGreaterThanOrEqual(CAR.min[1] - LABEL_REACH * 1.4 - 1e-9);
      expect(l.y).toBeLessThanOrEqual(CAR.max[1] + LABEL_REACH * 1.4 + 1e-9);
    }
  });

  it("titles each row next to its ticks, just outside them and inside the framed extent", () => {
    const g = gridGeometry(worldFrame(CAR), CAR);
    expect(g.titles.map((t) => t.text)).toEqual(["x [m]", "y [m]"]);
    const [tx, ty] = g.titles as [(typeof g.titles)[0], (typeof g.titles)[0]];
    expect(tx.y).toBeLessThan(g.row);
    expect(tx.y).toBeGreaterThan(CAR.min[1] - 0.45 * 1.8);
    expect(tx.x).toBeCloseTo((CAR.min[0] + CAR.max[0]) / 2);
    expect(ty.x).toBeGreaterThan(g.column);
    expect(ty.x).toBeLessThan(CAR.max[0] + 0.45 * 1.8);
    expect(ty.y).toBeCloseTo(0);
  });

  it("builds inert traces in the grid tokens", () => {
    const tr = buildGridTraces(worldFrame(CAR), CAR, TOKENS);
    expect(tr.length).toBeGreaterThan(2);
    for (const t of tr) expect(t).toMatchObject({ hoverinfo: "skip", showlegend: false });
    const colours = tr.filter((t) => t["mode"] === "lines").map((t) => (t["line"] as { color: string }).color);
    expect(new Set(colours)).toEqual(new Set(["#ddd", "#ccc"]));
  });
});

describe("reference plane", () => {
  it("covers the car box with margin, at the offset along the normal", () => {
    const cases = [
      ["xy", 2, [0, 1]],
      ["xz", 1, [0, 2]],
      ["yz", 0, [1, 2]],
    ] as const;
    for (const [kind, n, inPlane] of cases) {
      const g = planeGeometry(kind, 120, CAR);
      expect(g.corners).toHaveLength(4);
      expect(g.outline).toHaveLength(5);
      expect(g.outline[4]).toEqual(g.outline[0]);
      for (const c of g.corners) expect(c[n]).toBeCloseTo(0.12);
      for (const k of inPlane) {
        const vals = g.corners.map((c) => c[k]);
        expect(Math.min(...vals)).toBeLessThan(CAR.min[k]!);
        expect(Math.max(...vals)).toBeGreaterThan(CAR.max[k]!);
      }
      expect(g.label.text).toBe(`${"xyz"[n]} = 120 mm`);
    }
  });

  it("labels near the corner nearest the default camera, half the margin outside the car box", () => {
    const half = (k: number) => ((CAR.max[k]! - CAR.min[k]!) * PLANE_MARGIN) / 2;
    // near end per axis: high x, low y, high z (the top, clear of the ground tick labels)
    const near = (k: number) => (k === 1 ? CAR.min[1]! - half(1) : CAR.max[k]! + half(k));
    for (const [kind, n] of [["xy", 2], ["xz", 1], ["yz", 0]] as const) {
      const g = planeGeometry(kind, 0, CAR);
      // the horizontal plane's near corner is the lowest point on screen: its label takes the middle of the near edge
      const mid = (CAR.min[0]! + CAR.max[0]!) / 2;
      for (const k of [0, 1, 2].filter((i) => i !== n)) expect(g.label.at[k]).toBeCloseTo(kind === "xy" && k === 0 ? mid : near(k));
      // still inside the plane rectangle, and on the camera side of its centre
      for (const k of [0, 1, 2].filter((i) => i !== n)) {
        const vals = g.corners.map((c) => c[k]!);
        expect(g.label.at[k]!).toBeGreaterThan(Math.min(...vals));
        expect(g.label.at[k]!).toBeLessThan(Math.max(...vals));
      }
    }
  });

  it("sets the plane label as a 14 px ink scene annotation at the label anchor", () => {
    expect(buildPlaneAnnotations("none", 0, CAR, TOKENS)).toEqual([]);
    const [lab] = buildPlaneAnnotations("xy", 120, CAR, TOKENS);
    expect(lab).toMatchObject({ text: "z = 120 mm", showarrow: false, font: { size: 14, color: "#111", family: "Barlow" } });
    const at = planeGeometry("xy", 120, CAR).label.at;
    expect([lab!["x"], lab!["y"], lab!["z"]]).toEqual(at);
  });

  it("takes the slider range from the car box, plus 10 %", () => {
    expect(planeRange("xy", CAR)).toEqual({ min: -100, max: 500 });
    expect(planeRange("xz", CAR)).toEqual({ min: -840, max: 840 });
    expect(planeRange("yz", CAR)).toEqual({ min: -1780, max: 380 });
    expect(clampPlaneOffset("xy", 9999, CAR)).toBe(500);
    expect(clampPlaneOffset("xy", -9999, CAR)).toBe(-100);
    expect(clampPlaneOffset("xy", 12.4, CAR)).toBe(12);
  });

  it("draws surface and outline in the layout blue, nothing for none", () => {
    expect(buildPlaneTraces("none", 0, CAR, TOKENS)).toEqual([]);
    const tr = buildPlaneTraces("xz", 0, CAR, TOKENS);
    expect(tr.map((t) => t["type"])).toEqual(["mesh3d", "scatter3d"]);
    expect(tr[0]).toMatchObject({ color: "#00f", hoverinfo: "skip" });
    expect(tr[0]!["opacity"] as number).toBeLessThan(0.3);
    expect(buildPlaneAnnotations("xz", 0, CAR, TOKENS)[0]!["text"]).toBe("y = 0 mm");
  });
});

describe("camera is independent of the grid and the plane", () => {
  const frame = sceneFrame({ min: [-1.5, -0.6, 0], max: [0.1, 0.6, 0.4] });
  const base = { figure: FIGURE, ghost: null, tokens: TOKENS, frame, camera: cameraForLayout({ reset: false, saved: null }) };

  it("keeps camera, ranges and aspect when the plane or offset changes", () => {
    const a = buildSceneFigure(base);
    const b = buildSceneFigure({ ...base, plane: { kind: "xy", offsetMm: 120 } });
    const c = buildSceneFigure({ ...base, plane: { kind: "yz", offsetMm: -300 } });
    const sc = (f: typeof a) => f.layout["scene"] as Record<string, Record<string, unknown>>;
    for (const f of [b, c]) {
      expect(sc(f)["camera"]).toEqual(sc(a)["camera"]);
      expect(sc(f)["aspectratio"]).toEqual(sc(a)["aspectratio"]);
      for (const ax of ["xaxis", "yaxis", "zaxis"]) expect(sc(f)[ax]!["range"]).toEqual(sc(a)[ax]!["range"]);
    }
    expect(b.data.length).toBe(a.data.length + 2);
  });

  it("hides the Plotly axes and uses the world box", () => {
    const sc = buildSceneFigure(base).layout["scene"] as Record<string, Record<string, unknown>>;
    expect(sc["xaxis"]).toMatchObject({ visible: false, showbackground: false });
    expect(sc["xaxis"]!["range"]).toEqual(frame.world!.ranges[0]);
    expect(sc["aspectratio"]).toEqual({ x: 1, y: 1, z: 1 });
  });

  it("frames the car, not the world: the eye is nearer than for the whole world box", () => {
    const canvas = { width: 1440, height: 800 };
    const car = fitCamera(frame, canvas);
    const whole = fitCamera({ ...frame, world: undefined, aspect: { x: 1, y: 1, z: 1 } }, canvas);
    const d = (c: typeof car) => Math.hypot(c.eye!.x - c.center!.x, c.eye!.y - c.center!.y, c.eye!.z - c.center!.z);
    // the car box is about 1/WORLD_SPAN of the world box, so the framed camera sits much closer in scene units
    expect(d(car)).toBeLessThan(d(whole) / 2);
    // and equals what the car-only framing would give, in world units
    const own = fitCamera({ ...frame, world: undefined }, canvas);
    expect(d(car)).toBeCloseTo(d(own) * frame.world!.scale, 3);
  });
});

describe("plane persistence", () => {
  it("falls back to the default on unreadable storage", () => {
    expect(loadPlane()).toEqual(defaultPlaneState());
    expect(() => savePlane(defaultPlaneState())).not.toThrow();
    expect(PLANE_KEY).toBe("suspkin.plane");
  });
});
