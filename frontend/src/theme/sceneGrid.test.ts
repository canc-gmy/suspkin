import { describe, expect, it } from "vitest";
import type { Tokens } from "./plotlyTemplate";
import { buildSceneFigure, cameraForLayout, fitCamera, sceneFrame, type Bounds } from "./scene3d";
import {
  buildGridTraces,
  buildPlaneTraces,
  clampPlaneOffset,
  FADE_BANDS,
  gridGeometry,
  LABEL_GAP,
  niceStep,
  planeGeometry,
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

  it("puts the tick labels in metres outside the car box", () => {
    const g = gridGeometry(worldFrame(CAR), CAR);
    const gap = LABEL_GAP * 1.8;
    expect(g.row).toBeLessThan(CAR.min[1] - gap + 1e-9);
    expect(g.column).toBeGreaterThan(CAR.max[0] + gap - 1e-9);
    const xl = g.labels.filter((l) => l.y === g.row);
    const yl = g.labels.filter((l) => l.x === g.column);
    expect(xl.length).toBeGreaterThan(2);
    expect(yl.length).toBeGreaterThan(2);
    for (const l of g.labels) {
      const inside = l.x >= CAR.min[0] && l.x <= CAR.max[0] && l.y >= CAR.min[1] && l.y <= CAR.max[1];
      expect(inside).toBe(false);
      expect(l.text).toMatch(/^-?\d+\.\d$/);
    }
    expect(g.titles.map((t) => t.text)).toEqual(["x [m]", "y [m]"]);
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

  it("labels at a far corner of the outline", () => {
    const g = planeGeometry("xy", 0, CAR);
    const xs = g.corners.map((c) => c[0]);
    const ys = g.corners.map((c) => c[1]);
    expect(g.label.at[0]).toBe(Math.min(...xs));
    expect(g.label.at[1]).toBe(Math.max(...ys));
  });

  it("takes the slider range from the car box, plus 10 %", () => {
    expect(planeRange("xy", CAR)).toEqual({ min: -100, max: 500 });
    expect(planeRange("xz", CAR)).toEqual({ min: -840, max: 840 });
    expect(planeRange("yz", CAR)).toEqual({ min: -1780, max: 380 });
    expect(clampPlaneOffset("xy", 9999, CAR)).toBe(500);
    expect(clampPlaneOffset("xy", -9999, CAR)).toBe(-100);
    expect(clampPlaneOffset("xy", 12.4, CAR)).toBe(12);
  });

  it("draws surface, outline and label in the layout blue, nothing for none", () => {
    expect(buildPlaneTraces("none", 0, CAR, TOKENS)).toEqual([]);
    const tr = buildPlaneTraces("xz", 0, CAR, TOKENS);
    expect(tr.map((t) => t["type"])).toEqual(["mesh3d", "scatter3d", "scatter3d"]);
    expect(tr[0]).toMatchObject({ color: "#00f", hoverinfo: "skip" });
    expect(tr[0]!["opacity"] as number).toBeLessThan(0.3);
    expect(tr[2]!["text"]).toEqual(["y = 0 mm"]);
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
    expect(b.data.length).toBe(a.data.length + 3);
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
