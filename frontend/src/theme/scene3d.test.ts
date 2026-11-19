import { describe, expect, it } from "vitest";
import type { Tokens } from "./plotlyTemplate";
import {
  buildSceneFigure,
  cameraForLayout,
  fitCamera,
  cameraFromRelayout,
  coordValues,
  DEFAULT_CAMERA,
  ghostTraces,
  posedHardpoints,
  sceneBounds,
  sceneFrame,
} from "./scene3d";

const TOKENS = {
  sheet: "#fff",
  sheetRaised: "#eee",
  ink: "#111",
  inkMuted: "#777",
  grid: "#ccc",
  gridMinor: "#ddd",
  rule: "#bbb",
  layoutBlue: "#00f",
  danger: "#f00",
  regionDomain: "#0a0",
  regionRequired: "#0aa",
  regionKeepout: "#a05",
  fontUi: "Barlow",
  rev: {},
} as Tokens;

function f8(values: number[]): { dtype: string; bdata: string } {
  const buf = new DataView(new ArrayBuffer(values.length * 8));
  values.forEach((v, i) => buf.setFloat64(i * 8, v, true));
  return { dtype: "f8", bdata: btoa(String.fromCharCode(...new Uint8Array(buf.buffer))) };
}

const FIGURE = {
  data: [
    { type: "scatter3d", mode: "lines", x: [0, 1, null, 2], y: [0, 0.5, null, 1], z: [0, 0.2, null, 0.4], line: { color: "#1f77b4", width: 3 }, name: "wishbones" },
    { type: "mesh3d", x: f8([-1, 3]), y: f8([-2, 2]), z: f8([0, 0]) },
    { type: "scatter3d", mode: "markers", x: [0.5], y: [0.5], z: [0.5], marker: { color: "#222222", size: 3 } },
  ],
  layout: { scene: { aspectmode: "data", camera: { eye: { x: 9, y: 9, z: 9 } } } },
};

describe("scene bounds and frame", () => {
  it("decodes plotly.py binary arrays and skips nulls", () => {
    expect(coordValues(f8([1.5, -2]))).toEqual([1.5, -2]);
    expect(coordValues([1, null, 2])).toEqual([1, 2]);
  });

  it("takes the bounding box over every trace", () => {
    expect(sceneBounds(FIGURE)).toEqual({ min: [-1, -2, 0], max: [3, 2, 0.5] });
    expect(sceneBounds({ data: [] })).toBeNull();
  });

  it("adds a 10% margin per axis and derives the aspect from the ranges", () => {
    const f = sceneFrame({ min: [-1, -2, 0], max: [3, 2, 0.5] });
    expect(f.ranges[0]![0]).toBeCloseTo(-1.4);
    expect(f.ranges[0]![1]).toBeCloseTo(3.4);
    expect(f.ranges[2]![1]).toBeCloseTo(0.55);
    expect(f.aspect.x).toBeCloseTo(1);
    expect(f.aspect.y).toBeCloseTo(4.8 / 4.8);
    expect(f.aspect.z).toBeCloseTo(0.6 / 4.8);
  });

  it("does not collapse a flat axis", () => {
    const f = sceneFrame({ min: [0, 0, 1], max: [1, 1, 1] });
    expect(f.ranges[2]![1]).toBeGreaterThan(f.ranges[2]![0]);
    expect(f.aspect.z).toBeGreaterThan(0);
  });
});

describe("reference ghost", () => {
  it("keeps only line traces, thin, muted, translucent and inert", () => {
    const g = ghostTraces(FIGURE.data, TOKENS);
    expect(g).toHaveLength(1);
    expect(g[0]).toMatchObject({
      mode: "lines",
      line: { color: "#777", width: 1 },
      opacity: 0.4,
      showlegend: false,
      hoverinfo: "skip",
    });
  });

  it("is drawn behind the selected revision", () => {
    const fig = buildSceneFigure({ figure: FIGURE, ghost: FIGURE, tokens: TOKENS, frame: null, camera: DEFAULT_CAMERA });
    expect(fig.data).toHaveLength(1 + FIGURE.data.length);
    expect(fig.data[0]).toMatchObject({ hoverinfo: "skip" });
    expect(fig.data[1]).toMatchObject({ name: "wishbones" });
  });
});

describe("camera preservation", () => {
  const saved = { eye: { x: 1, y: 2, z: 3 } };

  it("uses the saved camera, the default only at the start or on reset", () => {
    expect(cameraForLayout({ reset: false, saved: null })).toEqual(DEFAULT_CAMERA);
    expect(cameraForLayout({ reset: false, saved })).toBe(saved);
    expect(cameraForLayout({ reset: true, saved })).toEqual(DEFAULT_CAMERA);
  });

  it("reads the camera from relayout events", () => {
    expect(cameraFromRelayout({ "scene.camera": saved })).toBe(saved);
    expect(cameraFromRelayout({ "scene.camera.eye": 1 })).toBeNull();
    expect(cameraFromRelayout(undefined)).toBeNull();
  });

  it("ignores the server camera and uses constant uirevisions", () => {
    const frame = sceneFrame(sceneBounds(FIGURE)!);
    const a = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame, camera: DEFAULT_CAMERA });
    const scene = a.layout["scene"] as Record<string, unknown>;
    expect(scene["camera"]).toEqual(DEFAULT_CAMERA);
    expect(scene["aspectmode"]).toBe("manual");
    expect(scene["uirevision"]).toBe(a.layout["uirevision"]);
    const b = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: { ...TOKENS, ink: "#eee" }, frame, camera: DEFAULT_CAMERA });
    expect(b.layout["uirevision"]).toBe(a.layout["uirevision"]);
    const withCam = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame, camera: saved });
    expect((withCam.layout["scene"] as Record<string, unknown>)["camera"]).toBe(saved);
  });
});

describe("rings and surface", () => {
  it("adds one open 6 px ring trace in the layout blue at the given posed points", () => {
    const pts: [number, number, number][] = [[0.1, 0.2, 0.3], [1, 2, 3]];
    const fig = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame: null, camera: DEFAULT_CAMERA, highlight: pts });
    const ring = fig.data[fig.data.length - 1] as Record<string, unknown>;
    expect(ring["x"]).toEqual([0.1, 1]);
    expect(ring["z"]).toEqual([0.3, 3]);
    const m = ring["marker"] as Record<string, unknown>;
    expect([m["symbol"], m["size"], m["color"]]).toEqual(["circle-open", 6, "#00f"]);
    const none = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame: null, camera: DEFAULT_CAMERA, highlight: [] });
    expect(none.data).toHaveLength(FIGURE.data.length);
  });

  it("reads posed hardpoints by name and drops malformed entries", () => {
    expect(posedHardpoints({ hardpoints: { "FL:a": [1, 2, 3], "FL:b": [1, 2], "FL:c": [1, "x", 3] } })).toEqual({ "FL:a": [1, 2, 3] });
    expect(posedHardpoints(null)).toEqual({});
  });

  it("uses one surface colour for paper and scene, and keeps the legend in a strip above the scene", () => {
    const fig = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame: null, camera: DEFAULT_CAMERA });
    const scene = fig.layout["scene"] as Record<string, unknown>;
    expect(fig.layout["paper_bgcolor"]).toBe(TOKENS.sheetRaised);
    expect(scene["bgcolor"]).toBe(TOKENS.sheetRaised);
    const legend = fig.layout["legend"] as Record<string, unknown>;
    expect([legend["x"], legend["xanchor"], legend["yanchor"], legend["orientation"]]).toEqual([0, "left", "bottom", "h"]);
    expect(fig.layout["showlegend"]).toBe(true);
    const bare = buildSceneFigure({ figure: FIGURE, ghost: null, tokens: TOKENS, frame: null, camera: DEFAULT_CAMERA, legend: false });
    expect(bare.layout["showlegend"]).toBe(false);
    expect((bare.layout["margin"] as Record<string, number>)["t"]).toBeLessThan((fig.layout["margin"] as Record<string, number>)["t"]!);
  });
});

describe("default framing", () => {
  const frame = { ranges: [], aspect: { x: 1, y: 0.4, z: 0.25 } } as unknown as Parameters<typeof fitCamera>[0];
  const dist = (c: ReturnType<typeof fitCamera>) =>
    Math.hypot(c.eye!.x - c.center!.x, c.eye!.y - c.center!.y, c.eye!.z - c.center!.z);

  it("falls back to the default camera without a frame or a measured canvas", () => {
    expect(fitCamera(null, { width: 800, height: 600 })).toEqual(DEFAULT_CAMERA);
    expect(fitCamera(frame, null)).toEqual(DEFAULT_CAMERA);
    expect(fitCamera(frame, { width: 0, height: 600 })).toEqual(DEFAULT_CAMERA);
  });

  it("keeps the viewing direction: the eye stays on the default ray from the look-at point", () => {
    const c = fitCamera(frame, { width: 1200, height: 600 });
    const d0 = DEFAULT_CAMERA.eye!;
    const n0 = Math.hypot(d0.x, d0.y, d0.z);
    const v = { x: c.eye!.x - c.center!.x, y: c.eye!.y - c.center!.y, z: c.eye!.z - c.center!.z };
    const n = Math.hypot(v.x, v.y, v.z);
    expect(v.x / n).toBeCloseTo(d0.x / n0, 9);
    expect(v.y / n).toBeCloseTo(d0.y / n0, 9);
    expect(v.z / n).toBeCloseTo(d0.z / n0, 9);
    expect(c.up).toEqual(DEFAULT_CAMERA.up);
  });

  it("moves closer for a smaller fill, and farther for a narrower canvas", () => {
    const wide = fitCamera(frame, { width: 1200, height: 600 });
    const narrow = fitCamera(frame, { width: 300, height: 600 });
    expect(dist(narrow)).toBeGreaterThan(dist(wide));
    expect(dist(fitCamera(frame, { width: 1200, height: 600 }, 0.5))).toBeGreaterThan(dist(wide));
  });
});
