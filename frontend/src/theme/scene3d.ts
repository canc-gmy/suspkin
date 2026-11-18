import { themeScene, type PlotlyObject, type Tokens } from "./plotlyTemplate";

export type Vec3 = [number, number, number];
export interface Bounds {
  min: Vec3;
  max: Vec3;
}
export interface Camera {
  eye?: { x: number; y: number; z: number };
  center?: { x: number; y: number; z: number };
  up?: { x: number; y: number; z: number };
  projection?: { type: string };
}

/** Constant: the 3D camera and axes keep their user state across data and theme changes. */
export const SCENE_UIREVISION = "scene3d";
export const DEFAULT_CAMERA: Camera = {
  eye: { x: 1.6, y: -1.6, z: 0.9 },
  center: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 0, z: 1 },
};
export const GHOST_OPACITY = 0.4;
export const GHOST_WIDTH = 1;
const MARGIN = 0.1;
const AXES = ["x", "y", "z"] as const;

const BYTES: Record<string, { size: number; read: (v: DataView, o: number) => number }> = {
  f8: { size: 8, read: (v, o) => v.getFloat64(o, true) },
  f4: { size: 4, read: (v, o) => v.getFloat32(o, true) },
  i4: { size: 4, read: (v, o) => v.getInt32(o, true) },
  i2: { size: 2, read: (v, o) => v.getInt16(o, true) },
  i1: { size: 1, read: (v, o) => v.getInt8(o) },
};

/** Numbers of a Plotly coordinate array: a plain list or plotly.py's `{dtype, bdata}` encoding. */
export function coordValues(a: unknown): number[] {
  if (Array.isArray(a)) return a.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const enc = a as { dtype?: unknown; bdata?: unknown } | null;
  if (enc && typeof enc.dtype === "string" && typeof enc.bdata === "string") {
    const spec = BYTES[enc.dtype];
    if (!spec) return [];
    const bin = atob(enc.bdata);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    const view = new DataView(buf.buffer);
    const out: number[] = [];
    for (let o = 0; o + spec.size <= buf.length; o += spec.size) {
      const v = spec.read(view, o);
      if (Number.isFinite(v)) out.push(v);
    }
    return out;
  }
  return [];
}

/** Bounding box of every trace coordinate of a scene figure (null when there are none). */
export function sceneBounds(fig: { data?: unknown } | null | undefined): Bounds | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const tr of (fig?.data as PlotlyObject[] | undefined) ?? []) {
    AXES.forEach((k, i) => {
      for (const v of coordValues(tr[k])) {
        if (v < min[i]!) min[i] = v;
        if (v > max[i]!) max[i] = v;
      }
    });
  }
  return min.every(Number.isFinite) && max.every(Number.isFinite) ? { min, max } : null;
}

export interface SceneFrame {
  ranges: [number, number][];
  /** Axis lengths relative to the longest one, for `aspectmode: "manual"`. */
  aspect: { x: number; y: number; z: number };
}

/** Fixed axis ranges (bounding box plus 10% margin per axis) and the matching aspect ratio. */
export function sceneFrame(b: Bounds): SceneFrame {
  const ranges = AXES.map((_, i): [number, number] => {
    const lo = b.min[i]!;
    const hi = b.max[i]!;
    const pad = Math.max(hi - lo, 1e-6) * MARGIN;
    return [lo - pad, hi + pad];
  });
  const len = ranges.map(([lo, hi]) => hi - lo);
  const top = Math.max(...len);
  return { ranges, aspect: { x: len[0]! / top, y: len[1]! / top, z: len[2]! / top } };
}

/** Line traces of `data` restyled as a reference ghost: thin muted, translucent, inert. */
export function ghostTraces(data: PlotlyObject[], t: Pick<Tokens, "inkMuted">): PlotlyObject[] {
  return data
    .filter((tr) => tr["type"] === "scatter3d" && String(tr["mode"] ?? "").includes("lines"))
    .map((tr) => ({
      type: "scatter3d",
      mode: "lines",
      x: tr["x"],
      y: tr["y"],
      z: tr["z"],
      line: { color: t.inkMuted, width: GHOST_WIDTH },
      opacity: GHOST_OPACITY,
      showlegend: false,
      hoverinfo: "skip",
    }));
}

/**
 * The camera to put in the layout: the user's last camera (captured from `plotly_relayout`), or
 * the default on the first render and on an explicit reset. It is never taken from the server
 * figure, so a pose change cannot move it (Plotly alone drops it when the layout omits it).
 */
export function cameraForLayout(o: { reset: boolean; saved: Camera | null }): Camera {
  return o.reset ? DEFAULT_CAMERA : (o.saved ?? DEFAULT_CAMERA);
}

/** Camera carried by a `plotly_relayout` event, or null if the event did not touch it. */
export function cameraFromRelayout(ev: Record<string, unknown> | undefined): Camera | null {
  const c = ev?.["scene.camera"];
  return c && typeof c === "object" ? (c as Camera) : null;
}

export interface SceneFigureInput {
  figure: { data?: unknown; layout?: unknown };
  /** Reference scene at the same pose, or null when the ghost is off. */
  ghost: { data?: unknown } | null;
  tokens: Tokens;
  frame: SceneFrame | null;
  camera: Camera;
}

/** Themed 3D figure: styling from the tokens, fixed ranges and aspect, optional ghost behind. */
export function buildSceneFigure(i: SceneFigureInput): { data: PlotlyObject[]; layout: PlotlyObject } {
  const t = i.tokens;
  const themed = themeScene(i.figure, t);
  const sc = themed.layout["scene"] as PlotlyObject;
  const axisOf = (name: "xaxis" | "yaxis" | "zaxis", idx: number): PlotlyObject => ({
    ...(sc[name] as PlotlyObject),
    ...(i.frame ? { range: i.frame.ranges[idx], autorange: false } : {}),
  });
  const scene: PlotlyObject = {
    ...sc,
    xaxis: axisOf("xaxis", 0),
    yaxis: axisOf("yaxis", 1),
    zaxis: axisOf("zaxis", 2),
    aspectmode: i.frame ? "manual" : "data",
    ...(i.frame ? { aspectratio: i.frame.aspect } : {}),
    uirevision: SCENE_UIREVISION,
  };
  delete scene["camera"];
  scene["camera"] = i.camera;
  const ghost = i.ghost ? ghostTraces((i.ghost.data as PlotlyObject[] | undefined) ?? [], t) : [];
  return {
    data: [...ghost, ...themed.data],
    layout: { ...themed.layout, uirevision: SCENE_UIREVISION, autosize: true, paper_bgcolor: t.sheet, scene },
  };
}
