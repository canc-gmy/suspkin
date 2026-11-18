import type { PlotlyObject, Tokens } from "./plotlyTemplate";
import type { Bounds, Vec3 } from "./scene3d";

// ---------------------------------------------------------------------------------------------
// World box: the large cube the Plotly scene is laid out in. The ground grid fills it; the car is
// a small box in its middle, and the default camera frames only the car.

/** World edge as a multiple of the car's longest horizontal side. */
export const WORLD_SPAN = 3;

export interface World {
  ranges: [number, number][];
  /** A cube: every axis has the same length, so one scene unit is the same on all axes. */
  aspect: { x: number; y: number; z: number };
  /** Car box edge (its longest side) over the world edge: the car's share of the scene box. */
  scale: number;
}

/** World box concentric with the (padded) car box, `WORLD_SPAN` times its longest horizontal side. */
export function worldFrame(car: Bounds): World {
  const len = car.min.map((lo, i) => car.max[i]! - lo);
  const edge = WORLD_SPAN * Math.max(len[0]!, len[1]!, 1e-6);
  const ranges = car.min.map((lo, i): [number, number] => {
    const c = (lo + car.max[i]!) / 2;
    return [c - edge / 2, c + edge / 2];
  });
  return { ranges, aspect: { x: 1, y: 1, z: 1 }, scale: Math.max(...len) / edge };
}

// ---------------------------------------------------------------------------------------------
// Ground grid

/** Minor lines per major interval. */
export const MINOR_PER_MAJOR = 5;
/** Target number of major intervals across the world box. */
const MAJOR_TARGET = 10;
/** Segments each grid line is cut into, so the fade can follow the distance along the line. */
const SEGMENTS = 8;
/** Fade bands: share of the world half-edge where each opacity starts (the car edge is at 1/3). */
export const FADE_BANDS: readonly { from: number; opacity: number }[] = [
  { from: 0, opacity: 1 },
  { from: 0.3, opacity: 0.7 },
  { from: 0.45, opacity: 0.4 },
  { from: 0.6, opacity: 0.15 },
];
/** Tick rows sit this far outside the car box, as a share of the car's longest horizontal side. */
export const LABEL_GAP = 0.15;
/** Axis titles sit this much further out than the tick row, same share. */
export const TITLE_GAP = 0.08;
/** Ticks are labelled only up to this share of the car's size beyond its box: what the default framing shows. */
export const LABEL_REACH = 0.2;
export const GRID_LABEL_SIZE = 12;
export const PLANE_LABEL_SIZE = 14;
export const GRID_UNIT = "m";

/** The 1, 2 or 5 times a power of ten (or 10) nearest to `raw`. */
export function niceStep(raw: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export interface GridSegment {
  a: [number, number];
  b: [number, number];
}

export interface GridGeometry {
  /** Ground height (z) the grid is drawn at. */
  z: number;
  major: number;
  minor: number;
  /** Grid segments in the xy plane, by line class and fade band (index into `FADE_BANDS`). */
  segments: { minor: GridSegment[][]; major: GridSegment[][] };
  /** Tick labels: position (x, y) and text. x labels sit on one row, y labels on one column. */
  labels: { x: number; y: number; text: string }[];
  /** Axis titles, one step outside the middle of each label row. */
  titles: { x: number; y: number; text: string }[];
  /** The row (y of the x labels) and the column (x of the y labels). */
  row: number;
  column: number;
}

function formatTick(v: number, step: number): string {
  const s = v.toFixed(step < 0.1 ? 2 : 1);
  return Number(s) === 0 ? s.replace("-", "") : s;
}

function bandOf(r: number): number {
  let band = 0;
  FADE_BANDS.forEach((b, i) => {
    if (r >= b.from) band = i;
  });
  return band;
}

/**
 * Ground grid over the world box, with major/minor lines that fade towards the edges and tick
 * labels (metres, 1 decimal) on two rows just outside the car box, so they stay clear of the points.
 */
export function gridGeometry(world: World, car: Bounds, z = 0): GridGeometry {
  const [xr, yr] = world.ranges as [[number, number], [number, number]];
  const edge = xr[1] - xr[0];
  const major = niceStep(edge / MAJOR_TARGET);
  const minor = major / MINOR_PER_MAJOR;
  const cx = (xr[0] + xr[1]) / 2;
  const cy = (yr[0] + yr[1]) / 2;
  const half = edge / 2;
  const segments: GridGeometry["segments"] = {
    minor: FADE_BANDS.map(() => []),
    major: FADE_BANDS.map(() => []),
  };
  const line = (vertical: boolean, v: number, lo: number, hi: number, isMajor: boolean) => {
    for (let s = 0; s < SEGMENTS; s++) {
      const t0 = lo + ((hi - lo) * s) / SEGMENTS;
      const t1 = lo + ((hi - lo) * (s + 1)) / SEGMENTS;
      const tm = (t0 + t1) / 2;
      const r = Math.max(Math.abs(vertical ? v - cx : tm - cx), Math.abs(vertical ? tm - cy : v - cy)) / half;
      const seg: GridSegment = vertical ? { a: [v, t0], b: [v, t1] } : { a: [t0, v], b: [t1, v] };
      segments[isMajor ? "major" : "minor"][bandOf(r)]!.push(seg);
    }
  };
  const eps = minor * 1e-6;
  const lines = (lo: number, hi: number, fn: (v: number, isMajor: boolean) => void) => {
    const k0 = Math.ceil((lo + eps) / minor);
    const k1 = Math.floor((hi - eps) / minor);
    for (let k = k0; k <= k1; k++) fn(k * minor, k % MINOR_PER_MAJOR === 0);
  };
  lines(xr[0], xr[1], (v, m) => line(true, v, yr[0], yr[1], m));
  lines(yr[0], yr[1], (v, m) => line(false, v, xr[0], xr[1], m));

  const side = Math.max(car.max[0] - car.min[0], car.max[1] - car.min[1]);
  const gap = LABEL_GAP * side;
  // the viewer looks from +x, -y, at the near corner: the rows go on the near sides (low y, high x),
  // close to the car, within the framed area
  const row = car.min[1] - gap;
  const column = car.max[0] + gap;
  const labels: GridGeometry["labels"] = [];
  const titles: GridGeometry["titles"] = [];
  const ticks = (lo: number, hi: number): number[] => {
    const out: number[] = [];
    for (let k = Math.ceil((lo + eps) / major); k <= Math.floor((hi - eps) / major); k++) out.push(k * major);
    return out;
  };
  const reach = (k: number) => LABEL_REACH * (car.max[k]! - car.min[k]!);
  const xt = ticks(Math.max(xr[0], car.min[0] - reach(0)), Math.min(xr[1], car.max[0] + reach(0)));
  const yt = ticks(Math.max(yr[0], car.min[1] - reach(1)), Math.min(yr[1], car.max[1] + reach(1)));
  const titleGap = TITLE_GAP * side;
  for (const v of xt) labels.push({ x: v, y: row, text: formatTick(v, major) });
  for (const v of yt) labels.push({ x: column, y: v, text: formatTick(v, major) });
  // titles sit next to the ticks nearest the car centre, one step outside their row or column
  if (xt.length > 0) titles.push({ x: (car.min[0] + car.max[0]) / 2, y: row - titleGap, text: `x [${GRID_UNIT}]` });
  if (yt.length > 0) titles.push({ x: column + titleGap, y: (car.min[1] + car.max[1]) / 2, text: `y [${GRID_UNIT}]` });
  return { z, major, minor, segments, labels, titles, row, column };
}

function segmentsTrace(segs: readonly GridSegment[], z: number, color: string, width: number, opacity: number): PlotlyObject {
  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  for (const s of segs) {
    x.push(s.a[0], s.b[0], null);
    y.push(s.a[1], s.b[1], null);
  }
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z: x.map((v) => (v === null ? null : z)),
    line: { color, width },
    opacity,
    showlegend: false,
    hoverinfo: "skip",
  };
}

function textTrace(
  pts: readonly { x: number; y: number; text: string }[],
  z: number,
  color: string,
  family: string,
): PlotlyObject {
  return {
    type: "scatter3d",
    mode: "text",
    x: pts.map((p) => p.x),
    y: pts.map((p) => p.y),
    z: pts.map(() => z),
    text: pts.map((p) => p.text),
    textposition: "middle center",
    textfont: { family, size: GRID_LABEL_SIZE, color },
    showlegend: false,
    hoverinfo: "skip",
  };
}

/** Grid traces (drawn first, behind the car): minor, major, tick labels and axis titles. */
export function buildGridTraces(world: World, car: Bounds, t: Tokens, z = 0): PlotlyObject[] {
  const g = gridGeometry(world, car, z);
  const out: PlotlyObject[] = [];
  for (const [cls, colour] of [
    ["minor", t.gridMinor],
    ["major", t.grid],
  ] as const) {
    g.segments[cls].forEach((segs, band) => {
      if (segs.length > 0) out.push(segmentsTrace(segs, z, colour, 1, FADE_BANDS[band]!.opacity));
    });
  }
  if (g.labels.length > 0) out.push(textTrace(g.labels, z, t.inkMuted, t.fontUi));
  if (g.titles.length > 0) out.push(textTrace(g.titles, z, t.inkMuted, t.fontUi));
  return out;
}

// ---------------------------------------------------------------------------------------------
// Reference plane

export type PlaneKind = "none" | "xy" | "xz" | "yz";
export const PLANE_KINDS: readonly PlaneKind[] = ["none", "xy", "xz", "yz"];
export const PLANE_LABELS: Record<PlaneKind, string> = {
  none: "None",
  xy: "XY (horizontal)",
  xz: "XZ (side)",
  yz: "YZ (front)",
};
/** Axis index of the plane normal. */
const NORMAL: Record<Exclude<PlaneKind, "none">, 0 | 1 | 2> = { xy: 2, xz: 1, yz: 0 };
const AXIS_NAME = ["x", "y", "z"] as const;
export const PLANE_OPACITY = 0.12;
export const PLANE_OUTLINE_WIDTH = 2;
/** The plane overshoots the car box by this share on each side. */
export const PLANE_MARGIN = 0.15;
export const MM = 1000;

/** Default offset (mm): ground, centreline and front axle all sit at 0 in the ISO frame. */
export function planeDefaultOffset(): number {
  return 0;
}

/** Slider range in whole mm along the plane normal: the car box along that axis, plus 10 % per side. */
export function planeRange(kind: Exclude<PlaneKind, "none">, car: Bounds): { min: number; max: number } {
  const i = NORMAL[kind];
  const lo = car.min[i]!;
  const hi = car.max[i]!;
  const pad = (hi - lo) * 0.1;
  return { min: Math.floor((lo - pad) * MM + 1e-9), max: Math.ceil((hi + pad) * MM - 1e-9) };
}

/** Offset (mm) held inside the slider range. */
export function clampPlaneOffset(kind: Exclude<PlaneKind, "none">, mm: number, car: Bounds): number {
  const r = planeRange(kind, car);
  return Math.min(r.max, Math.max(r.min, Number.isFinite(mm) ? Math.round(mm) : 0));
}

export interface PlaneGeometry {
  /** Corners in order around the outline. */
  corners: Vec3[];
  /** Closed outline (first corner repeated). */
  outline: Vec3[];
  label: { at: Vec3; text: string };
}

/** Rectangle of the plane in the body frame, covering the car box plus `PLANE_MARGIN`. */
export function planeGeometry(kind: Exclude<PlaneKind, "none">, offsetMm: number, car: Bounds): PlaneGeometry {
  const n = NORMAL[kind];
  const [a, b] = ([0, 1, 2] as const).filter((k) => k !== n) as [number, number];
  const ext = (k: number): [number, number] => {
    const pad = (car.max[k]! - car.min[k]!) * PLANE_MARGIN;
    return [car.min[k]! - pad, car.max[k]! + pad];
  };
  const [a0, a1] = ext(a);
  const [b0, b1] = ext(b);
  const at = (u: number, v: number): Vec3 => {
    const p: Vec3 = [0, 0, 0];
    p[n] = offsetMm / MM;
    p[a] = u;
    p[b] = v;
    return p;
  };
  const corners = [at(a0, b0), at(a1, b0), at(a1, b1), at(a0, b1)];
  // Label at the plane corner nearest the default viewpoint (+x, -y), at the top for the vertical
  // planes (clear of the ground tick labels), pulled in to half the plane margin outside the car box
  // so it stays in the framed area. The horizontal plane's near corner is the lowest point on screen, so
  // there the label goes to the middle of the near edge.
  const nearEnd = (k: number): -1 | 1 => (k === 1 ? -1 : 1);
  const anchor = (k: number, lo: number, hi: number): number => {
    const half = ((hi - lo) * PLANE_MARGIN) / (1 + 2 * PLANE_MARGIN) / 2;
    return nearEnd(k) === 1 ? hi - half : lo + half;
  };
  return {
    corners,
    outline: [...corners, corners[0]!],
    label: { at: at(n === 2 ? (a0 + a1) / 2 : anchor(a, a0, a1), anchor(b, b0, b1)), text: `${AXIS_NAME[n]} = ${Math.round(offsetMm)} mm` },
  };
}

/** Translucent plane and its outline in the layout blue; inert. Empty for "none". */
export function buildPlaneTraces(kind: PlaneKind, offsetMm: number, car: Bounds, t: Tokens): PlotlyObject[] {
  if (kind === "none") return [];
  const g = planeGeometry(kind, offsetMm, car);
  const col = (k: 0 | 1 | 2) => g.corners.map((c) => c[k]);
  return [
    {
      type: "mesh3d",
      x: col(0),
      y: col(1),
      z: col(2),
      i: [0, 0],
      j: [1, 2],
      k: [2, 3],
      color: t.layoutBlue,
      opacity: PLANE_OPACITY,
      flatshading: true,
      lighting: { ambient: 1, diffuse: 0, specular: 0 },
      showlegend: false,
      hoverinfo: "skip",
    },
    {
      type: "scatter3d",
      mode: "lines",
      x: g.outline.map((p) => p[0]),
      y: g.outline.map((p) => p[1]),
      z: g.outline.map((p) => p[2]),
      line: { color: t.layoutBlue, width: PLANE_OUTLINE_WIDTH },
      showlegend: false,
      hoverinfo: "skip",
    },
  ];
}

/**
 * The plane label as a 3D scene annotation (14 px `--ink`). An annotation, not a text trace: Plotly
 * draws a moved gl3d text trace far too small, and the offset slider moves it.
 */
export function buildPlaneAnnotations(kind: PlaneKind, offsetMm: number, car: Bounds, t: Tokens): PlotlyObject[] {
  if (kind === "none") return [];
  const { label } = planeGeometry(kind, offsetMm, car);
  return [
    {
      x: label.at[0],
      y: label.at[1],
      z: label.at[2],
      text: label.text,
      showarrow: false,
      xanchor: "center",
      yanchor: "middle",
      captureevents: false,
      font: { family: t.fontUi, size: PLANE_LABEL_SIZE, color: t.ink },
    },
  ];
}
