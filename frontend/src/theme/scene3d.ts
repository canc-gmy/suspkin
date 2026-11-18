import { themeScene, type PlotlyObject, type Tokens } from "./plotlyTemplate";
import { buildGridTraces, buildPlaneTraces, worldFrame, type PlaneKind, type World } from "./sceneGrid";

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
const VIEW_EYE = { x: 1.6, y: -1.6, z: 0.9 };
export const DEFAULT_CAMERA: Camera = {
  eye: VIEW_EYE,
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
  /** The large box the scene is laid out in (ground grid); the car box above sits in its middle. */
  world?: World;
}

/**
 * Car box (bounding box plus 10% margin per axis) with its aspect ratio, and the world box around
 * it that the Plotly axes use. The camera and the overlay clipping work on the car box.
 */
export function sceneFrame(b: Bounds): SceneFrame {
  const ranges = AXES.map((_, i): [number, number] => {
    const lo = b.min[i]!;
    const hi = b.max[i]!;
    const pad = Math.max(hi - lo, 1e-6) * MARGIN;
    return [lo - pad, hi + pad];
  });
  const len = ranges.map(([lo, hi]) => hi - lo);
  const top = Math.max(...len);
  const box: Bounds = { min: [ranges[0]![0], ranges[1]![0], ranges[2]![0]], max: [ranges[0]![1], ranges[1]![1], ranges[2]![1]] };
  return { ranges, aspect: { x: len[0]! / top, y: len[1]! / top, z: len[2]! / top }, world: worldFrame(box) };
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

/** Share of the canvas the car should fill (width or height, whichever limits) by default. */
export const FRAME_FILL = 0.8;
/** Plotly's 3D camera looks through a 45 degree vertical field of view. */
const HALF_FOV_TAN = Math.tan(Math.PI / 8);
/**
 * Plotly draws the scene box (plus its tick labels) about 0.82 times the size the pinhole model
 * predicts, measured in the browser, so the model is asked for that much more.
 */
const DRAWN_SCALE = 1;

/**
 * Default camera for a scene: the standard viewing direction, with the eye distance chosen so the
 * scene box fills `FRAME_FILL` of the canvas (perspective included). Without a frame or a measured
 * canvas it falls back to `DEFAULT_CAMERA`. The scene box has the aspect ratio as its side lengths.
 */
export function fitCamera(
  frame: SceneFrame | null,
  canvas: { width: number; height: number } | null,
  fill: number = FRAME_FILL,
): Camera {
  if (!frame || !canvas || !(canvas.width > 0) || !(canvas.height > 0)) return DEFAULT_CAMERA;
  const e = VIEW_EYE;
  const len = Math.hypot(e.x, e.y, e.z);
  const u: Vec3 = [e.x / len, e.y / len, e.z / len]; // from the centre towards the eye
  const right = normalize([-u[1], u[0], 0]);
  const up = cross(right, [-u[0], -u[1], -u[2]]);
  // the car box is a small part of the scene box when a world box surrounds it
  const share = frame.world?.scale ?? 1;
  const half = [frame.aspect.x / 2, frame.aspect.y / 2, frame.aspect.z / 2].map((h) => h * share);
  const corners: Vec3[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    corners.push([sx * half[0]!, sy * half[1]!, sz * half[2]!]);
  }
  const aspect = canvas.width / canvas.height;

  // Perspective pins the centre to the middle of the canvas, but the near corners reach farther
  // than the far ones. So the look-at point is shifted until the box is centred, and then the
  // distance is searched until the box fills the wanted share of the canvas.
  const solve = (d: number) => {
    let sx = 0;
    let sy = 0;
    let ext = { x0: 0, x1: 0, y0: 0, y1: 0 };
    for (let it = 0; it < 8; it++) {
      const c0: Vec3 = [sx * right[0] + sy * up[0], sx * right[1] + sy * up[1], sx * right[2] + sy * up[2]];
      ext = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
      for (const p of corners) {
        const rel: Vec3 = [p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]];
        const k = 1 / ((d - dot(rel, u)) * HALF_FOV_TAN);
        const px = dot(rel, right) * k;
        const py = dot(rel, up) * k;
        ext = { x0: Math.min(ext.x0, px), x1: Math.max(ext.x1, px), y0: Math.min(ext.y0, py), y1: Math.max(ext.y1, py) };
      }
      sx += ((ext.x0 + ext.x1) / 2) * d * HALF_FOV_TAN;
      sy += ((ext.y0 + ext.y1) / 2) * d * HALF_FOV_TAN;
    }
    const cover = Math.max((ext.x1 - ext.x0) / 2 / aspect, (ext.y1 - ext.y0) / 2) * DRAWN_SCALE;
    return { cover, centre: [sx * right[0] + sy * up[0], sx * right[1] + sy * up[1], sx * right[2] + sy * up[2]] as Vec3 };
  };
  let lo = Math.hypot(...half) * 1.01; // outside the box
  let hi = lo * 64;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    if (solve(mid).cover > fill) lo = mid;
    else hi = mid;
  }
  const c = solve(hi).centre;
  const eye: Vec3 = [c[0] + hi * u[0], c[1] + hi * u[1], c[2] + hi * u[2]];
  return { eye: { x: eye[0], y: eye[1], z: eye[2] }, center: { x: c[0], y: c[1], z: c[2] }, up: DEFAULT_CAMERA.up! };
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/**
 * The camera to put in the layout: the user's last camera (captured from `plotly_relayout`), or
 * the default on the first render and on an explicit reset. It is never taken from the server
 * figure, so a pose change cannot move it (Plotly alone drops it when the layout omits it).
 */
export function cameraForLayout(o: { reset: boolean; saved: Camera | null; fallback?: Camera }): Camera {
  const dflt = o.fallback ?? DEFAULT_CAMERA;
  return o.reset ? dflt : (o.saved ?? dflt);
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
  /** Posed positions (m, scene frame) of the hardpoints to ring; empty or absent: no rings. */
  highlight?: readonly Vec3[];
  /** Rotation-centre traces from `buildCentreTraces`, drawn last. */
  centres?: readonly PlotlyObject[];
  /** Show the legend strip above the scene (default true). */
  legend?: boolean;
  /** Reference plane (body frame, offset in mm); absent or "none": no plane. */
  plane?: { kind: PlaneKind; offsetMm: number };
}

export const RING_SIZE = 6;

/** Posed hardpoint positions by scene name ("FL:uca_outer"), as the scene endpoint returns them. */
export function posedHardpoints(fig: { hardpoints?: unknown } | null | undefined): Record<string, Vec3> {
  const raw = fig?.hardpoints;
  const out: Record<string, Vec3> = {};
  if (!raw || typeof raw !== "object") return out;
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n))) {
      out[k] = [v[0] as number, v[1] as number, v[2] as number];
    }
  }
  return out;
}

/** Open 6 px rings in the layout blue around the given points; inert (no hover, no legend). */
export function ringTrace(points: readonly Vec3[], t: Pick<Tokens, "layoutBlue">): PlotlyObject {
  return {
    type: "scatter3d",
    mode: "markers",
    x: points.map((p) => p[0]),
    y: points.map((p) => p[1]),
    z: points.map((p) => p[2]),
    marker: { symbol: "circle-open", size: RING_SIZE, color: t.layoutBlue, line: { color: t.layoutBlue, width: 2 } },
    showlegend: false,
    hoverinfo: "skip",
  };
}

/** Themed 3D figure: styling from the tokens, fixed ranges and aspect, optional ghost behind. */
export function buildSceneFigure(i: SceneFigureInput): { data: PlotlyObject[]; layout: PlotlyObject } {
  const t = i.tokens;
  const themed = themeScene(i.figure, t, { legend: i.legend ?? true });
  const sc = themed.layout["scene"] as PlotlyObject;
  // with a world box the Plotly axes are hidden (no walls, ticks or titles): the grid is drawn as traces
  const world = i.frame?.world;
  const axisOf = (name: "xaxis" | "yaxis" | "zaxis", idx: number): PlotlyObject => ({
    ...(sc[name] as PlotlyObject),
    ...(i.frame ? { range: (world ?? i.frame).ranges[idx], autorange: false } : {}),
    ...(world ? { visible: false, showbackground: false, showgrid: false, zeroline: false, showline: false } : {}),
  });
  const scene: PlotlyObject = {
    ...sc,
    xaxis: axisOf("xaxis", 0),
    yaxis: axisOf("yaxis", 1),
    zaxis: axisOf("zaxis", 2),
    aspectmode: i.frame ? "manual" : "data",
    ...(i.frame ? { aspectratio: world ? world.aspect : i.frame.aspect } : {}),
    uirevision: SCENE_UIREVISION,
  };
  delete scene["camera"];
  scene["camera"] = i.camera;
  const ghost = i.ghost ? ghostTraces((i.ghost.data as PlotlyObject[] | undefined) ?? [], t) : [];
  const car = frameBox(i.frame);
  const grid = world && car ? buildGridTraces(world, car, t) : [];
  const plane = car && i.plane ? buildPlaneTraces(i.plane.kind, i.plane.offsetMm, car, t) : [];
  return {
    data: [
      ...grid,
      ...ghost,
      ...themed.data,
      ...(i.highlight && i.highlight.length > 0 ? [ringTrace(i.highlight, t)] : []),
      ...(i.centres ?? []),
      ...plane,
    ],
    layout: { ...themed.layout, uirevision: SCENE_UIREVISION, autosize: true, paper_bgcolor: t.sheetRaised, scene },
  };
}

// ---------------------------------------------------------------------------------------------
// Rotation centres

export type CentreKind = "roll" | "fv" | "sv" | "pitch";
export const CENTRE_KINDS: readonly CentreKind[] = ["roll", "fv", "sv", "pitch"];
export const CENTRE_LABELS: Record<CentreKind, string> = {
  roll: "Roll centres and roll axis",
  fv: "Front-view instant centres",
  sv: "Side-view instant centres",
  pitch: "Pitch centre",
};

type ColourKey = "layoutBlue" | "ink" | "inkMuted";
/** Marker symbol, token colour and legend name of each kind of centre. */
export const CENTRE_STYLE: Record<CentreKind, { symbol: string; colour: ColourKey; legend: string; size: number }> = {
  roll: { symbol: "diamond", colour: "layoutBlue", legend: "Roll centres", size: 5 },
  fv: { symbol: "x", colour: "ink", legend: "Front-view ICs", size: 4 },
  sv: { symbol: "cross", colour: "inkMuted", legend: "Side-view ICs", size: 4 },
  pitch: { symbol: "diamond-open", colour: "ink", legend: "Pitch centre", size: 5 },
};
export const CENTRE_LINE_WIDTH = 1;
export const CENTRE_AXIS_WIDTH = 2;
export const CENTRE_EDGE_SYMBOL = "square-open";
const CORNER_IDS = ["FL", "FR", "RL", "RR"] as const;

type Pt = Vec3 | null;
export interface Centres {
  roll: { front: Pt; rear: Pt };
  fv_ic: Record<string, Pt>;
  sv_ic: Record<string, Pt>;
  contact: Record<string, Pt>;
  pitch: Pt;
}

function vec3(v: unknown): Pt {
  return Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n))
    ? [v[0] as number, v[1] as number, v[2] as number]
    : null;
}

function pointMap(raw: unknown): Record<string, Pt> {
  const out: Record<string, Pt> = {};
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  for (const id of CORNER_IDS) out[id] = vec3(o[id]);
  return out;
}

/** The `centres` block of a scene response (null when it has none, e.g. an old snapshot). */
export function parseCentres(fig: { centres?: unknown } | null | undefined): Centres | null {
  const raw = fig?.centres;
  if (!raw || typeof raw !== "object") return null;
  const c = raw as Record<string, unknown>;
  const roll = (c["roll"] ?? {}) as Record<string, unknown>;
  return {
    roll: { front: vec3(roll["front"]), rear: vec3(roll["rear"]) },
    fv_ic: pointMap(c["fv_ic"]),
    sv_ic: pointMap(c["sv_ic"]),
    contact: pointMap(c["contact"]),
    pitch: vec3(c["pitch"]),
  };
}

const EPS = 1e-9;

export function inBox(p: Vec3, box: Bounds): boolean {
  return p.every((v, i) => v >= box.min[i]! - EPS && v <= box.max[i]! + EPS);
}

export interface ClippedSegment {
  a: Vec3;
  b: Vec3;
  /** True when that end was moved onto the box (the original end is outside). */
  aClipped: boolean;
  bClipped: boolean;
}

/** Part of the segment a-b inside the box (Liang-Barsky); null when none of it is inside. */
export function clipSegmentToBox(a: Vec3, b: Vec3, box: Bounds): ClippedSegment | null {
  let t0 = 0;
  let t1 = 1;
  for (let i = 0; i < 3; i++) {
    const d = b[i]! - a[i]!;
    const lo = box.min[i]! - a[i]!;
    const hi = box.max[i]! - a[i]!;
    if (Math.abs(d) < 1e-15) {
      if (lo > EPS || hi < -EPS) return null; // parallel to this slab and outside it
      continue;
    }
    let ta = lo / d;
    let tb = hi / d;
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta);
    t1 = Math.min(t1, tb);
    if (t0 > t1) return null;
  }
  const at = (t: number): Vec3 => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  return { a: at(t0), b: at(t1), aClipped: t0 > 0, bClipped: t1 < 1 };
}

export interface CentreTraces {
  traces: PlotlyObject[];
  /** One line per instant centre outside the scene box, plus out-of-view roll and pitch centres. */
  notes: string[];
}

function markerTrace(kind: CentreKind, pts: readonly Vec3[], t: Tokens): PlotlyObject {
  const st = CENTRE_STYLE[kind];
  return {
    type: "scatter3d",
    mode: "markers",
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
    z: pts.map((p) => p[2]),
    name: st.legend,
    legendgroup: kind,
    showlegend: true,
    marker: { symbol: st.symbol, size: st.size, color: t[st.colour], line: { color: t[st.colour], width: 1 } },
    hoverinfo: "skip",
  };
}

function lineTrace(
  kind: CentreKind,
  segs: readonly (readonly [Vec3, Vec3])[],
  t: Tokens,
  width: number,
  dash: string,
  legend = false,
): PlotlyObject {
  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  const z: (number | null)[] = [];
  for (const [a, b] of segs) {
    x.push(a[0], b[0], null);
    y.push(a[1], b[1], null);
    z.push(a[2], b[2], null);
  }
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z,
    name: CENTRE_STYLE[kind].legend,
    legendgroup: kind,
    showlegend: legend,
    line: { color: t[CENTRE_STYLE[kind].colour], width, dash },
    hoverinfo: "skip",
  };
}

function edgeTrace(kind: CentreKind, pts: readonly Vec3[], t: Tokens): PlotlyObject {
  const c = t[CENTRE_STYLE[kind].colour];
  return {
    type: "scatter3d",
    mode: "markers",
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
    z: pts.map((p) => p[2]),
    legendgroup: kind,
    showlegend: false,
    marker: { symbol: CENTRE_EDGE_SYMBOL, size: 4, color: c, line: { color: c, width: 1 } },
    hoverinfo: "skip",
  };
}

/** Distance and direction word of an instant centre against its contact patch, for the note. */
function icNote(id: string, view: "fv" | "sv", ic: Vec3, cp: Vec3): string {
  if (view === "fv") {
    const d = ic[1] - cp[1];
    const outboard = id.endsWith("L") ? d > 0 : d < 0;
    return `${id} front-view IC ${Math.abs(d).toFixed(2)} m ${outboard ? "outboard" : "inboard"}`;
  }
  const d = ic[0] - cp[0];
  return `${id} side-view IC ${Math.abs(d).toFixed(2)} m ${d > 0 ? "forward" : "rearward"}`;
}

/**
 * Markers and construction lines of the ticked kinds, clipped to `box` (the fixed scene box), so
 * they never change the axis ranges. Undefined (null) centres are skipped.
 */
export function buildCentreTraces(
  centres: Centres | null,
  kinds: ReadonlySet<CentreKind>,
  box: Bounds | null,
  t: Tokens,
): CentreTraces {
  const out: CentreTraces = { traces: [], notes: [] };
  if (!centres || !box) return out;

  if (kinds.has("roll")) {
    const named: [string, Pt][] = [
      ["Front", centres.roll.front],
      ["Rear", centres.roll.rear],
    ];
    const shown: Vec3[] = [];
    for (const [name, p] of named) {
      if (!p) continue;
      if (inBox(p, box)) shown.push(p);
      else out.notes.push(`${name} roll centre outside the view`);
    }
    const { front, rear } = centres.roll;
    const seg = front && rear ? clipSegmentToBox(front, rear, box) : null;
    if (seg) out.traces.push(lineTrace("roll", [[seg.a, seg.b]], t, CENTRE_AXIS_WIDTH, "solid", shown.length === 0));
    if (shown.length > 0) out.traces.push(markerTrace("roll", shown, t));
  }

  for (const [kind, ics] of [
    ["fv", centres.fv_ic],
    ["sv", centres.sv_ic],
  ] as const) {
    if (!kinds.has(kind)) continue;
    const segs: [Vec3, Vec3][] = [];
    const inside: Vec3[] = [];
    const edge: Vec3[] = [];
    const any = CORNER_IDS.some((id) => ics[id] && centres.contact[id]);
    for (const id of CORNER_IDS) {
      const ic = ics[id];
      const cp = centres.contact[id];
      if (!ic || !cp) continue;
      const seg = clipSegmentToBox(cp, ic, box);
      if (seg) {
        segs.push([seg.a, seg.b]);
        if (seg.bClipped) edge.push(seg.b);
      }
      if (inBox(ic, box)) inside.push(ic);
      else out.notes.push(icNote(id, kind, ic, cp));
    }
    if (!any) continue;
    // Plotly lists no legend entry for an empty trace: with every IC off the box, the line carries it
    if (segs.length > 0) out.traces.push(lineTrace(kind, segs, t, CENTRE_LINE_WIDTH, "dash", inside.length === 0));
    if (edge.length > 0) out.traces.push(edgeTrace(kind, edge, t));
    if (inside.length > 0) out.traces.push(markerTrace(kind, inside, t));
  }

  if (kinds.has("pitch") && centres.pitch) {
    if (inBox(centres.pitch, box)) out.traces.push(markerTrace("pitch", [centres.pitch], t));
    else out.notes.push("Pitch centre outside the view");
  }
  return out;
}

/** The scene box (the fixed axis ranges) as bounds, or null before the frame is known. */
export function frameBox(frame: SceneFrame | null): Bounds | null {
  if (!frame) return null;
  const r = frame.ranges;
  return { min: [r[0]![0], r[1]![0], r[2]![0]], max: [r[0]![1], r[1]![1], r[2]![1]] };
}

// ---------------------------------------------------------------------------------------------
// Constraint regions

export type RegionRole = "domain" | "inside" | "outside";
export const REGION_ROLES: readonly RegionRole[] = ["domain", "inside", "outside"];
export const REGION_LABELS: Record<RegionRole, string> = {
  domain: "Design domains",
  inside: "Requirements (inside)",
  outside: "Keep-outs (outside)",
};
type RegionColourKey = "regionDomain" | "regionRequired" | "regionKeepout";
/** Colour token, dash and legend name of each role; line style repeats what colour says. */
export const REGION_STYLE: Record<RegionRole, { colour: RegionColourKey; dash: string }> = {
  domain: { colour: "regionDomain", dash: "solid" },
  inside: { colour: "regionRequired", dash: "dot" },
  outside: { colour: "regionKeepout", dash: "dash" },
};
export const REGION_LINE_WIDTH = 1;
/** The right copy of a symmetric domain is drawn fainter than the variable's own domain. */
export const REGION_MIRROR_OPACITY = 0.5;

/** The part of a region the 3D view needs (the API row, minus its descriptive fields). */
export interface RegionLike {
  role: RegionRole;
  mirrored?: boolean;
  lines: readonly (readonly (readonly number[])[])[];
}

function regionLine(
  rs: readonly RegionLike[],
  role: RegionRole,
  box: Bounds,
  t: Tokens,
  mirrored: boolean,
): PlotlyObject | null {
  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  const z: (number | null)[] = [];
  for (const r of rs) {
    for (const ln of r.lines) {
      for (let i = 0; i + 1 < ln.length; i++) {
        const p = ln[i]!;
        const q = ln[i + 1]!;
        const seg = clipSegmentToBox([p[0]!, p[1]!, p[2]!], [q[0]!, q[1]!, q[2]!], box);
        if (!seg) continue;
        x.push(seg.a[0], seg.b[0], null);
        y.push(seg.a[1], seg.b[1], null);
        z.push(seg.a[2], seg.b[2], null);
      }
    }
  }
  if (x.length === 0) return null;
  const st = REGION_STYLE[role];
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z,
    name: REGION_LABELS[role],
    legendgroup: `region-${role}`,
    showlegend: !mirrored,
    opacity: mirrored ? REGION_MIRROR_OPACITY : 1,
    line: { color: t[st.colour], width: REGION_LINE_WIDTH, dash: st.dash },
    hoverinfo: "skip",
  };
}

/**
 * Wireframes of the ticked roles, clipped to the scene box so they never change the axis ranges.
 * Regions are fixed in the body frame, like the chassis points, so the pose does not touch them.
 */
export function buildRegionTraces(
  regions: readonly RegionLike[] | null | undefined,
  roles: ReadonlySet<RegionRole>,
  box: Bounds | null,
  t: Tokens,
): PlotlyObject[] {
  if (!regions || !box) return [];
  const out: PlotlyObject[] = [];
  for (const role of REGION_ROLES) {
    if (!roles.has(role)) continue;
    const mine = regions.filter((r) => r.role === role);
    for (const mirrored of [false, true]) {
      const tr = regionLine(
        mine.filter((r) => Boolean(r.mirrored) === mirrored),
        role,
        box,
        t,
        mirrored,
      );
      if (tr) out.push(tr);
    }
  }
  return out;
}
