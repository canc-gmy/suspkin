import type { Hardpoint } from "../api/types";

/** A point counts as moved when it is further than this (mm) from the reference. */
export const DIFF_MM = 0.1;

export type Vec3 = [number, number, number];
type Axle = Hardpoint["axle"];
type Side = Hardpoint["side"];

/** Hardpoints of a revision: null when they could not be read. */
export type RevisionPoints = readonly Hardpoint[] | null;

export interface DeltaCell {
  /** Per-axis difference to the reference (mm). */
  d: Vec3;
  /** Vector magnitude (mm). */
  mag: number;
}

export interface HardpointRow {
  key: string;
  axle: Axle;
  point: string;
  side: Side;
  /** A right point taken from `right_overrides` in at least one revision. */
  overridden: boolean;
  /** Position in mm per revision id (null: the revision has no such point or no hardpoints). */
  values: Record<string, Vec3 | null>;
  /** Difference to the reference per revision id (null: not computable; absent for the reference). */
  deltas: Record<string, DeltaCell | null>;
  /** Any revision is further than 0.1 mm from the reference. */
  differs: boolean;
}

export const pointKey = (h: Pick<Hardpoint, "axle" | "point" | "side">) => `${h.axle}|${h.point}|${h.side}`;

const mm = (h: Hardpoint): Vec3 => [h.x * 1000, h.y * 1000, h.z * 1000];

/** Scene label prefix of a corner ("FL", "FR", "RL", "RR"). */
export function cornerId(axle: Axle, side: Side): string {
  return `${axle === "front" ? "F" : "R"}${side === "left" ? "L" : "R"}`;
}

/** Scene label of a hardpoint ("FL:uca_outer"), the key of the posed positions in a scene. */
export const sceneName = (h: Pick<Hardpoint, "axle" | "point" | "side">) => `${cornerId(h.axle, h.side)}:${h.point}`;

function delta(a: Vec3, ref: Vec3): DeltaCell {
  const d: Vec3 = [a[0] - ref[0], a[1] - ref[1], a[2] - ref[2]];
  return { d, mag: Math.hypot(d[0], d[1], d[2]) };
}

/**
 * One row per (axle, point, side) over every revision.
 *
 * `points` maps a revision id to its hardpoints (null when unreadable). Deltas are against
 * `refId` and exist for every revision except the reference. A point present in a revision but
 * absent from the reference (or the other way round) counts as differing.
 */
export function buildHardpointRows(
  points: Readonly<Record<string, RevisionPoints>>,
  ids: readonly string[],
  refId: string | null,
): HardpointRow[] {
  const maps = new Map<string, Map<string, Hardpoint>>();
  for (const id of ids) {
    const list = points[id];
    if (list) maps.set(id, new Map(list.map((h) => [pointKey(h), h])));
  }
  const first = new Map<string, Hardpoint>();
  const overridden = new Set<string>();
  for (const m of maps.values()) {
    for (const [k, h] of m) {
      if (!first.has(k)) first.set(k, h);
      if (h.overridden) overridden.add(k);
    }
  }
  const refMap = refId ? maps.get(refId) : undefined;
  const rows: HardpointRow[] = [];
  for (const [key, h] of first) {
    const values: Record<string, Vec3 | null> = {};
    const deltas: Record<string, DeltaCell | null> = {};
    let differs = false;
    const refPos = refMap?.get(key);
    for (const id of ids) {
      const mine = maps.get(id)?.get(key);
      values[id] = mine ? mm(mine) : null;
      if (id === refId) continue;
      if (mine && refPos) {
        const cell = delta(mm(mine), mm(refPos));
        deltas[id] = cell;
        if (cell.mag > DIFF_MM) differs = true;
      } else {
        deltas[id] = null;
        // present on one side only (both readable): the point was added or removed
        if (refMap && maps.has(id) && (mine !== undefined) !== (refPos !== undefined)) differs = true;
      }
    }
    rows.push({ key, axle: h.axle, point: h.point, side: h.side, overridden: overridden.has(key), values, deltas, differs });
  }
  return sortRows(rows);
}

const AXLE_ORDER: Record<Axle, number> = { front: 0, rear: 1 };

/** Front before rear; inside an axle the differing rows first, then point name, left before right. */
export function sortRows(rows: readonly HardpointRow[]): HardpointRow[] {
  return [...rows].sort(
    (a, b) =>
      AXLE_ORDER[a.axle] - AXLE_ORDER[b.axle] ||
      Number(b.differs) - Number(a.differs) ||
      a.point.localeCompare(b.point) ||
      Number(a.side === "right") - Number(b.side === "right"),
  );
}

export interface RowFilter {
  differingOnly: boolean;
  /** Case-insensitive substring of the point name. */
  text: string;
}

export function filterRows(rows: readonly HardpointRow[], f: RowFilter): HardpointRow[] {
  const q = f.text.trim().toLowerCase();
  return rows.filter((r) => (!f.differingOnly || r.differs) && (q === "" || r.point.toLowerCase().includes(q)));
}

/**
 * Scene names ("FL:uca_outer") of the points of `rev` that are further than 0.1 mm from the
 * reference, or missing from it. Empty when either revision is unreadable.
 */
export function changedSceneNames(rev: RevisionPoints, ref: RevisionPoints): string[] {
  if (!rev || !ref) return [];
  const refMap = new Map(ref.map((h) => [pointKey(h), h]));
  const out: string[] = [];
  for (const h of rev) {
    const r = refMap.get(pointKey(h));
    if (!r || delta(mm(h), mm(r)).mag > DIFF_MM) out.push(sceneName(h));
  }
  return out;
}
