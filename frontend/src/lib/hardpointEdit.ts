import type { DeriveRequest, Hardpoint, HardpointEdit } from "../api/types";
import { pointKey, type Vec3 } from "./hardpoints";

export type Axis = "x" | "y" | "z";
export const AXES: readonly Axis[] = ["x", "y", "z"];
const INDEX: Record<Axis, 0 | 1 | 2> = { x: 0, y: 1, z: 2 };

type Axle = Hardpoint["axle"];
type Side = Hardpoint["side"];
type Id = { axle: Axle; point: string };

/** Edit mode of the Hardpoints tab: values typed in mm, only where they differ from the source. */
export interface EditState {
  /** Revision the edits are made on. */
  sourceId: string;
  /** Typed values (mm) per row key `axle|point|side`. */
  cells: Record<string, Partial<Record<Axis, number>>>;
  /** Explicit "Right side only" choice per `axle|point`; absent: on when the source has an override. */
  rightOnly: Record<string, boolean>;
}

export type EditAction =
  | { type: "start"; sourceId: string }
  | { type: "set"; axle: Axle; point: string; side: Side; axis: Axis; mm: number | null; base: Vec3 }
  | { type: "rightOnly"; axle: Axle; point: string; on: boolean }
  | { type: "reset"; axle: Axle; point: string }
  | { type: "discard" };

/** `null` means not editing. */
export type EditModel = EditState | null;

export const pairKey = (h: Id) => `${h.axle}|${h.point}`;
const rowKey = (h: Id, side: Side) => pointKey({ ...h, side });

const EPS = 1e-9;

export function editReducer(state: EditModel, a: EditAction): EditModel {
  if (a.type === "start") return { sourceId: a.sourceId, cells: {}, rightOnly: {} };
  if (a.type === "discard") return null;
  if (!state) return state;
  if (a.type === "set") {
    const key = rowKey(a, a.side);
    const row = { ...state.cells[key] };
    const same = a.mm === null || Math.abs(a.mm - a.base[INDEX[a.axis]]) < EPS;
    if (same) delete row[a.axis];
    else row[a.axis] = a.mm as number;
    const cells = { ...state.cells };
    if (Object.keys(row).length === 0) delete cells[key];
    else cells[key] = row;
    return { ...state, cells };
  }
  const pk = pairKey(a);
  if (a.type === "rightOnly") {
    const cells = { ...state.cells };
    if (!a.on) delete cells[rowKey(a, "right")]; // back to mirroring: the right point follows the left
    return { ...state, cells, rightOnly: { ...state.rightOnly, [pk]: a.on } };
  }
  const cells = { ...state.cells };
  delete cells[rowKey(a, "left")];
  delete cells[rowKey(a, "right")];
  const rightOnly = { ...state.rightOnly };
  delete rightOnly[pk];
  return { ...state, cells, rightOnly };
}

type BaseMap = ReadonlyMap<string, Hardpoint>;

/** Index the source revision's hardpoints by row key. */
export const indexBase = (points: readonly Hardpoint[]): BaseMap => new Map(points.map((h) => [pointKey(h), h]));

const mmOf = (h: Hardpoint): Vec3 => [h.x * 1000, h.y * 1000, h.z * 1000];

/** "Right side only" as it applies to a point: the explicit choice, else whether it has an override. */
export function isRightOnly(state: EditState, base: BaseMap, id: Id): boolean {
  const explicit = state.rightOnly[pairKey(id)];
  if (explicit !== undefined) return explicit;
  return base.get(rowKey(id, "right"))?.overridden === true;
}

const withCells = (v: Vec3, cells: Partial<Record<Axis, number>> | undefined): Vec3 => [
  cells?.x ?? v[0],
  cells?.y ?? v[1],
  cells?.z ?? v[2],
];

/** Position (mm) of a row as the edit would leave it; the right row mirrors the left unless right-only. */
export function effectiveMm(state: EditState, base: BaseMap, id: Id, side: Side): Vec3 | null {
  const h = base.get(rowKey(id, side));
  if (!h) return null;
  if (side === "left") return withCells(mmOf(h), state.cells[rowKey(id, "left")]);
  if (isRightOnly(state, base, id)) return withCells(mmOf(h), state.cells[rowKey(id, "right")]);
  const left = effectiveMm(state, base, id, "left");
  return left ? [left[0], -left[1], left[2]] : mmOf(h);
}

/** Whether a cell differs from the source (a mirrored right cell follows its left cell). */
export function cellChanged(state: EditState, base: BaseMap, id: Id, side: Side, axis: Axis): boolean {
  const h = base.get(rowKey(id, side));
  const now = effectiveMm(state, base, id, side);
  if (!h || !now) return false;
  return Math.abs(now[INDEX[axis]] - mmOf(h)[INDEX[axis]]) > EPS;
}

/** Number of points (left and right rows counted together) with at least one edited value. */
export function changedPoints(state: EditModel, base: BaseMap): number {
  if (!state) return 0;
  const seen = new Set<string>();
  for (const [key, h] of base) {
    if (!key.endsWith("|left")) continue;
    const id = { axle: h.axle, point: h.point };
    if (buildPointEdits(state, base, id).length > 0) seen.add(pairKey(id));
  }
  return seen.size;
}

const toM = (mm: number) => Number((mm / 1000).toFixed(9));

function edit(id: Id, side: HardpointEdit["side"], v: Partial<Record<Axis, number>>): HardpointEdit {
  const out: HardpointEdit = { axle: id.axle, point: id.point, side };
  for (const a of AXES) {
    const val = v[a];
    if (val !== undefined) out[a] = toM(val);
  }
  return out;
}

function buildPointEdits(state: EditState, base: BaseMap, id: Id): HardpointEdit[] {
  const left = state.cells[rowKey(id, "left")];
  const right = state.cells[rowKey(id, "right")];
  const baseRight = base.get(rowKey(id, "right"));
  const rightOnly = isRightOnly(state, base, id);
  if (!rightOnly) {
    // an override the user switched back to mirroring is dropped by a full "both" edit
    if (baseRight?.overridden && state.rightOnly[pairKey(id)] === false && !left) {
      const l = effectiveMm(state, base, id, "left");
      if (l) return [edit(id, "both", { x: l[0], y: l[1], z: l[2] })];
    }
    return left ? [edit(id, "both", left)] : [];
  }
  const out: HardpointEdit[] = [];
  // the right edit goes first: its missing coordinates come from the source's right point
  if (right) out.push(edit(id, "right", right));
  else if (left && baseRight && !baseRight.overridden) {
    // pin the right point where it is, so the left edit does not drag it along
    out.push(edit(id, "right", { x: baseRight.x * 1000, y: baseRight.y * 1000, z: baseRight.z * 1000 }));
  }
  if (left) out.push(edit(id, "left", left));
  return out;
}

/** The derive request for the current edits: mm to m, and the side of each edit. Empty: nothing to solve. */
export function buildEdits(state: EditModel, base: BaseMap): DeriveRequest["edits"] {
  if (!state) return [];
  const ids = new Map<string, Id>();
  for (const h of base.values()) ids.set(pairKey(h), { axle: h.axle, point: h.point });
  return [...ids.values()].flatMap((id) => buildPointEdits(state, base, id));
}

/** File name for a revision label: letters, digits, dot, dash and underscore, ending in .yaml. */
export function yamlFileName(label: string): string {
  const stem = label
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^[_.]+|_+$/g, "");
  return `${stem || "revision"}.yaml`;
}
