import { PLANE_KINDS, planeDefaultOffset, type PlaneKind } from "../theme/sceneGrid";

export const PLANE_KEY = "suspkin.plane";

type Active = Exclude<PlaneKind, "none">;
export interface PlaneState {
  kind: PlaneKind;
  /** Offset in mm along the normal, kept per orientation so switching back restores it. */
  offsets: Record<Active, number>;
}

export function defaultPlaneState(): PlaneState {
  const d = planeDefaultOffset();
  return { kind: "none", offsets: { xy: d, xz: d, yz: d } };
}

/** Plane choice saved for the session; anything unreadable gives the default (no plane). */
export function loadPlane(): PlaneState {
  const out = defaultPlaneState();
  try {
    const raw = JSON.parse(localStorage.getItem(PLANE_KEY) ?? "null") as Record<string, unknown> | null;
    if (!raw || typeof raw !== "object") return out;
    const kind = PLANE_KINDS.find((k) => k === raw["kind"]);
    if (kind) out.kind = kind;
    const offs = raw["offsets"] as Record<string, unknown> | undefined;
    for (const k of ["xy", "xz", "yz"] as const) {
      const v = offs?.[k];
      if (typeof v === "number" && Number.isFinite(v)) out.offsets[k] = v;
    }
  } catch {
    // unreadable storage: use the default
  }
  return out;
}

export function savePlane(state: PlaneState): void {
  try {
    localStorage.setItem(PLANE_KEY, JSON.stringify(state));
  } catch {
    // storage may be unavailable (private mode, snapshot opened from a file): keep the state in memory
  }
}
