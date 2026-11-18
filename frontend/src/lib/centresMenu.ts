import { CENTRE_KINDS, REGION_ROLES, type CentreKind, type RegionRole } from "../theme/scene3d";

export const CENTRES_KEY = "suspkin.centres";

/** Ticked kinds saved for the session; anything unreadable gives none. */
export function loadCentreKinds(): Set<CentreKind> {
  try {
    const raw = JSON.parse(localStorage.getItem(CENTRES_KEY) ?? "[]") as unknown;
    if (!Array.isArray(raw)) return new Set();
    return new Set(CENTRE_KINDS.filter((k) => raw.includes(k)));
  } catch {
    return new Set();
  }
}

export function saveCentreKinds(kinds: ReadonlySet<CentreKind>): void {
  try {
    localStorage.setItem(CENTRES_KEY, JSON.stringify(CENTRE_KINDS.filter((k) => kinds.has(k))));
  } catch {
    // storage may be unavailable (private mode, snapshot opened from a file): keep the state in memory
  }
}

/** A new set with `kind` switched on or off. */
export function toggleCentreKind(kinds: ReadonlySet<CentreKind>, kind: CentreKind, on: boolean): Set<CentreKind> {
  const next = new Set(kinds);
  if (on) next.add(kind);
  else next.delete(kind);
  return next;
}

export const REGIONS_KEY = "suspkin.regions";

/** Ticked region roles saved for the session; anything unreadable gives none. */
export function loadRegionRoles(): Set<RegionRole> {
  try {
    const raw = JSON.parse(localStorage.getItem(REGIONS_KEY) ?? "[]") as unknown;
    if (!Array.isArray(raw)) return new Set();
    return new Set(REGION_ROLES.filter((k) => raw.includes(k)));
  } catch {
    return new Set();
  }
}

export function saveRegionRoles(roles: ReadonlySet<RegionRole>): void {
  try {
    localStorage.setItem(REGIONS_KEY, JSON.stringify(REGION_ROLES.filter((k) => roles.has(k))));
  } catch {
    // see saveCentreKinds
  }
}

/** A new set with `role` switched on or off. */
export function toggleRegionRole(roles: ReadonlySet<RegionRole>, role: RegionRole, on: boolean): Set<RegionRole> {
  const next = new Set(roles);
  if (on) next.add(role);
  else next.delete(role);
  return next;
}
