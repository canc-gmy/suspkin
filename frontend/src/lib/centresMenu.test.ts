import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  CENTRES_KEY,
  loadCentreKinds,
  loadRegionRoles,
  saveCentreKinds,
  saveRegionRoles,
  toggleCentreKind,
  toggleRegionRole,
} from "./centresMenu";

describe("centres menu state", () => {
  beforeEach(() => localStorage.clear());
  it("round-trips through localStorage and ignores junk", () => {
    saveCentreKinds(new Set(["fv", "pitch"]));
    expect([...loadCentreKinds()].sort()).toEqual(["fv", "pitch"]);
    localStorage.setItem(CENTRES_KEY, "not json");
    expect(loadCentreKinds().size).toBe(0);
    localStorage.setItem(CENTRES_KEY, JSON.stringify(["roll", "bogus"]));
    expect([...loadCentreKinds()]).toEqual(["roll"]);
  });
  it("survives unavailable storage", () => {
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("denied");
    });
    expect(() => saveCentreKinds(new Set(["roll"]))).not.toThrow();
    spy.mockRestore();
  });
  it("toggles without mutating", () => {
    const a = new Set(["roll"] as const);
    const b = toggleCentreKind(a, "sv", true);
    expect([...b].sort()).toEqual(["roll", "sv"]);
    expect(a.size).toBe(1);
    expect(toggleCentreKind(b, "roll", false).has("roll")).toBe(false);
  });

  it("keeps region roles separately and toggles them", () => {
    saveRegionRoles(new Set(["outside", "domain"]));
    expect([...loadRegionRoles()].sort()).toEqual(["domain", "outside"]);
    expect(loadCentreKinds().size).toBe(0);
    expect(toggleRegionRole(new Set(["domain"] as const), "inside", true).size).toBe(2);
  });
});
