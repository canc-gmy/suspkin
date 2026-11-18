import { describe, expect, it } from "vitest";
import { pickRegionSource, regionSources } from "./regionSource";

const ds = [
  { id: "a", label: "ref", has_study: false },
  { id: "b", label: "raised", has_study: false },
  { id: "c", label: "study", has_study: true },
  { id: "d", label: "run", has_study: true },
];
const letters = { a: "A", b: "B", c: "C", d: "D" };

describe("region source", () => {
  const src = regionSources(ds, letters);
  it("lists only revisions with a study, by letter and label", () => {
    expect(src.map((s) => s.id)).toEqual(["c", "d"]);
    expect(src[0]!.name).toBe("C  study");
  });
  it("prefers the selected revision's own study", () => {
    expect(pickRegionSource(src, "d", null)).toBe("d");
  });
  it("falls back to the first open study when the selection has none", () => {
    expect(pickRegionSource(src, "b", null)).toBe("c");
  });
  it("honours an explicit pick while it is open, else ignores it", () => {
    expect(pickRegionSource(src, "b", "d")).toBe("d");
    expect(pickRegionSource(src, "b", "zzz")).toBe("c");
  });
  it("is null when no revision has a study", () => {
    expect(pickRegionSource([], "a", null)).toBeNull();
  });
});
