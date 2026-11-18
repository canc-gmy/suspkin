import { describe, expect, it } from "vitest";
import type { Tokens } from "./plotlyTemplate";
import {
  buildCentreTraces,
  buildSceneFigure,
  clipSegmentToBox,
  parseCentres,
  type Bounds,
  type CentreKind,
} from "./scene3d";

const box: Bounds = { min: [-1, -1, 0], max: [1, 1, 1] };
const tokens = { ink: "#111", inkMuted: "#777", layoutBlue: "#00f", sheetRaised: "#fff", regionDomain: "#0a0", regionKeepout: "#a05", regionRequired: "#0aa" } as unknown as Tokens;

describe("clipSegmentToBox", () => {
  it("keeps a segment inside", () => {
    const s = clipSegmentToBox([0, 0, 0.5], [0.5, 0.5, 0.5], box)!;
    expect(s.a).toEqual([0, 0, 0.5]);
    expect(s.aClipped || s.bClipped).toBe(false);
  });
  it("clips the far end onto the box edge", () => {
    const s = clipSegmentToBox([0, 0, 0.5], [4, 0, 0.5], box)!;
    expect(s.b[0]).toBeCloseTo(1, 12);
    expect(s.bClipped).toBe(true);
    expect(s.aClipped).toBe(false);
  });
  it("clips both ends of a crossing segment", () => {
    const s = clipSegmentToBox([-3, 0, 0.5], [3, 0, 0.5], box)!;
    expect(s.a[0]).toBeCloseTo(-1, 12);
    expect(s.b[0]).toBeCloseTo(1, 12);
    expect(s.aClipped && s.bClipped).toBe(true);
  });
  it("returns null for a segment outside, also one parallel to a slab", () => {
    expect(clipSegmentToBox([2, 2, 0.5], [3, 3, 0.5], box)).toBeNull();
    expect(clipSegmentToBox([-3, 5, 0.5], [3, 5, 0.5], box)).toBeNull();
  });
});

const centres = parseCentres({
  centres: {
    roll: { front: [0, 0, 0.05], rear: null },
    fv_ic: { FL: [0, 3, 0.4], FR: null, RL: [0, 0.2, 0.3], RR: null },
    sv_ic: { FL: [-0.5, 0.5, 0.2], FR: null, RL: null, RR: null },
    contact: { FL: [0, 0.6, 0], FR: [0, -0.6, 0], RL: [-1, 0.6, 0], RR: [-1, -0.6, 0] },
    pitch: null,
  },
})!;
const all = new Set<CentreKind>(["roll", "fv", "sv", "pitch"]);

describe("buildCentreTraces", () => {
  it("skips null centres and draws nothing for no kinds", () => {
    expect(buildCentreTraces(centres, new Set(), box, tokens).traces).toEqual([]);
    const r = buildCentreTraces(centres, new Set<CentreKind>(["roll", "pitch"]), box, tokens);
    expect(r.traces).toHaveLength(1); // front roll marker only: no axis line, no pitch centre
    expect(r.traces[0]!["x"]).toEqual([0]);
    expect(buildCentreTraces(null, all, box, tokens).traces).toEqual([]);
  });
  it("clips an outside IC at the box edge and lists its distance", () => {
    const r = buildCentreTraces(centres, new Set<CentreKind>(["fv"]), box, tokens);
    expect(r.notes).toEqual(["FL front-view IC 2.40 m outboard"]);
    const edge = r.traces.find((t) => (t["marker"] as { symbol: string } | undefined)?.symbol === "square-open")!;
    expect(edge["y"]).toEqual([1]);
    const line = r.traces.find((t) => t["mode"] === "lines")!;
    expect((line["y"] as (number | null)[]).filter((v) => v !== null && Math.abs(v) > 1 + 1e-9)).toEqual([]);
    const marker = r.traces.find((t) => t["name"] === "Front-view ICs" && t["mode"] === "markers")!;
    expect(marker["y"]).toEqual([0.2]);
  });
  it("words the side-view note by direction", () => {
    const far = parseCentres({ centres: { ...centres, sv_ic: { FL: [-5, 0.6, 0.2] } } })!;
    expect(buildCentreTraces(far, new Set<CentreKind>(["sv"]), box, tokens).notes).toEqual([
      "FL side-view IC 5.00 m rearward",
    ]);
  });
  it("appends to the figure without touching the scene ranges", () => {
    const r = buildCentreTraces(centres, all, box, tokens);
    const frame = { ranges: [[-1, 1], [-1, 1], [0, 1]] as [number, number][], aspect: { x: 1, y: 1, z: 0.5 } };
    const fig = buildSceneFigure({
      figure: { data: [], layout: { scene: { xaxis: {}, yaxis: {}, zaxis: {} } } },
      ghost: null,
      tokens,
      frame,
      camera: {},
      centres: r.traces,
    });
    expect(fig.data).toHaveLength(r.traces.length);
    expect(((fig.layout["scene"] as Record<string, unknown>)["xaxis"] as Record<string, unknown>)["range"]).toEqual([-1, 1]);
  });
});

import { buildRegionTraces, type RegionRole } from "./scene3d";

describe("buildRegionTraces", () => {
  const rs = [
    { role: "domain" as const, lines: [[[0, 0, 0.2], [0.5, 0, 0.2]]] },
    { role: "domain" as const, mirrored: true, lines: [[[0, 0, 0.2], [0.5, 0, 0.2]]] },
    { role: "outside" as const, lines: [[[0.5, 0, 0.5], [3, 0, 0.5], [3, 3, 0.5]]] },
    { role: "inside" as const, lines: [[[5, 5, 5], [6, 6, 6]]] },
  ];
  it("draws only the ticked roles, mirrored domains without a legend entry", () => {
    const r = buildRegionTraces(rs, new Set<RegionRole>(["domain"]), box, tokens);
    expect(r.map((t) => [t["name"], t["showlegend"]])).toEqual([
      ["Design domains", true],
      ["Design domains", false],
    ]);
    expect(buildRegionTraces(rs, new Set(), box, tokens)).toEqual([]);
    expect(buildRegionTraces(undefined, new Set<RegionRole>(["domain"]), box, tokens)).toEqual([]);
  });
  it("clips to the box and drops what is fully outside", () => {
    const [keep] = buildRegionTraces(rs, new Set<RegionRole>(["outside"]), box, tokens);
    const xs = (keep!["x"] as (number | null)[]).filter((v): v is number => v !== null);
    expect(Math.max(...xs)).toBeCloseTo(1, 12);
    expect(buildRegionTraces(rs, new Set<RegionRole>(["inside"]), box, tokens)).toEqual([]);
  });
  it("styles by token and dash", () => {
    const [d] = buildRegionTraces(rs, new Set<RegionRole>(["domain"]), box, tokens);
    expect((d!["line"] as { color: string; dash: string }).color).toBe("#0a0");
    const [o] = buildRegionTraces(rs, new Set<RegionRole>(["outside"]), box, tokens);
    expect((o!["line"] as { dash: string }).dash).toBe("dash");
  });
});
