import { describe, expect, it } from "vitest";
import { poseAxes, poseToSi, ZERO_POSE } from "./pose";

describe("pose", () => {
  it("takes slider ranges from the study parameters", () => {
    const ax = poseAxes({ heave: 0.03, roll_deg: 2.5, pitch_deg: 1, rack: 0.04 });
    expect(ax.map((a) => [a.key, a.max, a.unit])).toEqual([
      ["heave", 30, "mm"],
      ["roll", 2.5, "deg"],
      ["pitch", 1, "deg"],
      ["rack", 40, "mm"],
    ]);
  });

  it("falls back when parameters are missing or unusable", () => {
    const ax = poseAxes({ heave: null, roll_deg: "x", rack: [-0.02, 0.01] });
    expect(ax.map((a) => a.max)).toEqual([25, 2, 1.5, 20]);
    expect(poseAxes({}).map((a) => a.max)).toEqual([25, 2, 1.5, 30]);
  });

  it("converts to SI", () => {
    const si = poseToSi({ heave: 10, roll: 1, pitch: -2, rack: 5 });
    expect(si.heave).toBeCloseTo(0.01, 12);
    expect(si.roll).toBeCloseTo(Math.PI / 180, 12);
    expect(si.pitch).toBeCloseTo(-2 * (Math.PI / 180), 12);
    expect(si.rack).toBeCloseTo(0.005, 12);
    expect(poseToSi(ZERO_POSE)).toEqual({ heave: 0, roll: 0, pitch: 0, rack: 0 });
  });
});
