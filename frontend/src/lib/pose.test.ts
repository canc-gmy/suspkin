import { describe, expect, it } from "vitest";
import type { PoseInput } from "../api/types";
import { poseAxes, poseToSi, ZERO_POSE } from "./pose";

const RAD = Math.PI / 180;
// what the backend sends (SI -> display factors come from suspkin.units)
const INPUTS: PoseInput[] = [
  { name: "heave", label: "Heave", si_unit: "m", display_unit: "mm", factor: 1000, param: "heave", param_factor: 1000, default_max: 25, step: 0.5 },
  { name: "roll", label: "Roll", si_unit: "rad", display_unit: "deg", factor: 1 / RAD, param: "roll_deg", param_factor: 1, default_max: 2, step: 0.1 },
  { name: "pitch", label: "Pitch", si_unit: "rad", display_unit: "deg", factor: 1 / RAD, param: "pitch_deg", param_factor: 1, default_max: 1.5, step: 0.1 },
  { name: "rack", label: "Rack", si_unit: "m", display_unit: "mm", factor: 1000, param: "rack", param_factor: 1000, default_max: 30, step: 0.5 },
];

describe("pose", () => {
  it("takes slider ranges from the study parameters", () => {
    const ax = poseAxes({ heave: 0.03, roll_deg: 2.5, pitch_deg: 1, rack: 0.04 }, INPUTS);
    expect(ax.map((a) => [a.key, a.max, a.unit])).toEqual([
      ["heave", 30, "mm"],
      ["roll", 2.5, "deg"],
      ["pitch", 1, "deg"],
      ["rack", 40, "mm"],
    ]);
  });

  it("falls back to the backend defaults when parameters are missing or unusable", () => {
    const ax = poseAxes({ heave: null, roll_deg: "x", rack: [-0.02, 0.01] }, INPUTS);
    expect(ax.map((a) => a.max)).toEqual([25, 2, 1.5, 20]);
    expect(poseAxes({}, INPUTS).map((a) => a.max)).toEqual([25, 2, 1.5, 30]);
    expect(poseAxes({}, [])).toEqual([]);
  });

  it("converts to SI with the backend factors", () => {
    const si = poseToSi({ heave: 10, roll: 1, pitch: -2, rack: 5 }, INPUTS);
    expect(si.heave).toBeCloseTo(0.01, 12);
    expect(si.roll).toBeCloseTo(RAD, 12);
    expect(si.pitch).toBeCloseTo(-2 * RAD, 12);
    expect(si.rack).toBeCloseTo(0.005, 12);
    expect(poseToSi(ZERO_POSE, INPUTS)).toEqual({ heave: 0, roll: 0, pitch: 0, rack: 0 });
  });
});
