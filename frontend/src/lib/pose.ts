import type { PoseInput, ScenePose } from "../api/types";

export type PoseKey = PoseInput["name"];

/** Pose in display units (the units named by the backend's pose metadata). */
export type Pose = Record<PoseKey, number>;

export const ZERO_POSE: Pose = { heave: 0, roll: 0, pitch: 0, rack: 0 };

export interface PoseAxis {
  key: PoseKey;
  label: string;
  unit: string;
  max: number;
  step: number;
}

const positive = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null);

/** Largest magnitude of a study parameter that is a number or a list of numbers. */
function magnitude(v: unknown): number | null {
  if (Array.isArray(v)) {
    const m = Math.max(...v.map((x) => Math.abs(Number(x))));
    return Number.isFinite(m) && m > 0 ? m : null;
  }
  return positive(v);
}

/** Slider axes; ranges (+-max) come from the dataset's study parameters, units from the backend. */
export function poseAxes(params: Record<string, unknown>, inputs: readonly PoseInput[]): PoseAxis[] {
  return inputs.map((i) => {
    const m = magnitude(params[i.param]);
    return {
      key: i.name,
      label: i.label,
      unit: i.display_unit,
      max: m === null ? i.default_max : m * i.param_factor,
      step: i.step,
    };
  });
}

/** Display units to the API's SI units, using the backend's conversion factors. */
export function poseToSi(p: Pose, inputs: readonly PoseInput[]): ScenePose {
  const out: ScenePose = {};
  for (const i of inputs) out[i.name] = p[i.name] / i.factor;
  return out;
}
