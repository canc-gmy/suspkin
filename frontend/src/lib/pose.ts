import type { ScenePose } from "../api/types";

export type PoseKey = "heave" | "roll" | "pitch" | "rack";

/** Pose in display units: heave and rack in mm, roll and pitch in deg. */
export type Pose = Record<PoseKey, number>;

export const ZERO_POSE: Pose = { heave: 0, roll: 0, pitch: 0, rack: 0 };

export interface PoseAxis {
  key: PoseKey;
  label: string;
  unit: "mm" | "deg";
  max: number;
  step: number;
}

const FALLBACK = { heave: 25, roll: 2, pitch: 1.5, rack: 30 };

const positive = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null);

/** Slider axes; ranges (+-max) come from the dataset's study parameters, with fallbacks. */
export function poseAxes(params: Record<string, unknown>): PoseAxis[] {
  const mm = (v: unknown) => {
    const x = positive(v);
    return x === null ? null : x * 1000;
  };
  const rack = Array.isArray(params["rack"])
    ? Math.max(...(params["rack"] as unknown[]).map((v) => Math.abs(Number(v)) * 1000))
    : mm(params["rack"]);
  return [
    { key: "heave", label: "Heave", unit: "mm", max: mm(params["heave"]) ?? FALLBACK.heave, step: 0.5 },
    { key: "roll", label: "Roll", unit: "deg", max: positive(params["roll_deg"]) ?? FALLBACK.roll, step: 0.1 },
    { key: "pitch", label: "Pitch", unit: "deg", max: positive(params["pitch_deg"]) ?? FALLBACK.pitch, step: 0.1 },
    {
      key: "rack",
      label: "Rack",
      unit: "mm",
      max: rack !== null && Number.isFinite(rack) && rack > 0 ? rack : FALLBACK.rack,
      step: 0.5,
    },
  ];
}

/** Display units to the API's SI units (m, rad). */
export function poseToSi(p: Pose): Required<ScenePose> {
  const rad = Math.PI / 180;
  return { heave: p.heave / 1000, roll: p.roll * rad, pitch: p.pitch * rad, rack: p.rack / 1000 };
}
