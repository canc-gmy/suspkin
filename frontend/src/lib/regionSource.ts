import type { DatasetInfo } from "../api";

/** One revision whose study can supply the regions. */
export interface RegionSource {
  id: string;
  name: string;
}

/** Hint under the "Study regions" heading while no open revision has a study. */
export const NO_STUDY_HINT = "Open a study YAML or an optimizer run to show its regions.";

/** Every open revision with a study, named by letter and label, in the open order. */
export function regionSources(
  datasets: readonly Pick<DatasetInfo, "id" | "label" | "has_study">[],
  letters: Readonly<Record<string, string>>,
): RegionSource[] {
  return datasets.filter((d) => d.has_study).map((d) => ({ id: d.id, name: `${letters[d.id] ?? "?"}  ${d.label}` }));
}

/**
 * The study whose regions are drawn: the user's pick if still open, else the selected revision's
 * own study, else the first open revision with a study; null when none has one.
 */
export function pickRegionSource(
  sources: readonly RegionSource[],
  selectedId: string | null,
  chosen: string | null,
): string | null {
  if (chosen && sources.some((s) => s.id === chosen)) return chosen;
  if (selectedId && sources.some((s) => s.id === selectedId)) return selectedId;
  return sources[0]?.id ?? null;
}
