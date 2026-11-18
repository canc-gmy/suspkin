import type { DatasetInfo } from "../api";
import { useSetResolution, useSource } from "../api/hooks";
import { READ_ONLY_TIP } from "../api/snapshotSource";
import { Segmented } from "./Segmented";

const STEPS = [31, 61, 121];

/** The Resolution control value: the shared step count of the movable revisions, or "" if mixed. */
export function sharedSteps(datasets: readonly DatasetInfo[]): string {
  const movable = datasets.filter((d) => d.resolution_fixed === false);
  const first = movable[0]?.steps;
  return first !== undefined && STEPS.includes(first) && movable.every((d) => d.steps === first) ? String(first) : "";
}

/** Points per sweep for revisions opened from vehicle files; bundles and runs keep their stored points. */
export function ResolutionControl({ datasets }: { datasets: DatasetInfo[] | undefined }) {
  const readOnly = useSource().readOnly;
  const set = useSetResolution();
  const movable = (datasets ?? []).filter((d) => d.resolution_fixed === false);
  const disabled = readOnly || movable.length === 0 || set.isPending;
  const tip = readOnly
    ? READ_ONLY_TIP
    : movable.length === 0
      ? "Open a vehicle file to change the resolution. Bundles and runs keep their stored points."
      : "Points per sweep";

  return (
    <span className="resolution" title={tip}>
      <span className="muted">Resolution</span>
      <Segmented<string>
        label="Resolution in points per sweep"
        value={sharedSteps(datasets ?? [])}
        disabled={disabled}
        options={STEPS.map((n) => ({ value: String(n), label: String(n) }))}
        onChange={(v) => set.mutate({ ids: movable.map((d) => d.id), steps: Number(v) })}
      />
      {set.isPending && (
        <span className="resolution-note" role="status">
          Solving…
        </span>
      )}
      {set.isError && (
        <span className="error" role="alert">
          {`Could not change the resolution: ${set.error.message}`}
        </span>
      )}
    </span>
  );
}
