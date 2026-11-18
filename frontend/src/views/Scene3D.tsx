import { useCallback, useEffect, useRef, useState } from "react";
import type { DatasetInfo, PoseInput } from "../api";
import { useHealth, useScene, useSource } from "../api/hooks";
import { PlotlyChart } from "../components/PlotlyChart";
import { poseAxes, poseToSi, ZERO_POSE, type Pose } from "../lib/pose";
import { useUi } from "../state/store";
import { readTokens, themeScene } from "../theme/plotlyTemplate";
import type { Theme } from "../theme/revisions";

const NO_POSE: PoseInput[] = [];
const SCENE_BUTTONS = ["resetCameraDefault3d"];

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** 3D view of one revision at an adjustable body pose. */
export function Scene3D({ datasets }: { datasets: DatasetInfo[] }) {
  const readOnly = useSource().readOnly;
  const poseInputs = useHealth().data?.pose ?? NO_POSE;
  const letters = useUi((s) => s.letters);
  const reference = useUi((s) => s.reference);
  const [selected, setSelected] = useState<string | null>(null);
  const [pose, setPose] = useState<Pose>(ZERO_POSE);

  const id =
    (selected && datasets.some((d) => d.id === selected) ? selected : null) ??
    (reference && datasets.some((d) => d.id === reference) ? reference : null) ??
    datasets[0]?.id ??
    null;
  const ds = datasets.find((d) => d.id === id);
  const axes = poseAxes((ds?.params ?? {}) as Record<string, unknown>, poseInputs);

  const debounced = useDebounced(pose, 150);
  const scene = useScene(id, poseToSi(readOnly ? ZERO_POSE : debounced, poseInputs));

  // keep the last figure that solved, so an unsolvable pose does not blank the view
  const last = useRef<Record<string, unknown> | null>(null);
  if (scene.data) last.current = scene.data;
  const figure = scene.data ?? last.current;

  const build = useCallback(
    (theme: Theme) => themeScene(figure ?? {}, readTokens(theme)),
    [figure],
  );

  const error = scene.isError
    ? `Could not show this pose: ${scene.error.message.replace(/^pose not solvable:\s*/, "")}. Move a slider back toward zero or press Reset pose.`
    : null;

  return (
    <section aria-label="3D">
      <div className="toolbar">
        <h2 className="view-title">3D</h2>
        <label className="field">
          <span>Revision</span>
          <select
            value={id ?? ""}
            onChange={(e) => {
              setSelected(e.target.value);
              setPose(ZERO_POSE);
            }}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {`${letters[d.id] ?? "?"}  ${d.label}`}
              </option>
            ))}
          </select>
        </label>
        <span className="spacer" />
        <button type="button" className="btn secondary" disabled={readOnly} onClick={() => setPose(ZERO_POSE)}>
          Reset pose
        </button>
      </div>

      <div className="sliders">
        {axes.map((a) => (
          <label key={a.key} className="slider">
            <span className="slider-head">
              <span>{a.label}</span>
              <output className="slider-value">{`${pose[a.key].toFixed(1)} ${a.unit}`}</output>
            </span>
            <input
              type="range"
              min={-a.max}
              max={a.max}
              step={a.step}
              value={pose[a.key]}
              disabled={readOnly}
              aria-label={`${a.label} [${a.unit}]`}
              onChange={(e) => setPose({ ...pose, [a.key]: Number(e.target.value) })}
            />
          </label>
        ))}
      </div>
      {readOnly && <p className="muted">Static pose in snapshots.</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {scene.isPending && !figure && <p className="muted">Loading the 3D view.</p>}
      {figure && (
        <PlotlyChart
          build={build}
          ariaLabel={`3D view of ${ds?.label ?? "the revision"}`}
          filename={`scene_${ds?.label ?? "revision"}`}
          height={600}
          buttons={SCENE_BUTTONS}
        />
      )}
    </section>
  );
}
