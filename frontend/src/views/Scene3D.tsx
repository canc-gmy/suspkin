import { useEffect, useMemo, useRef, useState } from "react";
import type { PlotlyHTMLElement } from "plotly.js-dist-min";
import type { DatasetInfo, PoseInput } from "../api";
import { useHardpoints, useHealth, useRegions, useScene, useSource } from "../api/hooks";
import { Disclosure } from "../components/Disclosure";
import { OverlaysMenu } from "../components/OverlaysMenu";
import { loadCentreKinds, loadRegionRoles, saveCentreKinds, saveRegionRoles } from "../lib/centresMenu";
import { changedSceneNames } from "../lib/hardpoints";
import { pickRegionSource, regionSources } from "../lib/regionSource";
import { poseAxes, poseToSi, ZERO_POSE, type Pose } from "../lib/pose";
import { useUi } from "../state/store";
import { readTokens } from "../theme/plotlyTemplate";
import {
  buildCentreTraces,
  buildRegionTraces,
  buildSceneFigure,
  cameraForLayout,
  cameraFromRelayout,
  fitCamera,
  frameBox,
  parseCentres,
  posedHardpoints,
  sceneBounds,
  sceneFrame,
  type Camera,
  type CentreKind,
  type RegionRole,
} from "../theme/scene3d";
import "./Scene3D.css";

const NO_POSE: PoseInput[] = [];

type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule["default"]> | null = null;

/** Same lazy load as PlotlyChart: Plotly is large, so it loads on first use. */
function loadPlotly(): Promise<PlotlyModule["default"]> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m) => m.default);
  return plotlyPromise;
}

/** `relayout` is missing from the bundled typings. */
type Relayout = { relayout: (el: HTMLElement, update: Record<string, unknown>) => Promise<unknown> };

const MIN_CANVAS = 320;
/** Below this width the legend starts closed, so the canvas keeps its room on a phone. */
const WIDE_VIEWPORT = 900;
/** Height of the strip above the scene while the legend is on (matches the layout margin). */
const LEGEND_STRIP = 48;

/** Camera that frames the whole car in the canvas as it is now (see `fitCamera`). */
function framedCamera(el: HTMLElement | null, frame: Parameters<typeof fitCamera>[0], legend: boolean): Camera {
  const r = el?.getBoundingClientRect();
  // the scene sits below the strip reserved for the legend
  return fitCamera(frame, r ? { width: r.width, height: r.height - (legend ? LEGEND_STRIP : 8) } : null);
}

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
  const poseSi = poseToSi(readOnly ? ZERO_POSE : debounced, poseInputs);
  const scene = useScene(id, poseSi);
  const [ghostOn, setGhostOn] = useState(false);
  const [legendOn, setLegendOn] = useState(() => typeof window === "undefined" || window.innerWidth >= WIDE_VIEWPORT);
  const ghostId = ghostOn && reference && reference !== id && datasets.some((d) => d.id === reference) ? reference : null;
  const ghost = useScene(ghostId, poseSi);
  const staticScene = useScene(id, poseToSi(ZERO_POSE, poseInputs));

  // keep the last figure that solved, so an unsolvable pose does not blank the view
  const last = useRef<Record<string, unknown> | null>(null);
  if (scene.data) last.current = scene.data;
  const figure = scene.data ?? last.current;

  const frame = useMemo(() => {
    const b = sceneBounds(staticScene.data ?? figure);
    return b ? sceneFrame(b) : null;
    // the frame follows the revision, never the pose
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [staticScene.data, id]);
  const ghostFigure = ghostId ? (ghost.data ?? null) : null;

  // rings on the points that moved more than 0.1 mm against the reference, at the posed positions
  const [highlightOn, setHighlightOn] = useState(true);
  const isRef = id !== null && id === reference;
  const compare = highlightOn && !isRef && id !== null && reference !== null && datasets.some((d) => d.id === reference);
  const hp = useHardpoints(compare ? [id as string, reference as string] : []);
  const mineHp = hp[0]?.data;
  const refHp = hp[1]?.data;
  const highlight = useMemo(() => {
    if (!compare) return [];
    const posed = posedHardpoints(figure as { hardpoints?: unknown } | null);
    return changedSceneNames(mineHp ?? null, refHp ?? null)
      .map((n) => posed[n])
      .filter((p): p is NonNullable<typeof p> => p !== undefined);
  }, [compare, figure, mineHp, refHp]);

  // rotation centres: clipped to the fixed scene box, so they never move the ranges or the camera
  const [centreKinds, setCentreKinds] = useState<Set<CentreKind>>(loadCentreKinds);
  const theme = useUi((s) => s.theme);
  // study regions: static in the body frame (like the chassis points), clipped to the same box
  const [regionRoles, setRegionRoles] = useState<Set<RegionRole>>(loadRegionRoles);
  // the study is picked independently of the revision; its regions are body-frame geometry, drawn
  // in the selected revision's scene (which is already in the body frame at the current pose)
  const setOpenDialog = useUi((s) => s.setOpenDialog);
  const [studyPick, setStudyPick] = useState<string | null>(null);
  const studies = useMemo(() => regionSources(datasets, letters), [datasets, letters]);
  const studyId = pickRegionSource(studies, id, studyPick);
  const regionsQuery = useRegions(studyId && regionRoles.size > 0 ? studyId : null);
  const regionTraces = useMemo(
    () => buildRegionTraces(regionsQuery.data, regionRoles, frameBox(frame), readTokens(theme)),
    [regionsQuery.data, regionRoles, frame, theme],
  );
  const centres = useMemo(
    () =>
      buildCentreTraces(
        parseCentres(figure as { centres?: unknown } | null),
        centreKinds,
        frameBox(frame),
        readTokens(theme),
      ),
    [figure, centreKinds, frame, theme],
  );
  const host = useRef<HTMLDivElement>(null);
  const top = useRef<HTMLDivElement>(null);
  const rendered = useRef(false);
  const savedCamera = useRef<Camera | null>(null);
  const [canvasTop, setCanvasTop] = useState(0);

  useEffect(() => {
    const el = host.current;
    if (!el || !figure) return;
    let cancelled = false;
    void loadPlotly().then(async (P) => {
      if (cancelled) return;
      const fig = buildSceneFigure({
        figure,
        ghost: ghostFigure,
        tokens: readTokens(theme),
        frame,
        highlight,
        centres: [...regionTraces, ...centres.traces],
        camera: cameraForLayout({ reset: false, saved: savedCamera.current, fallback: framedCamera(el, frame, legendOn) }),
        legend: legendOn,
      });
      const gd = await P.react(el, fig.data, fig.layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: "hover",
        modeBarButtons: [["toImage", "resetCameraDefault3d"]],
      });
      if (cancelled || rendered.current) return;
      rendered.current = true;
      (gd as PlotlyHTMLElement).on("plotly_relayout", (ev) => {
        const c = cameraFromRelayout(ev as Record<string, unknown>);
        if (c) savedCamera.current = c;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [figure, ghostFigure, frame, theme, highlight, centres, regionTraces, legendOn]);

  const resetView = () => {
    const el = host.current;
    if (!el || !rendered.current) return;
    const camera = framedCamera(el, frame, legendOn);
    savedCamera.current = camera;
    void loadPlotly().then((P) => (P as unknown as Relayout).relayout(el, { "scene.camera": camera }));
  };

  // canvas height: from the bottom of the toolbar block to the bottom of the viewport
  useEffect(() => {
    const el = top.current;
    if (!el) return;
    const measure = () => setCanvasTop(Math.round(el.getBoundingClientRect().bottom + window.scrollY));
    measure();
    window.addEventListener("resize", measure);
    if (typeof ResizeObserver === "undefined") return () => window.removeEventListener("resize", measure);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => {
      window.removeEventListener("resize", measure);
      ro.disconnect();
    };
  }, []);

  useEffect(() => {
    const el = host.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    let frameId = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(() => {
        if (rendered.current) void loadPlotly().then((P) => P.Plots.resize(el));
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(frameId);
      ro.disconnect();
    };
  }, []);

  useEffect(() => {
    const el = host.current;
    return () => {
      rendered.current = false;
      if (el) void loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  const error = scene.isError
    ? `Could not show this pose: ${scene.error.message.replace(/^pose not solvable:\s*/, "")}. Move a slider back toward zero or press Reset pose.`
    : null;

  return (
    <section aria-label="3D" className="scene3d">
      <div ref={top}>
      <div className="toolbar">
        <h2 className="view-title">3D</h2>
        <label className="field">
          <span>Revision</span>
          <select
            aria-label="Revision"
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
        <button type="button" className="btn secondary" onClick={resetView}>
          Reset view
        </button>
      </div>
      <div className="toolbar options-bar">
        <OverlaysMenu
          centres={centreKinds}
          onCentres={(next) => {
            setCentreKinds(next);
            saveCentreKinds(next);
          }}
          regions={regionRoles}
          onRegions={(next) => {
            setRegionRoles(next);
            saveRegionRoles(next);
          }}
          sources={studies}
          source={studyId}
          onSource={setStudyPick}
          onOpenFiles={() => setOpenDialog(true)}
        />
        <button type="button" className="btn secondary" aria-pressed={legendOn} onClick={() => setLegendOn(!legendOn)}>
          Legend
        </button>
        {centres.notes.length > 0 && (
          <Disclosure label={`Notes (${centres.notes.length})`}>
            {() => (
              <ul className="centres-notes" aria-label="Centres outside the view">
                {centres.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
          </Disclosure>
        )}
        <label className="field">
          <input type="checkbox" aria-label="Show reference ghost" checked={ghostOn} onChange={(e) => setGhostOn(e.target.checked)} />
          <span>Show reference ghost</span>
        </label>
        <label className="field" title={isRef ? "Same as the reference" : undefined}>
          <input
            type="checkbox"
            aria-label="Highlight changes"
            checked={highlightOn && !isRef} disabled={isRef} onChange={(e) => setHighlightOn(e.target.checked)} />
          <span>Highlight changes</span>
        </label>
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
      </div>
      <div
        ref={host}
        className="scene3d-canvas"
        role="img"
        aria-label={`3D view of ${ds?.label ?? "the revision"}`}
        style={{ height: `max(${MIN_CANVAS}px, calc(100dvh - ${canvasTop}px))` }}
      />
    </section>
  );
}
