import { memo, useCallback, useMemo, useRef } from "react";
import type { DatasetInfo, FigureSpec, PanelSpec, SpecDoc } from "../api";
import { useTables } from "../api/hooks";
import { selectPanels } from "../lib/panels";
import { buildPanel, type RevisionData, type Table } from "../lib/traces";
import { useUi, visibleIds, type Mode } from "../state/store";
import { revisionColor, type Theme } from "../theme/revisions";
import { chartLayout, readTokens } from "../theme/plotlyTemplate";
import { PlotlyChart, type ChartHandle } from "./PlotlyChart";

interface PanelChartProps {
  fig: FigureSpec;
  panel: PanelSpec;
  revs: RevisionData[];
  reference: RevisionData | null;
  mode: Mode;
  linestyles: string[];
  onHoverX: (panelId: string, x: number | null) => void;
  chartRef: (panelId: string, handle: ChartHandle | null) => void;
  onPointer: (panelId: string | null) => void;
}

const PanelChart = memo(function PanelChart(p: PanelChartProps) {
  const { fig, panel, revs, reference, mode, linestyles, onHoverX, chartRef, onPointer } = p;

  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const plot = buildPanel(fig, panel, revs, reference, mode, linestyles, t.inkMuted);
      return {
        data: plot.traces,
        layout: chartLayout(t, {
          xTitle: plot.xTitle,
          yTitle: plot.yTitle,
          legend: plot.legend,
          yZero: mode === "delta",
        }),
      };
    },
    [fig, panel, revs, reference, mode, linestyles],
  );

  const icDropped = useMemo(
    () => buildPanel(fig, panel, revs, reference, mode, linestyles, "").icDropped,
    [fig, panel, revs, reference, mode, linestyles],
  );
  const setRef = useCallback((h: ChartHandle | null) => chartRef(panel.id, h), [chartRef, panel.id]);

  return (
    <section
      className="panel"
      aria-label={panel.title}
      onMouseEnter={() => onPointer(panel.id)}
      onMouseLeave={() => onPointer(null)}
    >
      <h3 className="panel-title">{panel.title}</h3>
      {panel.ic && icDropped > 0 && (
        <p className="panel-note">IC at infinity: {icDropped} points not drawn</p>
      )}
      <PlotlyChart
        ref={setRef}
        build={build}
        ariaLabel={`${panel.title} versus ${(panel.x ?? fig.x).label}`}
        filename={`${fig.id}_${panel.id}`}
        onHoverX={(x) => onHoverX(panel.id, x)}
      />
    </section>
  );
});

interface Props {
  fig: FigureSpec;
  spec: SpecDoc;
  datasets: DatasetInfo[];
  /** Show the figure title above the panels. */
  showTitle?: boolean;
}

/** Renders one FigureSpec: every panel overlays the visible revisions (or their delta). */
export function SpecFigure({ fig, spec, datasets, showTitle }: Props) {
  const order = useUi((s) => s.order);
  const hidden = useUi((s) => s.hidden);
  const letters = useUi((s) => s.letters);
  const reference = useUi((s) => s.reference);
  const mode = useUi((s) => s.mode);
  const theme = useUi((s) => s.theme);

  const byId = useMemo(() => new Map(datasets.map((d) => [d.id, d])), [datasets]);
  const visible = useMemo(() => visibleIds({ order, hidden }).filter((id) => byId.has(id)), [order, hidden, byId]);
  const wanted = useMemo(() => {
    const ids = [...visible];
    if (mode === "delta" && reference && byId.has(reference) && !ids.includes(reference)) ids.push(reference);
    return ids.filter((id) => byId.get(id)?.studies.includes(fig.study));
  }, [visible, mode, reference, byId, fig.study]);

  const results = useTables(wanted.map((id) => ({ id, study: fig.study })));
  const pending = results.some((r) => r.isPending);
  const failed = results.find((r) => r.isError);

  const sig = results.map((r) => r.dataUpdatedAt).join(",");
  const tables = useMemo(() => {
    const m = new Map<string, Table>();
    wanted.forEach((id, i) => {
      const d = results[i]?.data;
      if (d) m.set(id, d.columns);
    });
    return m;
    // `sig` changes exactly when a table arrives
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wanted, sig]);

  const makeRev = useCallback(
    (id: string): RevisionData | null => {
      const table = tables.get(id);
      const d = byId.get(id);
      if (!table || !d) return null;
      return {
        id,
        letter: letters[id] ?? "?",
        color: revisionColor(d.color, spec.palette, theme),
        table,
      };
    },
    [tables, byId, letters, spec.palette, theme],
  );

  const revs = useMemo(
    () => visible.map(makeRev).filter((r): r is RevisionData => r !== null),
    [visible, makeRev],
  );
  const ref = useMemo(() => (reference ? makeRev(reference) : null), [reference, makeRev]);

  const panels = useMemo(() => {
    const sets = [...revs, ...(mode === "delta" && ref ? [ref] : [])].map((r) => new Set(Object.keys(r.table)));
    return selectPanels(fig, sets);
  }, [fig, revs, ref, mode]);

  // linked hover: the chart under the pointer drives the others
  const handles = useRef(new Map<string, ChartHandle>());
  const active = useRef<string | null>(null);
  const chartRef = useCallback((id: string, h: ChartHandle | null) => {
    if (h) handles.current.set(id, h);
    else handles.current.delete(id);
  }, []);
  const onPointer = useCallback((id: string | null) => {
    if (id === null) {
      const prev = active.current;
      active.current = null;
      handles.current.forEach((h, k) => k !== prev && h.clearHover());
    } else {
      active.current = id;
    }
  }, []);
  const onHoverX = useCallback((id: string, x: number | null) => {
    if (active.current !== id) return;
    handles.current.forEach((h, k) => {
      if (k === id) return;
      if (x === null) h.clearHover();
      else h.hoverAt(x);
    });
  }, []);

  const hint =
    mode === "delta" && revs.filter((r) => r.id !== ref?.id).length === 0
      ? "Show a second revision to see its difference from the reference."
      : null;

  return (
    <div className="figure">
      {showTitle && <h2 className="figure-title">{fig.title}</h2>}
      {failed && (
        <p className="error" role="alert">
          Could not load the {fig.study} results: {String(failed.error?.message ?? "unknown error")}
        </p>
      )}
      {pending && wanted.length > 0 && <p className="muted">Loading curves.</p>}
      {hint && <p className="muted">{hint}</p>}
      {!pending && !failed && panels.length === 0 && (
        <p className="muted">The visible revisions have no {fig.study} results.</p>
      )}
      <div className="panel-grid">
        {panels.map((panel) => (
          <PanelChart
            key={panel.id}
            fig={fig}
            panel={panel}
            revs={revs}
            reference={ref}
            mode={mode}
            linestyles={spec.linestyles}
            onHoverX={onHoverX}
            chartRef={chartRef}
            onPointer={onPointer}
          />
        ))}
      </div>
    </div>
  );
}
