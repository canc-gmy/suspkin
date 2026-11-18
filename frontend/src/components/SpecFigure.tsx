import { memo, useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { DatasetInfo, FigureSpec, PanelSpec, SpecDoc } from "../api";
import { useTables } from "../api/hooks";
import { selectPanels } from "../lib/panels";
import { buildReadout, clampX, formatX, nudge } from "../lib/readout";
import { buildPanel, scaled, type RevisionData, type Table } from "../lib/traces";
import { useUi, visibleIds, type Mode } from "../state/store";
import { revisionColor, type Theme } from "../theme/revisions";
import { chartLayout, readTokens } from "../theme/plotlyTemplate";
import { PlotlyChart, type ChartCursor } from "./PlotlyChart";
import { Readout } from "./Readout";

interface PanelChartProps {
  fig: FigureSpec;
  panel: PanelSpec;
  revs: RevisionData[];
  reference: RevisionData | null;
  mode: Mode;
  linestyles: string[];
  showPoints: boolean;
  icDropped: number;
  /** Cursor line; null for a panel whose x axis is not the study x. */
  cursor: ChartCursor | null;
  onPointerX?: (x: number | null) => void;
  onPointerClick?: (x: number) => void;
}

const PanelChart = memo(function PanelChart(p: PanelChartProps) {
  const { fig, panel, revs, reference, mode, linestyles, showPoints, icDropped, cursor, onPointerX, onPointerClick } = p;

  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const plot = buildPanel(fig, panel, revs, reference, mode, linestyles, t.inkMuted, { showPoints });
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
    [fig, panel, revs, reference, mode, linestyles, showPoints],
  );

  return (
    <section className="panel" aria-label={panel.title}>
      <h3 className="panel-title">{panel.title}</h3>
      {panel.ic && icDropped > 0 && <p className="panel-note">IC at infinity: {icDropped} points not drawn</p>}
      <PlotlyChart
        build={build}
        ariaLabel={`${panel.title} versus ${(panel.x ?? fig.x).label}`}
        filename={`${fig.id}_${panel.id}`}
        cursor={cursor}
        onPointerX={onPointerX}
        onPointerClick={onPointerClick}
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

  const showPoints = useUi((s) => s.showPoints);
  const setShowPoints = useUi((s) => s.setShowPoints);
  const delta = mode === "delta";

  const plots = useMemo(
    () =>
      panels.map((panel) => ({
        panel,
        plot: buildPanel(fig, panel, revs, ref, mode, spec.linestyles, "", { showPoints }),
      })),
    [panels, fig, revs, ref, mode, spec.linestyles, showPoints],
  );

  // the cursor lives on the study x: it follows the pointer, or stays where it was pinned
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [pinnedX, setPinnedX] = useState<number | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const frame = useRef(0);
  const latest = useRef<number | null>(null);
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const domain = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of [...revs, ...(ref ? [ref] : [])]) {
      for (const v of scaled(r.table, fig.x.column, fig.x.factor) ?? []) {
        if (v === null) continue;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    return lo <= hi ? ([lo, hi] as const) : null;
  }, [revs, ref, fig.x.column, fig.x.factor]);

  const onPointerX = useCallback((x: number | null) => {
    latest.current = x;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => setHoverX(latest.current));
  }, []);
  const onPin = useCallback(
    (x: number) => {
      setPinnedX(clampX(x, domain));
      root.current?.focus({ preventScroll: true });
    },
    [domain],
  );
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      setPinnedX(null);
      return;
    }
    const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement;
    if (pinnedX === null || typing) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      setPinnedX(clampX(nudge(pinnedX, e.key === "ArrowLeft" ? -1 : 1, e.shiftKey, fig.x), domain));
    }
  };

  const cursorX = pinnedX ?? hoverX;
  const cursor = useMemo<ChartCursor>(() => ({ x: cursorX, pinned: pinnedX !== null }), [cursorX, pinnedX]);
  const table = useMemo(
    () => buildReadout(plots.map(({ panel, plot }) => ({ panel, curves: plot.curves })), revs, ref?.id ?? null, delta, cursorX),
    [plots, revs, ref, delta, cursorX],
  );
  const icOf = (id: string) => plots.find((p) => p.panel.id === id)?.plot.icDropped ?? 0;
  const xLabel = fig.x.label.charAt(0).toUpperCase() + fig.x.label.slice(1);

  const hint =
    mode === "delta" && revs.filter((r) => r.id !== ref?.id).length === 0
      ? "Show a second revision to see its difference from the reference."
      : null;

  return (
    <div className="figure" ref={root} tabIndex={0} role="group" aria-label={fig.title} onKeyDown={onKeyDown}>
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
      {panels.length > 0 && (
        <div className="figure-sticky">
          <Readout
            xLabel={xLabel}
            xUnit={fig.x.unit}
            x={cursorX}
            formatX={(v) => formatX(v, fig.x)}
            pinned={pinnedX !== null}
            table={table}
            showPoints={showPoints}
            onShowPoints={setShowPoints}
            onPinX={onPin}
          />
        </div>
      )}
      <div className="panel-grid">
        {panels.map((panel) => {
          const onStudyX = (panel.x ?? fig.x).column === fig.x.column;
          return (
            <PanelChart
              key={panel.id}
              fig={fig}
              panel={panel}
              revs={revs}
              reference={ref}
              mode={mode}
              linestyles={spec.linestyles}
              showPoints={showPoints}
              icDropped={icOf(panel.id)}
              cursor={onStudyX ? cursor : null}
              onPointerX={onStudyX ? onPointerX : undefined}
              onPointerClick={onStudyX ? onPin : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}
