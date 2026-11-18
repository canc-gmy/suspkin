import type { FigureSpec, PanelSpec, SeriesSpec } from "../api/spec";
import { LINE_WIDTH, type PlotlyObject } from "../theme/plotlyTemplate";
import { buildSpline, drawPoints, type Spline } from "./interp";
import { seriesIn } from "./panels";
import { withUnit } from "./format";

type Num = number | null;

export type Table = Readonly<Record<string, readonly (number | string | null)[]>>;

/** One revision as the chart builder sees it. */
export interface RevisionData {
  id: string;
  letter: string;
  color: string;
  table: Table;
}

export type TraceMode = "overlay" | "delta";

/** One drawn series of one revision, with the function the readout evaluates. */
export interface PanelCurve {
  revId: string;
  letter: string;
  color: string;
  /** Index in `panel.series` (it picks the line style). */
  seriesIndex: number;
  label: string;
  side: SeriesSpec["side"];
  unit: string;
  /** Value at study x `t` (display units); in delta mode the difference to the reference. */
  at: (t: number) => number | null;
}

export interface PanelPlot {
  traces: PlotlyObject[];
  /** Panel legend (series labels) shown: more than one drawn series. */
  legend: boolean;
  xTitle: string;
  yTitle: string;
  /** Curves the readout evaluates, in draw order. */
  curves: PanelCurve[];
  /** Points not drawn because the curve is undefined there (instant centre at infinity). */
  icDropped: number;
}

export interface BuildOptions {
  /** Add small markers at the real solver points. */
  showPoints?: boolean;
}

const DASH: Record<string, string> = { "-": "solid", "--": "dash", ":": "dot", "-.": "dashdot" };

/** Plotly dash name of a matplotlib-style line style from `spec.linestyles`. */
export function plotlyDash(style: string): string {
  return DASH[style] ?? "solid";
}

/** Line style of series index `k`: `linestyles[k % n]`. */
export function lineStyleAt(linestyles: readonly string[], k: number): string {
  return linestyles.length === 0 ? "-" : (linestyles[k % linestyles.length] as string);
}

/** Numeric column of a table scaled by `factor`; text and missing entries become null. */
export function scaled(table: Table, column: string, factor: number): Num[] | null {
  const col = table[column];
  if (!col) return null;
  return col.map((v) => (typeof v === "number" && Number.isFinite(v) ? v * factor : null));
}

const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const nulls = (v: readonly Num[]) => v.filter((q) => q === null).length;

interface SeriesSplines {
  /** y against the study x (the base x of the figure). */
  y: Spline;
  /** Panel x against the study x, only for a panel with its own x axis. */
  x: Spline | null;
  rawY: Num[];
  /** Study x and panel x of the real points. */
  rawT: Num[];
  rawX: Num[];
}

function splinesOf(rev: RevisionData, fig: FigureSpec, panel: PanelSpec, s: SeriesSpec): SeriesSplines | null {
  const xAxis = panel.x ?? fig.x;
  const own = xAxis.column !== fig.x.column;
  const t = scaled(rev.table, fig.x.column, fig.x.factor);
  const y = scaled(rev.table, s.column, s.factor);
  const px = own ? scaled(rev.table, xAxis.column, xAxis.factor) : t;
  if (!t || !y || !px) return null;
  return { y: buildSpline(t, y), x: own ? buildSpline(t, px) : null, rawY: y, rawT: t, rawX: px };
}

/** Panel x at study x `t` (identity for a panel on the study x). */
const panelX = (sp: SeriesSplines, t: number): Num => (sp.x ? sp.x.evaluate(t) : t);

const sub = (a: Num, b: Num): Num => (a === null || b === null ? null : a - b);

/**
 * Traces of one panel: one line per (revision x series); colour = revision, dash = series index.
 *
 * Every line is the dense resample of a natural cubic spline over the study x, the same
 * spline the readout evaluates. A panel with its own x axis is drawn parametrically: both its
 * x and its y are splines over the study x. In delta mode each revision is
 * (revision - reference), the revision spline evaluated on the reference's grid and drawn at
 * the reference's panel x; the reference itself is not drawn. In overlay mode the reference
 * gets a ring at study x = 0 (the static pose). `revs` are the visible revisions in open
 * order; `ref` may be one of them or a hidden reference that is only used in delta mode.
 */
export function buildPanel(
  fig: FigureSpec,
  panel: PanelSpec,
  revs: readonly RevisionData[],
  ref: RevisionData | null,
  mode: TraceMode,
  linestyles: readonly string[],
  mutedColor: string,
  opts: BuildOptions = {},
): PanelPlot {
  const xAxis = panel.x ?? fig.x;
  const delta = mode === "delta";
  const drawnSeries = new Set<number>();
  const traces: PlotlyObject[] = [];
  const points: PlotlyObject[] = [];
  const curves: PanelCurve[] = [];
  let icDropped = 0;
  const shown = delta ? revs.filter((r) => r.id !== ref?.id) : revs;

  const drawn = panel.series
    .map((s, k) => ({ s, k }))
    .filter(({ s }) => [...shown, ...(delta && ref ? [ref] : [])].some((r) => s.column in r.table));
  const legend = drawn.length > 1;

  for (const rev of shown) {
    const cols = new Set(Object.keys(rev.table));
    for (const { series: s, index: k } of seriesIn(panel, cols)) {
      const sp = splinesOf(rev, fig, panel, s);
      if (!sp) continue;
      let xs: Num[];
      let ys: Num[];
      let at: (t: number) => number | null;
      let px: Num[];
      let py: Num[];
      if (delta) {
        if (!ref) continue;
        const rs = splinesOf(ref, fig, panel, s);
        if (!rs) continue;
        const grid = rs.y.resample(drawPoints(rs.rawT.length));
        ys = grid.y.map((rv, i) => sub(sp.y.evaluate(grid.x[i] as number), rv));
        xs = grid.x.map((t, i) => (grid.y[i] === null ? t : panelX(rs, t)));
        at = (t) => sub(sp.y.evaluate(t), rs.y.evaluate(t));
        icDropped = Math.max(icDropped, nulls(sp.rawY), nulls(rs.rawY));
        px = rs.rawX;
        py = rs.rawT.map((t, i) => sub(t === null ? null : sp.y.evaluate(t), rs.rawY[i] ?? null));
      } else {
        const grid = sp.y.resample(drawPoints(sp.rawT.length));
        ys = grid.y;
        xs = grid.x.map((t, i) => (grid.y[i] === null ? t : panelX(sp, t)));
        at = (t) => sp.y.evaluate(t);
        if (panel.ic) icDropped = Math.max(icDropped, nulls(sp.rawY));
        px = sp.rawX;
        py = sp.rawY;
      }
      drawnSeries.add(k);
      curves.push({
        revId: rev.id,
        letter: rev.letter,
        color: rev.color,
        seriesIndex: k,
        label: s.label,
        side: s.side,
        unit: s.unit,
        at,
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        name: rev.letter,
        connectgaps: false,
        showlegend: false,
        hoverinfo: "none",
        line: { color: rev.color, width: LINE_WIDTH, dash: plotlyDash(lineStyleAt(linestyles, k)) },
      });
      if (opts.showPoints) {
        const keep = py.map((v, i) => v !== null && px[i] !== null);
        points.push({
          type: "scatter",
          mode: "markers",
          x: px.filter((_, i) => keep[i]),
          y: py.filter((_, i) => keep[i]),
          name: rev.letter,
          hoverinfo: "none",
          showlegend: false,
          marker: { symbol: "circle", size: 3, color: rev.color },
        });
      }
    }
  }
  traces.push(...points);

  // static-pose ring (study x = 0) on the reference revision only
  if (!delta && ref && revs.some((r) => r.id === ref.id)) {
    for (const { series: s } of seriesIn(panel, new Set(Object.keys(ref.table)))) {
      const rs = splinesOf(ref, fig, panel, s);
      const y0 = rs ? rs.y.evaluate(0) : null;
      const x0 = rs ? panelX(rs, 0) : null;
      if (y0 === null || x0 === null) continue;
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [x0],
        y: [y0],
        hoverinfo: "none",
        showlegend: false,
        marker: { symbol: "circle-open", size: 8, color: ref.color, line: { color: ref.color, width: 1.5 } },
      });
    }
  }

  if (legend) {
    for (const { s, k } of drawn) {
      if (!drawnSeries.has(k)) continue;
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [null],
        y: [null],
        name: s.label,
        hoverinfo: "skip",
        showlegend: true,
        line: { color: mutedColor, width: LINE_WIDTH, dash: plotlyDash(lineStyleAt(linestyles, k)) },
      });
    }
  }

  const first = panel.series[0];
  const baseY = panel.y_label || (first?.unit ? `[${first.unit}]` : "");
  return {
    traces,
    legend,
    curves,
    xTitle: sentence(withUnit(xAxis.label, xAxis.unit)),
    yTitle: delta ? `Δ ${baseY}`.trim() : baseY,
    icDropped,
  };
}
