import type { FigureSpec, PanelSpec } from "../api/spec";
import { LINE_WIDTH, type PlotlyObject } from "../theme/plotlyTemplate";
import { deltaToReference, interpolate, type Num } from "./delta";
import { seriesIn } from "./panels";
import { withUnit } from "./format";

export type Table = Readonly<Record<string, readonly (number | string | null)[]>>;

/** One revision as the chart builder sees it. */
export interface RevisionData {
  id: string;
  letter: string;
  color: string;
  table: Table;
}

export type TraceMode = "overlay" | "delta";

export interface PanelPlot {
  traces: PlotlyObject[];
  /** Panel legend (series labels) shown: more than one drawn series. */
  legend: boolean;
  xTitle: string;
  yTitle: string;
  /** Unified-hover heading parts: "<label> 12 mm". */
  xHover: { label: string; unit: string };
  /** Points not drawn because the curve is undefined there (instant centre at infinity). */
  icDropped: number;
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

/**
 * Traces of one panel: one line per (revision x series); colour = revision, dash = series index.
 *
 * In delta mode each revision is shown as (revision - reference) on the reference x grid and
 * the reference itself is not drawn. In overlay mode the reference gets a ring at x = 0 (the
 * static pose). `revs` are the visible revisions in open order; `ref` may be one of them or a
 * hidden reference that is only used in delta mode.
 */
export function buildPanel(
  fig: FigureSpec,
  panel: PanelSpec,
  revs: readonly RevisionData[],
  ref: RevisionData | null,
  mode: TraceMode,
  linestyles: readonly string[],
  mutedColor: string,
): PanelPlot {
  const xAxis = panel.x ?? fig.x;
  const delta = mode === "delta";
  const drawnSeries = new Set<number>();
  const traces: PlotlyObject[] = [];
  let icDropped = 0;
  const shown = delta ? revs.filter((r) => r.id !== ref?.id) : revs;

  const drawn = panel.series
    .map((s, k) => ({ s, k }))
    .filter(({ s }) => [...shown, ...(delta && ref ? [ref] : [])].some((r) => s.column in r.table));
  const legend = drawn.length > 1;

  for (const rev of shown) {
    const cols = new Set(Object.keys(rev.table));
    const x = scaled(rev.table, xAxis.column, xAxis.factor);
    if (!x) continue;
    for (const { series: s, index: k } of seriesIn(panel, cols)) {
      const y = scaled(rev.table, s.column, s.factor);
      if (!y) continue;
      let xs: Num[] = x;
      let ys: Num[] = y;
      if (delta) {
        if (!ref) continue;
        // difference on the study's base x (monotonic), drawn at the reference's panel x
        const rx = scaled(ref.table, xAxis.column, xAxis.factor);
        const ry = scaled(ref.table, s.column, s.factor);
        const rBase = scaled(ref.table, fig.x.column, fig.x.factor);
        const base = scaled(rev.table, fig.x.column, fig.x.factor);
        if (!rx || !ry || !rBase || !base) continue;
        xs = rx;
        ys = deltaToReference(rBase, ry, base, y);
      }
      drawnSeries.add(k);
      if (panel.ic) icDropped = Math.max(icDropped, ys.filter((v) => v === null).length);
      const unit = s.unit ? ` ${s.unit}` : "";
      const label = legend ? `${s.label}  ` : "";
      traces.push({
        type: "scatter",
        mode: "lines",
        xaxis: "x2",
        x: xs,
        y: ys,
        name: rev.letter,
        connectgaps: false,
        showlegend: false,
        line: { color: rev.color, width: LINE_WIDTH, dash: plotlyDash(lineStyleAt(linestyles, k)) },
        hovertemplate: `<span style="color:${rev.color}"><b>${rev.letter}</b></span>  ${label}%{y:.4~g}${unit}<extra></extra>`,
      });
    }
  }

  // static-pose ring (x = 0) on the reference revision only
  if (!delta && ref && revs.some((r) => r.id === ref.id)) {
    const rx = scaled(ref.table, xAxis.column, xAxis.factor);
    if (rx) {
      for (const { series: s } of seriesIn(panel, new Set(Object.keys(ref.table)))) {
        const ry = scaled(ref.table, s.column, s.factor);
        const y0 = ry ? interpolate(rx, ry, [0])[0] : null;
        if (y0 == null) continue;
        traces.push({
          type: "scatter",
          mode: "markers",
          xaxis: "x2",
          x: [0],
          y: [y0],
          hoverinfo: "skip",
          showlegend: false,
          marker: { symbol: "circle-open", size: 8, color: ref.color, line: { color: ref.color, width: 1.5 } },
        });
      }
    }
  }

  if (legend) {
    for (const { s, k } of drawn) {
      if (!drawnSeries.has(k)) continue;
      traces.push({
        type: "scatter",
        mode: "lines",
        xaxis: "x2",
        x: [null],
        y: [null],
        name: s.label,
        hoverinfo: "skip",
        showlegend: true,
        line: { color: mutedColor, width: LINE_WIDTH, dash: plotlyDash(lineStyleAt(linestyles, k)) },
      });
    }
  }

  // the visible x axis needs a trace of its own (the data sits on the hover axis x2)
  traces.push({ type: "scatter", mode: "none", x: [null], y: [null], hoverinfo: "skip", showlegend: false });

  const first = panel.series[0];
  const baseY = panel.y_label || (first?.unit ? `[${first.unit}]` : "");
  return {
    traces,
    legend,
    xTitle: sentence(withUnit(xAxis.label, xAxis.unit)),
    xHover: { label: sentence(xAxis.label), unit: xAxis.unit },
    yTitle: delta ? `Δ ${baseY}`.trim() : baseY,
    icDropped,
  };
}
