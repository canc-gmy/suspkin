import type { HistoryData } from "../api/types";
import type { PlotlyObject } from "../theme/plotlyTemplate";
import { LINE_WIDTH } from "../theme/plotlyTemplate";
import type { Num } from "./delta";

/** Margins beyond this are drawn clipped (as in the report figure); hover shows the true value. */
export const G_CLIP = 50;

/** Colours the optimization charts use (resolved tokens). */
export interface OptColors {
  accent: string;
  ink: string;
  muted: string;
  danger: string;
  baseline: string;
}

/** History reshaped for plotting: rows sorted by evaluation id. All analysis comes from the API. */
export interface HistoryView {
  evalId: number[];
  fTotal: Num[];
  feasible: boolean[];
  best: Num[];
  /** Non-dominated in all objectives. */
  paretoAll: boolean[];
  /** Non-dominated in one objective pair only, keyed "f_a|f_b". */
  paretoPairs: Record<string, boolean[]>;
  knee: number | null;
  objectives: string[];
  variables: string[];
  columns: Record<string, Num[]>;
}

const num = (v: unknown): Num => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Reshape the history response: numeric columns, rows ordered by `eval_id`. */
export function mapHistory(h: HistoryData): HistoryView {
  const objectives = h.objectives ?? [];
  const ids = (h.columns["eval_id"] ?? []).map((v) => num(v) ?? Number.NaN);
  const order = ids.map((_, i) => i).sort((a, b) => (ids[a] as number) - (ids[b] as number));
  const pick = <T,>(xs: readonly T[] | undefined, fallback: T): T[] => order.map((i) => xs?.[i] ?? fallback);
  const columns: Record<string, Num[]> = {};
  for (const [name, col] of Object.entries(h.columns)) {
    if (name.startsWith("x_") || name === "f_total" || objectives.includes(name)) {
      columns[name] = pick(col, null).map(num);
    }
  }
  return {
    evalId: order.map((i) => ids[i] as number),
    fTotal: pick(h.columns["f_total"], null).map(num),
    feasible: pick(h.columns["feasible"], false).map((v) => v === true || v === "True"),
    best: pick(h.best_so_far, null),
    paretoAll: pick(h.pareto_all, false),
    paretoPairs: Object.fromEntries(Object.entries(h.pareto_pairs ?? {}).map(([k, m]) => [k, pick(m, false)])),
    knee: h.knee_eval_id ?? null,
    objectives: [...objectives],
    variables: Object.keys(h.columns).filter((c) => c.startsWith("x_")),
    columns,
  };
}

/** Log axis when every value is positive and they span more than two decades. */
export function isLogScale(values: readonly Num[]): boolean {
  const present = values.filter((x): x is number => x !== null);
  if (present.length === 0 || present.some((x) => x <= 0)) return false;
  return Math.max(...present) / Math.min(...present) > 100;
}

const pickRows = <T,>(xs: readonly T[], keep: readonly boolean[]): T[] => xs.filter((_, i) => keep[i]);

/** Convergence: all evaluations (infeasible marked as crosses) and the best feasible so far. */
export function convergenceTraces(v: HistoryView, c: OptColors): PlotlyObject[] {
  const ok = v.fTotal.map((f) => f !== null);
  const feas = v.feasible.map((f, i) => f && ok[i] === true);
  const infeas = v.feasible.map((f, i) => !f && ok[i] === true);
  const hasBest = v.best.map((b) => b !== null);
  return [
    {
      type: "scatter",
      mode: "markers",
      name: "infeasible",
      x: pickRows(v.evalId, infeas),
      y: pickRows(v.fTotal, infeas),
      marker: { symbol: "x", size: 6, color: c.muted, opacity: 0.55 },
      hovertemplate: "eval %{x}  f_total %{y:.4g} (infeasible)<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "feasible",
      x: pickRows(v.evalId, feas),
      y: pickRows(v.fTotal, feas),
      marker: { symbol: "circle", size: 5, color: c.accent, opacity: 0.3 },
      hovertemplate: "eval %{x}  f_total %{y:.4g}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "best feasible",
      x: pickRows(v.evalId, hasBest),
      y: pickRows(v.best, hasBest),
      line: { shape: "hv", color: c.accent, width: LINE_WIDTH },
      hovertemplate: "best so far %{y:.4g}<extra></extra>",
    },
  ];
}

/** Pareto view of two objectives: front highlighted, knee marked, infeasible points crossed. */
export function paretoTraces(v: HistoryView, xName: string, yName: string, c: OptColors): PlotlyObject[] {
  const xs = v.columns[xName] ?? [];
  const ys = v.columns[yName] ?? [];
  const finite = xs.map((x, i) => x !== null && ys[i] != null);
  const infeas = v.feasible.map((f, i) => !f && finite[i] === true);
  // line: the 2D front of this pair only (the all-objective set is marked separately)
  const pair = v.paretoPairs[`${xName}|${yName}`] ?? v.paretoPairs[`${yName}|${xName}`] ?? [];
  const allNd = v.paretoAll.map((p, i) => p && finite[i] === true);
  const front = v.feasible.map((f, i) => f && pair[i] === true && finite[i] === true);
  const dom = v.feasible.map((f, i) => f && !front[i] && finite[i] === true);
  const frontIdx = front
    .map((f, i) => (f ? i : -1))
    .filter((i) => i >= 0)
    .sort((a, b) => (xs[a] as number) - (xs[b] as number));
  const hover = "eval %{customdata}  %{x:.4g}, %{y:.4g}<extra></extra>";
  const traces: PlotlyObject[] = [
    {
      type: "scatter",
      mode: "markers",
      name: "infeasible",
      x: pickRows(xs, infeas),
      y: pickRows(ys, infeas),
      customdata: pickRows(v.evalId, infeas),
      marker: { symbol: "x", size: 6, color: c.muted, opacity: 0.5 },
      hovertemplate: hover,
    },
    {
      type: "scatter",
      mode: "markers",
      name: "feasible",
      x: pickRows(xs, dom),
      y: pickRows(ys, dom),
      customdata: pickRows(v.evalId, dom),
      marker: { symbol: "circle", size: 6, color: c.accent, opacity: 0.3 },
      hovertemplate: hover,
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "front (this pair)",
      x: frontIdx.map((i) => xs[i]),
      y: frontIdx.map((i) => ys[i]),
      customdata: frontIdx.map((i) => v.evalId[i]),
      line: { color: c.accent, width: LINE_WIDTH },
      marker: { symbol: "circle", size: 7, color: c.accent },
      hovertemplate: hover,
    },
    {
      type: "scatter",
      mode: "markers",
      name: "non-dominated (all objectives)",
      x: pickRows(xs, allNd),
      y: pickRows(ys, allNd),
      customdata: pickRows(v.evalId, allNd),
      marker: { symbol: "circle-open", size: 11, color: c.ink, opacity: 0.55, line: { width: 1 } },
      hovertemplate: hover,
    },
  ];
  const k = v.knee === null ? -1 : v.evalId.indexOf(v.knee);
  if (k >= 0 && xs[k] != null && ys[k] != null) {
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "knee",
      x: [xs[k]],
      y: [ys[k]],
      customdata: [v.knee],
      marker: { symbol: "star", size: 15, color: c.ink, line: { color: c.accent, width: 1.5 } },
      hovertemplate: "knee, eval %{customdata}  %{x:.4g}, %{y:.4g}<extra></extra>",
    });
  }
  return traces;
}

/** Parallel coordinates of the design variables of the feasible evaluations, coloured by f_total. */
export function parcoordsTrace(v: HistoryView, c: OptColors): PlotlyObject | null {
  const rows = v.feasible.map((f, i) => f && v.fTotal[i] !== null);
  if (!rows.includes(true) || v.variables.length === 0) return null;
  const color = pickRows(v.fTotal, rows);
  const font = { color: c.ink, size: 12 };
  return {
    type: "parcoords",
    line: {
      color,
      colorscale: "Viridis",
      reversescale: true,
      showscale: true,
      colorbar: { title: { text: "f_total", font }, tickfont: font, thickness: 12 },
    },
    labelfont: font,
    labelangle: -35,
    tickfont: { color: c.muted, size: 11 },
    rangefont: { color: c.muted, size: 11 },
    dimensions: v.variables.map((name) => ({
      label: name.slice(2),
      range: [0, 1],
      values: pickRows(v.columns[name] ?? [], rows),
    })),
  };
}

export interface MarginRow {
  name: string;
  baseline: Num;
  best: Num;
}

/** Constraint margins g of the baseline and the best design, from `result.json` (g > 0 violates). */
export function marginRows(result: Record<string, unknown>): MarginRow[] {
  const g = (key: string): Record<string, unknown> => {
    const d = (result[key] as { g?: unknown } | undefined)?.g;
    return d && typeof d === "object" ? (d as Record<string, unknown>) : {};
  };
  const base = g("baseline");
  const best = g("best");
  const names = [...new Set([...Object.keys(base), ...Object.keys(best)])];
  return names.map((name) => ({ name, baseline: num(base[name]), best: num(best[name]) }));
}

const clip = (g: Num): Num => (g === null ? null : Math.max(-G_CLIP, Math.min(G_CLIP, g)));

/** Horizontal bars of g per constraint, baseline vs best, clipped at +-G_CLIP. */
export function marginTraces(rows: readonly MarginRow[], c: OptColors): PlotlyObject[] {
  const names = rows.map((r) => r.name);
  const bar = (name: string, pick: (r: MarginRow) => Num, color: string): PlotlyObject => ({
    type: "bar",
    orientation: "h",
    name,
    y: names,
    x: rows.map((r) => clip(pick(r))),
    customdata: rows.map((r) => pick(r)),
    marker: { color },
    hovertemplate: `${name}  g = %{customdata:.4g}<extra></extra>`,
  });
  return [bar("baseline", (r) => r.baseline, c.baseline), bar("best", (r) => r.best, c.accent)];
}
