import type { PlotlyObject } from "../theme/plotlyTemplate";
import { LINE_WIDTH } from "../theme/plotlyTemplate";
import type { ConstraintRow, EvolutionBand, Histogram, ObjectiveRow, ParamRow } from "./historyStats";
import type { OptColors } from "./optimization";

/** `#RRGGBB` with an alpha, for translucent fills; other colour strings are returned unchanged. */
export function withAlpha(color: string, alpha: number): string {
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return color;
  const n = parseInt(m[1] as string, 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** Histogram of f_total: feasible and infeasible bars on shared bins, best, median and mean as vertical lines. */
export function histogramTraces(h: Histogram, c: OptColors): PlotlyObject[] {
  const w = (h.edges[1] as number) - (h.edges[0] as number);
  const centres = h.feasible.map((_, k) => (h.edges[k] as number) + w / 2);
  const top = Math.max(1, ...h.feasible.map((v, k) => v + (h.infeasible[k] as number)));
  const bars = (name: string, y: number[], color: string): PlotlyObject => ({
    type: "bar",
    name,
    x: centres,
    y,
    width: w * 0.95,
    marker: { color, opacity: 0.75 },
    hovertemplate: `${name}  %{y} samples near %{x:.4g}<extra></extra>`,
  });
  const line = (name: string, x: number | null, dash: string): PlotlyObject[] =>
    x === null
      ? []
      : [
          {
            type: "scatter",
            mode: "lines",
            name: `${name} ${x.toPrecision(4)}`,
            x: [x, x],
            y: [0, top * 1.05],
            line: { color: c.ink, width: 1.5, dash },
            hoverinfo: "skip",
          },
        ];
  return [
    bars("feasible", h.feasible, c.accent),
    bars("infeasible", h.infeasible, c.muted),
    ...line("best", h.best, "solid"),
    ...line("median", h.median, "dash"),
    ...line("mean", h.mean, "dot"),
  ];
}

/** One box per objective (feasible samples). */
export function boxTraces(rows: readonly ObjectiveRow[], c: OptColors): PlotlyObject[] {
  return rows.map((r) => ({
    type: "box",
    name: r.name.replace(/^f_/, ""),
    y: r.values,
    boxpoints: false,
    marker: { color: c.accent },
    line: { color: c.accent, width: 1.5 },
    fillcolor: withAlpha(c.accent, 0.15),
    hovertemplate: `${r.name}  %{y:.4g}<extra></extra>`,
  }));
}

const unit = (v: number | null, p: ParamRow): number | null => (v === null ? null : (v - (p.lo as number)) / ((p.hi as number) - (p.lo as number)));

/** Dot-and-whisker per parameter, normalised to its explored range: all feasible, top 10 %, best. y is row index. */
export function paramTraces(rows: readonly ParamRow[], c: OptColors): PlotlyObject[] {
  const span = (p: ParamRow) => (p.hi as number) - (p.lo as number);
  const group = (name: string, color: string, pick: (p: ParamRow) => { m: number | null; s: number | null } | null, dy: number): PlotlyObject => {
    const pts = rows.map((p, i) => ({ p, i, g: pick(p) })).filter((r) => r.g !== null && r.g.m !== null);
    return {
      type: "scatter",
      mode: "markers",
      name,
      x: pts.map((r) => unit((r.g as { m: number }).m, r.p)),
      y: pts.map((r) => r.i + dy),
      error_x: { type: "data", array: pts.map((r) => ((r.g?.s ?? 0) as number) / span(r.p)), color, thickness: 1.5, width: 4 },
      marker: { symbol: "circle", size: 7, color },
      hoverinfo: "skip",
    };
  };
  const best: PlotlyObject = {
    type: "scatter",
    mode: "markers",
    name: "best design",
    x: rows.map((p) => unit(p.best, p)),
    y: rows.map((_, i) => i),
    marker: { symbol: "diamond", size: 9, color: c.ink },
    hoverinfo: "skip",
  };
  return [
    group("all feasible", c.muted, (p) => ({ m: p.all.mean, s: p.all.std }), -0.18),
    group("top 10 %", c.accent, (p) => (p.top ? { m: p.top.mean, s: p.top.std } : null), 0.18),
    best,
  ];
}

/** Horizontal bars of the pass rate per constraint (all ok samples). */
export function passRateTraces(rows: readonly ConstraintRow[], c: OptColors): PlotlyObject {
  return {
    type: "bar",
    orientation: "h",
    name: "pass rate",
    y: rows.map((r) => r.name.replace(/^g_/, "")),
    x: rows.map((r) => r.all.passRate),
    marker: { color: c.accent },
    hovertemplate: "%{y}  %{x:.1f} % pass<extra></extra>",
  };
}

/** Mean +- std band of f_total per window, with the best-so-far step line on top. */
export function evolutionTraces(
  bands: readonly EvolutionBand[],
  evalId: readonly number[],
  best: readonly (number | null)[],
  c: OptColors,
): PlotlyObject[] {
  const xs = bands.map((b) => b.x);
  const hi = bands.map((b) => b.mean + (b.std ?? 0));
  const lo = bands.map((b) => b.mean - (b.std ?? 0));
  const keep = best.map((b) => b !== null);
  return [
    { type: "scatter", mode: "lines", name: "upper", x: xs, y: hi, line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    {
      type: "scatter",
      mode: "lines",
      name: "mean ± std",
      x: xs,
      y: lo,
      fill: "tonexty",
      fillcolor: withAlpha(c.accent, 0.18),
      line: { width: 0 },
      hoverinfo: "skip",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "window mean",
      x: xs,
      y: bands.map((b) => b.mean),
      line: { color: c.accent, width: LINE_WIDTH },
      marker: { size: 4, color: c.accent },
      hovertemplate: "evaluation %{x:.0f}  mean %{y:.4g}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "best so far",
      x: evalId.filter((_, i) => keep[i]),
      y: best.filter((b) => b !== null),
      line: { shape: "hv", color: c.ink, width: LINE_WIDTH },
      hovertemplate: "best so far %{y:.4g}<extra></extra>",
    },
  ];
}
