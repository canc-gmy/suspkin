import type { AxisSpec, PanelSpec, SeriesSpec } from "../api/spec";
import { fixed, MINUS, MISSING, withUnit } from "./format";
import type { PanelCurve, RevisionData } from "./traces";

/**
 * Decimals of a readout value, from the unit of the series.
 *
 * The spec carries no per-panel decimals, so this is the one place that decides. The cursor is
 * finer than the solver, so each unit gets one decimal more than the summary table shows:
 * angles 3 (deg), lengths 2 (mm), percentages 2 (%). Any other unit (ratios, gains) gets 3.
 */
export function readoutDecimals(unit: string): number {
  switch (unit) {
    case "deg":
      return 3;
    case "mm":
    case "%":
      return 2;
    default:
      return 3;
  }
}

/** Fine nudge step of a cursor on `axis`, in display units: 0.1 for lengths, 0.01 for angles. */
export function nudgeStep(axis: Pick<AxisSpec, "dim" | "unit">): number {
  return axis.dim === "angle" || axis.unit === "deg" ? 0.01 : 0.1;
}

/** Text of an x value in display units: one decimal finer than the nudge step. */
export function formatX(x: number, axis: Pick<AxisSpec, "dim" | "unit">): string {
  return fixed(x, nudgeStep(axis) < 0.05 ? 3 : 2);
}

const SIDE: Record<NonNullable<SeriesSpec["side"]>, string> = { left: "L", right: "R", front: "F", rear: "R" };

/** Signed number text for Δ cells: "+0.12", "−0.12", "0.00". */
export function signed(value: number | null, decimals: number): string {
  if (value === null || !Number.isFinite(value)) return MISSING;
  const s = fixed(value, decimals);
  return s.startsWith(MINUS) || /^0(\.0*)?$/.test(s) ? s : `+${s}`;
}

export interface ReadoutPart {
  /** "L", "R", "F" ... in muted ink; null when a cell holds a single unlabelled value. */
  prefix: string | null;
  text: string;
}

export interface ReadoutCell {
  parts: ReadoutPart[];
  /** Plain text of the cell, for screen readers. */
  spoken: string;
}

export interface ReadoutColumn {
  panelId: string;
  title: string;
}

export interface ReadoutRow {
  revId: string;
  letter: string;
  color: string;
  cells: ReadoutCell[];
}

export interface ReadoutTable {
  columns: ReadoutColumn[];
  rows: ReadoutRow[];
}

export interface ReadoutPanel {
  panel: PanelSpec;
  curves: readonly PanelCurve[];
}

/** Column heading "Camber [deg]" (with "Δ " in delta mode). */
export function columnTitle(panel: PanelSpec, delta: boolean): string {
  const first = panel.series[0];
  const base = panel.title.includes("[") ? panel.title : withUnit(panel.title, first?.unit ?? "");
  return delta ? `Δ ${base}` : base;
}

/**
 * The readout table at study x `x` (null = no cursor, cells are empty).
 *
 * One row per visible revision (in `revs` order), one column per drawn panel. A cell holds
 * every series of the panel that the revision has; with more than one, each value is prefixed
 * (L/R, F/R, or the series label when it has no side). In delta mode the reference row reads
 * "ref" and the others show the signed difference. "-" marks a value outside a curve's range.
 */
export function buildReadout(
  panels: readonly ReadoutPanel[],
  revs: readonly Pick<RevisionData, "id" | "letter" | "color">[],
  refId: string | null,
  delta: boolean,
  x: number | null,
): ReadoutTable {
  const columns = panels.map(({ panel }) => ({ panelId: panel.id, title: columnTitle(panel, delta) }));
  const rows = revs.map((rev) => ({
    revId: rev.id,
    letter: rev.letter,
    color: rev.color,
    cells: panels.map(({ curves }): ReadoutCell => {
      if (delta && rev.id === refId) return { parts: [{ prefix: null, text: "ref" }], spoken: "reference" };
      const mine = curves.filter((c) => c.revId === rev.id).sort((a, b) => a.seriesIndex - b.seriesIndex);
      if (mine.length === 0) return { parts: [{ prefix: null, text: MISSING }], spoken: "no data" };
      const multi = mine.length > 1;
      const parts = mine.map((c): ReadoutPart => {
        const d = readoutDecimals(c.unit);
        const v = x === null ? null : c.at(x);
        const text = x === null ? "" : delta ? signed(v, d) : fixed(v, d);
        return { prefix: multi ? ((c.side && SIDE[c.side]) ?? c.label) : null, text };
      });
      return { parts, spoken: parts.map((p) => `${p.prefix ? `${p.prefix} ` : ""}${p.text}`).join(", ") };
    }),
  }));
  return { columns, rows };
}

/** The slice of a Plotly axis the pointer conversion needs. */
export interface PixelAxis {
  _offset: number;
  _length: number;
  p2d(pixel: number): number;
}

/**
 * Data x under the pointer, or null when it is outside the plot area.
 *
 * `rect` is the chart element's bounding box; `xa` / `ya` are Plotly's full-layout axes.
 */
export function pointerToX(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number },
  xa: PixelAxis,
  ya: Pick<PixelAxis, "_offset" | "_length">,
): number | null {
  const px = clientX - rect.left - xa._offset;
  const py = clientY - rect.top - ya._offset;
  if (px < 0 || px > xa._length || py < 0 || py > ya._length) return null;
  const v = xa.p2d(px);
  return Number.isFinite(v) ? v : null;
}

/** Parse a typed x in display units (accepts a decimal comma and a Unicode minus). */
export function parseX(text: string): number | null {
  const t = text.trim().replace(",", ".").replace(MINUS, "-");
  if (!/^[+-]?(\d+\.?\d*|\.\d+)$/.test(t)) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Clamp `x` to a domain (no-op for a null domain). */
export function clampX(x: number, domain: readonly [number, number] | null): number {
  return domain ? Math.min(domain[1], Math.max(domain[0], x)) : x;
}

/** Pinned x after one arrow press: `dir` is -1 or +1, Shift moves 10x. Rounded to kill float drift. */
export function nudge(x: number, dir: -1 | 1, shift: boolean, axis: Pick<AxisSpec, "dim" | "unit">): number {
  const step = nudgeStep(axis) * (shift ? 10 : 1);
  return Math.round((x + dir * step) * 1e6) / 1e6;
}
