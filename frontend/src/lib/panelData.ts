import type { FigureSpec, PanelSpec } from "../api/spec";
import { buildSpline } from "./interp";
import { seriesIn } from "./panels";
import { withUnit } from "./format";
import { scaled, type RevisionData } from "./traces";

type Num = number | null;

/** Header cells then data rows; an empty string is a missing value. */
export type DataRows = string[][];

/** x values closer than this (relative to 1) are the same solver point. */
const X_TOL = 1e-9;

/** Text of a number: at most 10 significant digits, so float noise from unit factors disappears. */
export function numText(v: number): string {
  return String(Number(v.toPrecision(10)));
}

const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);

interface Column {
  header: string;
  /** Value at each row (study x), "" when the revision has none. */
  at: (x: number) => string;
}

/** Look up a value at an exact solved x (within tolerance). */
function exact(t: readonly Num[], y: readonly Num[]): (x: number) => number | null {
  return (x) => {
    for (let i = 0; i < t.length; i++) {
      const ti = t[i];
      if (ti !== null && ti !== undefined && Math.abs(ti - x) <= X_TOL * Math.max(1, Math.abs(x))) {
        return y[i] ?? null;
      }
    }
    return null;
  };
}

const cell = (v: number | null) => (v === null ? "" : numText(v));

/**
 * The solved points behind one panel, in display units, as a table of strings.
 *
 * Overlay: rows are the union of the study x of all visible revisions, columns are the study x
 * ("Wheel travel [mm]") and then one per revision and series ("A camber [deg]"); a revision with
 * no point at a row's x leaves the cell empty. A panel with its own x axis also gets an
 * "A <x name> [unit]" column per revision.
 *
 * Delta: rows are the reference's solved x; columns are "B-A camber [deg]" for every revision
 * but the reference, the revision value taken at the reference x (its own point when it has
 * one, else its spline, like the plotted Δ). Nothing is drawn from the dense spline.
 */
export function panelRows(
  fig: FigureSpec,
  panel: PanelSpec,
  revs: readonly RevisionData[],
  ref: RevisionData | null,
  delta: boolean,
): DataRows {
  const xAxis = fig.x;
  const own = (panel.x ?? fig.x).column !== fig.x.column;
  const ownAxis = panel.x ?? fig.x;
  const shown = delta ? revs.filter((r) => r.id !== ref?.id) : revs;
  const columns: Column[] = [];
  const xs: number[] = [];
  const addXs = (r: RevisionData) => {
    for (const v of scaled(r.table, xAxis.column, xAxis.factor) ?? []) if (v !== null) xs.push(v);
  };

  if (delta) {
    if (!ref) return [];
    addXs(ref);
  } else {
    revs.forEach(addXs);
  }

  for (const rev of shown) {
    const t = scaled(rev.table, xAxis.column, xAxis.factor);
    if (!t) continue;
    const cols = new Set(Object.keys(rev.table));
    if (own && !delta) {
      const px = scaled(rev.table, ownAxis.column, ownAxis.factor);
      if (px) {
        columns.push({
          header: `${rev.letter} ${lower(withUnit(ownAxis.label, ownAxis.unit))}`,
          at: (x) => cell(exact(t, px)(x)),
        });
      }
    }
    for (const { series } of seriesIn(panel, cols)) {
      const y = scaled(rev.table, series.column, series.factor);
      if (!y) continue;
      const header = `${delta ? `${rev.letter}-${ref?.letter ?? "?"}` : rev.letter} ${lower(withUnit(series.label, series.unit))}`;
      if (!delta) {
        const get = exact(t, y);
        columns.push({ header, at: (x) => cell(get(x)) });
        continue;
      }
      const rt = ref ? scaled(ref.table, xAxis.column, xAxis.factor) : null;
      const ry = ref ? scaled(ref.table, series.column, series.factor) : null;
      if (!rt || !ry) continue;
      const mine = exact(t, y);
      const spline = buildSpline(t, y);
      const refAt = exact(rt, ry);
      columns.push({
        header,
        at: (x) => {
          const r = refAt(x);
          if (r === null) return "";
          const v = mine(x) ?? spline.evaluate(x);
          return v === null ? "" : numText(v - r);
        },
      });
    }
  }

  const sorted = [...new Set(xs)].sort((a, b) => a - b);
  const rows: number[] = [];
  for (const x of sorted) {
    const last = rows[rows.length - 1];
    if (last === undefined || Math.abs(x - last) > X_TOL * Math.max(1, Math.abs(x))) rows.push(x);
  }
  const head = [`${sentence(withUnit(xAxis.label, xAxis.unit))}`, ...columns.map((c) => c.header)];
  return [head, ...rows.map((x) => [numText(x), ...columns.map((c) => c.at(x))])];
}

const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** RFC 4180 quoting of one field. */
function quote(f: string): string {
  return /[",\n\r]/.test(f) ? `"${f.replace(/"/g, '""')}"` : f;
}

/** CSV text, comma separated, LF line ends. */
export function toCsv(rows: DataRows): string {
  return rows.map((r) => r.map(quote).join(",")).join("\n") + "\n";
}

/** Tab-separated text for pasting into a spreadsheet. */
export function toTsv(rows: DataRows): string {
  return rows.map((r) => r.map((f) => f.replace(/[\t\r\n]+/g, " ")).join("\t")).join("\n") + "\n";
}

/** File name of a panel's CSV. */
export const csvName = (fig: Pick<FigureSpec, "id">, panel: Pick<PanelSpec, "id">) => `${fig.id}_${panel.id}.csv`;
