import type { HistoryData } from "../api/types";

/** Penalty the optimizer stores for a failed evaluation (`F_FAIL`). */
export const F_FAIL = 1e3;
/** Share of the feasible samples that counts as "top". */
export const TOP_FRACTION = 0.1;
/** Fewer feasible samples than this gives no top-group statistics. */
export const TOP_MIN = 5;

type Num = number | null;

/** The history table reshaped for statistics: rows ordered by evaluation id. */
export interface StatsData {
  n: number;
  evalId: number[];
  fTotal: Num[];
  feasible: boolean[];
  status: string[];
  source: string[];
  seconds: Num[];
  objectives: string[];
  /** Objective columns (`f_*`), normalised. */
  f: Record<string, Num[]>;
  /** Constraint columns (`g_*`). */
  g: Record<string, Num[]>;
  /** Design parameters: physical `p_*` in mm when present, else the unit-cube `x_*`. */
  params: Record<string, Num[]>;
  paramUnit: "mm" | "unit cube";
  best: Num[];
  mode: string | null;
  bestEvalId: number | null;
  baselineFTotal: Num;
}

const num = (v: unknown): Num => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Reshape the history response for statistics. `p_*` columns are converted from m to mm. */
export function mapStats(h: HistoryData): StatsData {
  const cols = h.columns;
  const ids = (cols["eval_id"] ?? []).map((v) => num(v) ?? Number.NaN);
  const order = ids.map((_, i) => i).sort((a, b) => (ids[a] as number) - (ids[b] as number));
  const pick = <T,>(xs: readonly T[] | undefined, fallback: T): T[] => order.map((i) => xs?.[i] ?? fallback);
  const numCol = (name: string, scale = 1): Num[] =>
    pick(cols[name], null).map((v) => {
      const x = num(v);
      return x === null ? null : x * scale;
    });
  const names = Object.keys(cols);
  const objectives = (h.objectives ?? []).filter((o) => names.includes(o));
  const f = Object.fromEntries(objectives.map((o) => [o, numCol(o)]));
  const g = Object.fromEntries(names.filter((c) => c.startsWith("g_")).map((c) => [c, numCol(c)]));
  const pNames = names.filter((c) => /^p_.*_[xyz]$/.test(c));
  const usePhysical = pNames.length > 0;
  const params = Object.fromEntries(
    usePhysical ? pNames.map((c) => [c.slice(2), numCol(c, 1000)]) : names.filter((c) => c.startsWith("x_")).map((c) => [c.slice(2), numCol(c)]),
  );
  const res = h.result as { mode?: unknown; best_eval_id?: unknown; baseline?: { f_total?: unknown } } | undefined;
  return {
    n: order.length,
    evalId: order.map((i) => ids[i] as number),
    fTotal: numCol("f_total"),
    feasible: pick(cols["feasible"], false).map((v) => v === true || v === "True"),
    status: pick(cols["status"], "ok").map((v) => String(v)),
    source: pick(cols["source"], "").map((v) => String(v)),
    seconds: numCol("seconds"),
    objectives,
    f,
    g,
    params,
    paramUnit: usePhysical ? "mm" : "unit cube",
    best: pick(h.best_so_far, null),
    mode: typeof res?.mode === "string" ? res.mode : null,
    bestEvalId: num(res?.best_eval_id),
    baselineFTotal: num(res?.baseline?.f_total),
  };
}

// ---- basic statistics ---------------------------------------------------------------

/** Arithmetic mean; null for an empty list. */
export function mean(xs: readonly number[]): Num {
  return xs.length === 0 ? null : xs.reduce((a, b) => a + b, 0) / xs.length;
}

/** Sample standard deviation (divisor n - 1); null for fewer than 2 values. */
export function std(xs: readonly number[]): Num {
  if (xs.length < 2) return null;
  const m = mean(xs) as number;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

/** Quantile `q` in [0, 1] by linear interpolation between order statistics; null for an empty list. */
export function quantile(xs: readonly number[], q: number): Num {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const pos = Math.min(1, Math.max(0, q)) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return (s[lo] as number) + ((s[hi] as number) - (s[lo] as number)) * (pos - lo);
}

export interface Describe {
  n: number;
  mean: Num;
  std: Num;
  min: Num;
  q1: Num;
  median: Num;
  q3: Num;
  max: Num;
}

/** Count, mean, sample std, min, quartiles and max of the values. */
export function describe(xs: readonly number[]): Describe {
  return {
    n: xs.length,
    mean: mean(xs),
    std: std(xs),
    min: xs.length ? Math.min(...xs) : null,
    q1: quantile(xs, 0.25),
    median: quantile(xs, 0.5),
    q3: quantile(xs, 0.75),
    max: xs.length ? Math.max(...xs) : null,
  };
}

const present = (xs: readonly Num[]): number[] => xs.filter((x): x is number => x !== null);

/** Count of each distinct value, in order of first appearance. */
export function countBy(xs: readonly string[]): [string, number][] {
  const m = new Map<string, number>();
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
  return [...m];
}

// ---- row classes --------------------------------------------------------------------

/** A failed evaluation: status other than "ok", no finite f_total, or the failure penalty. */
export function failedRows(d: StatsData): boolean[] {
  return d.fTotal.map((f, i) => d.status[i] !== "ok" || f === null || f >= F_FAIL);
}

/** Row indices of feasible, non-failed evaluations. */
export function feasibleRows(d: StatsData): number[] {
  const failed = failedRows(d);
  return d.feasible.map((f, i) => (f && !failed[i] ? i : -1)).filter((i) => i >= 0);
}

/** Row indices of non-failed evaluations. */
export function okRows(d: StatsData): number[] {
  const failed = failedRows(d);
  return failed.map((x, i) => (x ? -1 : i)).filter((i) => i >= 0);
}

/**
 * The best `TOP_FRACTION` of `rows` by f_total (at least `minN` rows). Returns null when fewer
 * than `minN` rows are available.
 */
export function topRows(d: StatsData, rows: readonly number[], fraction = TOP_FRACTION, minN = TOP_MIN): number[] | null {
  if (rows.length < minN) return null;
  const k = Math.min(rows.length, Math.max(minN, Math.ceil(fraction * rows.length)));
  return [...rows].sort((a, b) => (d.fTotal[a] as number) - (d.fTotal[b] as number) || a - b).slice(0, k);
}

/** Row of the best design: `result.best_eval_id` when present, else the feasible minimum of f_total. */
export function bestRow(d: StatsData): number | null {
  if (d.bestEvalId !== null) {
    const i = d.evalId.indexOf(d.bestEvalId);
    if (i >= 0) return i;
  }
  const feas = feasibleRows(d);
  if (feas.length === 0) return null;
  return feas.reduce((a, b) => ((d.fTotal[b] as number) < (d.fTotal[a] as number) ? b : a));
}

/** Row whose f_total equals the stored baseline f_total (relative 1e-9), or null. */
export function baselineRow(d: StatsData): number | null {
  const b = d.baselineFTotal;
  if (b === null) return null;
  const i = d.fTotal.findIndex((f) => f !== null && Math.abs(f - b) <= 1e-9 * Math.max(1, Math.abs(b)));
  return i >= 0 ? i : null;
}

const at = (col: readonly Num[] | undefined, rows: readonly number[]): number[] => present(rows.map((i) => col?.[i] ?? null));

// ---- summary ------------------------------------------------------------------------

export interface Summary {
  total: number;
  byStatus: [string, number][];
  bySource: [string, number][];
  failed: number;
  feasibleCount: number;
  /** Feasible share of all evaluations, in percent. */
  feasibleShare: Num;
  fFeasible: Describe;
  fOk: Describe;
  seconds: Describe;
}

/** Counts and f_total statistics of the whole history. */
export function summarize(d: StatsData): Summary {
  const feas = feasibleRows(d);
  const failed = failedRows(d);
  return {
    total: d.n,
    byStatus: countBy(d.status),
    bySource: countBy(d.source.map((s) => s || "unknown")),
    failed: failed.filter(Boolean).length,
    feasibleCount: feas.length,
    feasibleShare: d.n === 0 ? null : (100 * feas.length) / d.n,
    fFeasible: describe(at(d.fTotal, feas)),
    fOk: describe(at(d.fTotal, okRows(d))),
    seconds: describe(present(d.seconds)),
  };
}

// ---- score distribution -------------------------------------------------------------

export interface Histogram {
  edges: number[];
  feasible: number[];
  infeasible: number[];
  /** Evaluations left out because they failed. */
  excluded: number;
  best: Num;
  median: Num;
  mean: Num;
}

/** Histogram of f_total of the non-failed rows on shared bins (about sqrt(n) of them, 5 to 40). */
export function scoreHistogram(d: StatsData, bins?: number): Histogram | null {
  const ok = okRows(d);
  const excluded = d.n - ok.length;
  if (ok.length === 0) return null;
  const vals = at(d.fTotal, ok);
  const lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const nb = bins ?? Math.min(40, Math.max(5, Math.ceil(Math.sqrt(ok.length))));
  if (hi === lo) hi = lo + 1;
  const w = (hi - lo) / nb;
  const edges = Array.from({ length: nb + 1 }, (_, k) => lo + k * w);
  const feasible = new Array<number>(nb).fill(0);
  const infeasible = new Array<number>(nb).fill(0);
  for (const i of ok) {
    const k = Math.min(nb - 1, Math.floor(((d.fTotal[i] as number) - lo) / w));
    (d.feasible[i] ? feasible : infeasible)[k]! += 1;
  }
  const fe = describe(at(d.fTotal, feasibleRows(d)));
  return { edges, feasible, infeasible, excluded, best: fe.min, median: fe.median, mean: fe.mean };
}

// ---- objectives ---------------------------------------------------------------------

export interface ObjectiveRow extends Describe {
  name: string;
  values: number[];
  bestDesign: Num;
}

/** Per objective: statistics of the feasible samples and the value at the best design. */
export function objectiveStats(d: StatsData): ObjectiveRow[] {
  const feas = feasibleRows(d);
  const b = bestRow(d);
  return d.objectives.map((name) => {
    const values = at(d.f[name], feas);
    return { name, values, ...describe(values), bestDesign: b === null ? null : (d.f[name]?.[b] ?? null) };
  });
}

// ---- design parameters --------------------------------------------------------------

export interface ParamRow {
  name: string;
  all: Describe;
  top: Describe | null;
  best: Num;
  baseline: Num;
  /** Range explored by the whole history (non-failed rows). */
  lo: Num;
  hi: Num;
}

/** Per design parameter: feasible and top-10 % statistics, best and baseline values. Constant parameters are dropped. */
export function paramStats(d: StatsData): ParamRow[] {
  const feas = feasibleRows(d);
  const top = topRows(d, feas);
  const b = bestRow(d);
  const base = baselineRow(d);
  const ok = okRows(d);
  const rows: ParamRow[] = [];
  for (const [name, col] of Object.entries(d.params)) {
    const range = at(col, ok);
    const lo = range.length ? Math.min(...range) : null;
    const hi = range.length ? Math.max(...range) : null;
    if (lo === null || hi === null || hi - lo < 1e-9) continue;
    rows.push({
      name,
      all: describe(at(col, feas)),
      top: top ? describe(at(col, top)) : null,
      best: b === null ? null : (col[b] ?? null),
      baseline: base === null ? null : (col[base] ?? null),
      lo,
      hi,
    });
  }
  return rows;
}

// ---- constraints --------------------------------------------------------------------

export interface ConstraintGroup {
  n: number;
  /** Percent of the samples with g at or below 0. */
  passRate: Num;
  meanG: Num;
  maxG: Num;
}

export interface ConstraintRow {
  name: string;
  all: ConstraintGroup;
  top: ConstraintGroup | null;
}

const constraintGroup = (col: readonly Num[] | undefined, rows: readonly number[]): ConstraintGroup => {
  const v = at(col, rows);
  return {
    n: v.length,
    passRate: v.length ? (100 * v.filter((x) => x <= 0).length) / v.length : null,
    meanG: mean(v),
    maxG: v.length ? Math.max(...v) : null,
  };
};

/** Per constraint: pass rate, mean and max g over the ok samples, and over the top 10 % of the feasible ones. */
export function constraintStats(d: StatsData): ConstraintRow[] {
  const ok = okRows(d);
  const top = topRows(d, feasibleRows(d));
  return Object.keys(d.g).map((name) => ({
    name,
    all: constraintGroup(d.g[name], ok),
    top: top ? constraintGroup(d.g[name], top) : null,
  }));
}

// ---- evolution ----------------------------------------------------------------------

export interface EvolutionBand {
  /** Mean evaluation id of the window. */
  x: number;
  mean: number;
  /** Sample std; null when the window holds fewer than 2 feasible samples. */
  std: Num;
  n: number;
}

/** Window length for the evolution bands: max(10, n / 20), rounded. */
export const windowSize = (n: number): number => Math.max(10, Math.round(n / 20));

/** Mean and std of f_total of the feasible samples in consecutive windows of evaluations. */
export function evolutionBands(d: StatsData, size = windowSize(d.n)): EvolutionBand[] {
  const feas = new Set(feasibleRows(d));
  const out: EvolutionBand[] = [];
  for (let start = 0; start < d.n; start += size) {
    const rows = Array.from({ length: Math.min(size, d.n - start) }, (_, k) => start + k).filter((i) => feas.has(i));
    if (rows.length === 0) continue;
    const v = at(d.fTotal, rows);
    out.push({ x: mean(rows.map((i) => d.evalId[i] as number)) as number, mean: mean(v) as number, std: std(v), n: v.length });
  }
  return out;
}
