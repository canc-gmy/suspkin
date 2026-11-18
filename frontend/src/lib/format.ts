import type { SummaryRowSpec } from "../api/spec";

export const MINUS = "−";
export const MISSING = "-";

/** Fixed-decimals text of an already-scaled value; no minus sign for a value that rounds to zero. */
export function fixed(value: number | null | undefined, decimals: number): string {
  if (value == null || !Number.isFinite(value)) return MISSING;
  const s = value.toFixed(decimals);
  if (/^-0(\.0*)?$/.test(s)) return s.slice(1);
  return s.startsWith("-") ? MINUS + s.slice(1) : s;
}

/** Summary cell: `value * factor` with `decimals` decimals ("-" when missing). */
export function formatValue(value: number | null | undefined, factor: number, decimals: number): string {
  return value == null || !Number.isFinite(value) ? MISSING : fixed(value * factor, decimals);
}

/** Δ cell: a plain signed number (`+0.12`, `−0.12`, `0.00`), "-" when either side is missing. */
export function formatDelta(
  value: number | null | undefined,
  ref: number | null | undefined,
  factor: number,
  decimals: number,
): string {
  if (value == null || ref == null || !Number.isFinite(value) || !Number.isFinite(ref)) return MISSING;
  const s = fixed((value - ref) * factor, decimals);
  return s.startsWith(MINUS) || /^0(\.0*)?$/.test(s) ? s : `+${s}`;
}

/** Number at `path` in a nested summary dict, or null when absent / not a finite number. */
export function lookup(summary: unknown, path: readonly string[]): number | null {
  let cur: unknown = summary;
  for (const key of path) {
    if (cur === null || typeof cur !== "object") return null;
    cur = (cur as Record<string, unknown>)[key];
  }
  return typeof cur === "number" && Number.isFinite(cur) ? cur : null;
}

/** Whether a summary row exists in a summary (rows whose path is absent are left out). */
export function rowPresent(row: SummaryRowSpec, summaries: readonly unknown[]): boolean {
  return summaries.some((s) => hasPath(s, row.path));
}

function hasPath(obj: unknown, path: readonly string[]): boolean {
  let cur: unknown = obj;
  for (const key of path) {
    if (cur === null || typeof cur !== "object" || !(key in (cur as object))) return false;
    cur = (cur as Record<string, unknown>)[key];
  }
  return true;
}

/** Axis or table label such as "Camber [deg]". */
export function withUnit(label: string, unit: string): string {
  return unit ? `${label} [${unit}]` : label;
}
