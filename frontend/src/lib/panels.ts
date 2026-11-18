import type { FigureSpec, PanelSpec } from "../api/spec";

/** Column names available in each visible dataset's table for a figure's study. */
export type ColumnSets = readonly ReadonlySet<string>[];

const anyHas = (sets: ColumnSets, col: string) => sets.some((s) => s.has(col));

/**
 * Panels of a figure to draw, following the renderer rules of the figure spec.
 *
 * A panel is drawn when at least one of its series exists in some dataset, its x column
 * exists, and every `requires` column exists in at least one dataset. A panel with
 * `alternative_to` is drawn only when the panel it names is not drawn.
 */
export function selectPanels(fig: FigureSpec, columns: ColumnSets): PanelSpec[] {
  const drawn = new Set<string>();
  const out: PanelSpec[] = [];
  for (const p of fig.panels) {
    if (p.alternative_to !== null && drawn.has(p.alternative_to)) continue;
    const xCol = (p.x ?? fig.x).column;
    const ok =
      p.series.some((s) => anyHas(columns, s.column)) &&
      anyHas(columns, xCol) &&
      p.requires.every((c) => anyHas(columns, c));
    if (ok) {
      drawn.add(p.id);
      out.push(p);
    }
  }
  return out;
}

/** Series of a panel that exist in a dataset, keeping their index (it picks the line style). */
export function seriesIn(panel: PanelSpec, columns: ReadonlySet<string>) {
  return panel.series
    .map((series, index) => ({ series, index }))
    .filter(({ series }) => columns.has(series.column));
}

const GROUP_LABELS: Record<string, string> = {
  corner_alignment: "Alignment",
  corner_position: "Position",
  corner_centres: "Centres",
  actuation: "Actuation",
  steering: "Steering",
  roll: "Roll",
  heave_pitch: "Heave & pitch",
};

/** Human tab label of a figure group. */
export function groupLabel(group: string): string {
  const known = GROUP_LABELS[group];
  if (known) return known;
  const s = group.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Figure groups in spec order. */
export function groupsOf(figures: readonly FigureSpec[]): string[] {
  return [...new Set(figures.map((f) => f.group))];
}

export type Axle = "front" | "rear";

/** True when a group has separate front and rear figures (ids ending `_front` / `_rear`). */
export function hasAxles(figs: readonly FigureSpec[]): boolean {
  return figs.some((f) => /_(front|rear)$/.test(f.id));
}

/** Figures of a group to show for the chosen axle (all of them for groups without axles). */
export function figuresFor(figures: readonly FigureSpec[], group: string, axle: Axle): FigureSpec[] {
  const figs = figures.filter((f) => f.group === group);
  return hasAxles(figs) ? figs.filter((f) => f.id.endsWith(`_${axle}`)) : figs;
}
