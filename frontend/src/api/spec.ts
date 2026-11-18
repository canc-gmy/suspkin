// Hand-typed figure spec served by GET /api/v1/figures (see plotting/spec.py).

export interface AxisSpec {
  column: string;
  label: string;
  dim: string;
  factor: number;
  unit: string;
}

export interface SeriesSpec {
  column: string;
  label: string;
  dim: string;
  side: "left" | "right" | "front" | "rear" | null;
  corner: string | null;
  factor: number;
  unit: string;
}

export interface PanelSpec {
  id: string;
  title: string;
  y_label: string;
  y_dim: string;
  ic: boolean;
  requires: string[];
  alternative_to: string | null;
  x: AxisSpec | null;
  series: SeriesSpec[];
}

export interface FigureSpec {
  id: string;
  group: string;
  title: string;
  study: string;
  columns: number;
  x: AxisSpec;
  panels: PanelSpec[];
}

export interface SummaryRowSpec {
  label: string;
  path: string[];
  dim: string | null;
  factor: number;
  unit: string;
  decimals: number;
}

export interface SpecDoc {
  schema: number;
  palette: string[];
  linestyles: string[];
  figures: FigureSpec[];
  summary_rows: SummaryRowSpec[];
}
