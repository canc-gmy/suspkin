import type { Theme } from "./revisions";

/** Resolved design tokens (CSS custom properties) for one theme. */
export interface Tokens {
  sheet: string;
  sheetRaised: string;
  ink: string;
  inkMuted: string;
  grid: string;
  gridMinor: string;
  rule: string;
  layoutBlue: string;
  danger: string;
  regionDomain: string;
  regionRequired: string;
  regionKeepout: string;
  fontUi: string;
  /** Palette-index colour overrides present on this theme (`--rev-N`). */
  rev: Record<number, string>;
}

/** Plotly objects are passed through as plain records; the template owns their structure. */
export type PlotlyObject = Record<string, unknown>;

/** Read the tokens of `theme` from the computed CSS (a hidden probe element carries `data-theme`). */
export function readTokens(theme: Theme): Tokens {
  const probe = document.createElement("div");
  probe.dataset.theme = theme;
  probe.style.display = "none";
  document.body.appendChild(probe);
  const cs = getComputedStyle(probe);
  const get = (name: string) => cs.getPropertyValue(name).trim();
  const rev: Record<number, string> = {};
  for (let i = 0; i < 16; i++) {
    const v = get(`--rev-${i}`);
    if (v) rev[i] = v;
  }
  const tokens: Tokens = {
    sheet: get("--sheet"),
    sheetRaised: get("--sheet-raised"),
    ink: get("--ink"),
    inkMuted: get("--ink-muted"),
    grid: get("--grid"),
    gridMinor: get("--grid-minor"),
    rule: get("--rule"),
    layoutBlue: get("--layout-blue"),
    danger: get("--danger"),
    regionDomain: get("--region-domain"),
    regionRequired: get("--region-required"),
    regionKeepout: get("--region-keepout"),
    fontUi: get("--font-ui") || "Barlow, sans-serif",
    rev,
  };
  probe.remove();
  return tokens;
}

export const LINE_WIDTH = 1.75;

export interface LayoutOptions {
  xTitle: string;
  yTitle: string;
  /** Show the panel legend (more than one series). */
  legend: boolean;
  /** Draw a zero line on y (Δ mode) in addition to the static-pose line on x. */
  yZero: boolean;
  /**
   * "closest" shows Plotly hover labels (scatter charts); "none" (default) turns them off: the
   * curve panels are read in the readout strip and mark the cursor with a layout shape.
   */
  hover?: "none" | "closest";
  /** Draw the static-pose zero line on x (default true). */
  xZero?: boolean;
  yType?: "linear" | "log";
}

function axis(t: Tokens, title: string, zero: boolean): PlotlyObject {
  return {
    title: { text: title, font: { family: t.fontUi, size: 14, color: t.ink }, standoff: 8 },
    showgrid: true,
    gridcolor: t.grid,
    gridwidth: 1,
    minor: { showgrid: true, gridcolor: t.gridMinor, gridwidth: 1, griddash: "dot" },
    showline: true,
    linecolor: t.ink,
    linewidth: 1,
    mirror: false,
    ticks: "outside",
    tickcolor: t.ink,
    ticklen: 4,
    tickfont: { family: t.fontUi, size: 12, color: t.inkMuted },
    zeroline: zero,
    zerolinecolor: t.inkMuted,
    zerolinewidth: 1,
  };
}

/** Plotly layout for one panel, from the DESIGN.md "Plot template". */
export function chartLayout(t: Tokens, o: LayoutOptions): PlotlyObject {
  const closest = o.hover === "closest";
  return {
    paper_bgcolor: t.sheet,
    plot_bgcolor: t.sheetRaised,
    font: { family: t.fontUi, size: 12, color: t.ink },
    margin: { l: 72, r: 16, t: o.legend ? 32 : 8, b: 52, pad: 0 },
    hovermode: closest ? "closest" : false,
    hoverdistance: closest ? 20 : -1,
    hoverlabel: {
      bgcolor: t.sheetRaised,
      bordercolor: t.rule,
      font: { family: t.fontUi, size: 13, color: t.ink },
    },
    showlegend: o.legend,
    legend: {
      orientation: "h",
      x: 1,
      xanchor: "right",
      y: 1.02,
      yanchor: "bottom",
      font: { family: t.fontUi, size: 12, color: t.inkMuted },
      bgcolor: "rgba(0,0,0,0)",
    },
    xaxis: axis(t, o.xTitle, o.xZero ?? true),
    yaxis: { ...axis(t, o.yTitle, o.yZero), ...(o.yType ? { type: o.yType } : {}) },
  };
}

/** Shape of the cursor line on a curve panel: dashed muted while following, solid blue when pinned. */
export function cursorShape(t: Tokens, x: number, pinned: boolean): PlotlyObject {
  return {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    layer: "above",
    line: pinned
      ? { color: t.layoutBlue, width: 1, dash: "solid" }
      : { color: t.inkMuted, width: 1, dash: "dash" },
  };
}

/** Layout for charts without cartesian axes (parallel coordinates). */
export function plainLayout(t: Tokens): PlotlyObject {
  return {
    paper_bgcolor: t.sheet,
    plot_bgcolor: t.sheetRaised,
    font: { family: t.fontUi, size: 12, color: t.ink },
    margin: { l: 72, r: 130, t: 110, b: 24 },
  };
}

/**
 * Apply the theme to a server-built 3D scene: background, grid and axes from the tokens, and
 * the near-black geometry (wheels, hardpoints, contact patches) in ink so it reads on both sheets.
 */
export function themeScene(
  fig: { data?: unknown; layout?: unknown },
  t: Tokens,
  opts: { legend?: boolean } = {},
): { data: PlotlyObject[]; layout: PlotlyObject } {
  const legendOn = opts.legend ?? true;
  const DARK = new Set(["#222222", "black", "#000000"]);
  const recolor = (obj: unknown): unknown => {
    if (!obj || typeof obj !== "object") return obj;
    const o = { ...(obj as Record<string, unknown>) };
    if (typeof o["color"] === "string" && DARK.has(o["color"].toLowerCase())) o["color"] = t.ink;
    return o;
  };
  const data = ((fig.data as PlotlyObject[] | undefined) ?? []).map((tr) => {
    const out: PlotlyObject = { ...tr };
    if (tr["line"]) out["line"] = recolor(tr["line"]);
    if (tr["marker"]) out["marker"] = recolor(tr["marker"]);
    if (tr["type"] === "mesh3d") out["color"] = t.rule;
    return out;
  });
  const base = (fig.layout as PlotlyObject | undefined) ?? {};
  const scene = (base["scene"] as PlotlyObject | undefined) ?? {};
  const axisOf = (name: string): PlotlyObject => {
    const a = (scene[name] as PlotlyObject | undefined) ?? {};
    const title = typeof a["title"] === "string" ? { text: a["title"] } : ((a["title"] as PlotlyObject | undefined) ?? {});
    return {
      ...a,
      title: { ...title, font: { family: t.fontUi, size: 13, color: t.ink } },
      color: t.ink,
      showbackground: true,
      backgroundcolor: t.sheetRaised,
      gridcolor: t.grid,
      zerolinecolor: t.inkMuted,
      linecolor: t.ink,
      tickfont: { family: t.fontUi, size: 11, color: t.inkMuted },
    };
  };
  return {
    data,
    layout: {
      ...base,
      title: { text: "" },
      uirevision: "pose", // keep the camera when the pose (and so the data) changes
      // one surface: paper, scene and axis walls share --sheet-raised, so there is no box or seam
      paper_bgcolor: t.sheetRaised,
      font: { family: t.fontUi, size: 12, color: t.ink },
      // the legend is a compact row in a strip at the top of the canvas, above the scene, so it
      // never covers the car; the strip is only reserved while the legend is on
      margin: { l: 0, r: 0, t: legendOn ? 48 : 8, b: 0 },
      showlegend: legendOn,
      legend: {
        x: 0,
        xanchor: "left",
        y: 1,
        yanchor: "bottom",
        orientation: "h",
        font: { family: t.fontUi, size: 12, color: t.inkMuted },
        bgcolor: "rgba(0,0,0,0)",
      },
      scene: {
        ...scene,
        xaxis: axisOf("xaxis"),
        yaxis: axisOf("yaxis"),
        zaxis: axisOf("zaxis"),
        bgcolor: t.sheetRaised,
        uirevision: "pose",
      },
    },
  };
}
