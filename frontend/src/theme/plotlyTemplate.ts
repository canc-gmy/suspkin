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
  /** "x unified" (default, curves over a shared x) or "closest" (scatter charts). */
  hover?: "x unified" | "closest";
  /** Draw the static-pose zero line on x (default true). */
  xZero?: boolean;
  yType?: "linear" | "log";
  /** Show the x spike line across panels (default true for unified hover). */
  spikes?: boolean;
  /**
   * Heading of the unified hover ("Wheel travel 12 mm"). Needs every trace on `xaxis: "x2"`:
   * a hidden overlaying axis carries the label as tick prefix and the unit as suffix (shown
   * only for the hover value, never on ticks).
   */
  xHover?: { label: string; unit: string };
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
  const xaxis = {
    ...axis(t, o.xTitle, o.xZero ?? true),
    showspikes: o.spikes ?? (o.hover ?? "x unified") === "x unified",
    spikemode: "across",
    spikesnap: "cursor",
    spikecolor: t.inkMuted,
    spikethickness: 1,
    spikedash: "dot",
  };
  const hoverAxis = o.xHover
    ? {
        xaxis2: {
          overlaying: "x",
          matches: "x",
          showticklabels: false,
          showgrid: false,
          zeroline: false,
          showline: false,
          ticks: "",
          hoverformat: ".4~g",
          tickprefix: `${o.xHover.label} `,
          showtickprefix: "first",
          ticksuffix: o.xHover.unit ? ` ${o.xHover.unit}` : "",
          showticksuffix: "first",
          showspikes: xaxis.showspikes,
          spikemode: "across",
          spikesnap: "data" /* programmatic (linked) hover has no cursor to snap to */,
          spikecolor: t.inkMuted,
          spikethickness: 1,
          spikedash: "dot",
        },
      }
    : {};
  return {
    ...hoverAxis,
    paper_bgcolor: t.sheet,
    plot_bgcolor: t.sheetRaised,
    font: { family: t.fontUi, size: 12, color: t.ink },
    margin: { l: 72, r: 16, t: o.legend ? 32 : 8, b: 52, pad: 0 },
    hovermode: o.hover ?? "x unified",
    hoverdistance: o.hover === "closest" ? 20 : -1,
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
    xaxis: { ...xaxis, ...(o.xHover ? { showspikes: false } : {}) },
    yaxis: { ...axis(t, o.yTitle, o.yZero), ...(o.yType ? { type: o.yType } : {}) },
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
export function themeScene(fig: { data?: unknown; layout?: unknown }, t: Tokens): { data: PlotlyObject[]; layout: PlotlyObject } {
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
      paper_bgcolor: t.sheet,
      font: { family: t.fontUi, size: 12, color: t.ink },
      margin: { l: 0, r: 0, t: 8, b: 0 },
      legend: { font: { family: t.fontUi, size: 12, color: t.inkMuted }, bgcolor: "rgba(0,0,0,0)" },
      scene: {
        ...scene,
        xaxis: axisOf("xaxis"),
        yaxis: axisOf("yaxis"),
        zaxis: axisOf("zaxis"),
        bgcolor: t.sheet,
        uirevision: "pose",
      },
    },
  };
}
