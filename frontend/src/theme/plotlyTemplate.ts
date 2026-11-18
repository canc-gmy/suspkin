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
  return {
    paper_bgcolor: t.sheet,
    plot_bgcolor: t.sheetRaised,
    font: { family: t.fontUi, size: 12, color: t.ink },
    margin: { l: 72, r: 16, t: o.legend ? 32 : 8, b: 52, pad: 0 },
    hovermode: "x unified",
    hoverdistance: -1,
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
    xaxis: {
      ...axis(t, o.xTitle, true),
      showspikes: true,
      spikemode: "across",
      spikesnap: "cursor",
      spikecolor: t.inkMuted,
      spikethickness: 1,
      spikedash: "dot",
    },
    yaxis: axis(t, o.yTitle, o.yZero),
  };
}
