import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import type { PlotlyHTMLElement } from "plotly.js-dist-min";
import type { PlotlyObject } from "../theme/plotlyTemplate";
import type { Theme } from "../theme/revisions";
import { useUi } from "../state/store";

export interface ChartFigure {
  data: PlotlyObject[];
  layout: PlotlyObject;
}

export interface ChartHandle {
  /** Show the unified hover of this chart at x (used to link sibling charts). */
  hoverAt: (x: number) => void;
  clearHover: () => void;
}

interface Props {
  /** Builds the figure for a theme; called with the current theme and with "light" for PNG export. */
  build: (theme: Theme) => ChartFigure;
  ariaLabel: string;
  filename: string;
  height?: number;
  /** Extra modebar buttons after the PNG download (default: reset axes). */
  buttons?: string[];
  /** x value under the cursor, or null when the cursor leaves. */
  onHoverX?: (x: number | null) => void;
}

const DEFAULT_BUTTONS = ["resetScale2d"];

type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule["default"]> | null = null;

/** Plotly is large: load it on first use so the app shell appears immediately. */
function loadPlotly(): Promise<PlotlyModule["default"]> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m) => m.default);
  return plotlyPromise;
}

function hoveredX(e: unknown): number | null {
  const ev = e as { xvals?: unknown[]; points?: { x?: unknown }[] } | undefined;
  const v = ev?.xvals?.[0] ?? ev?.points?.[0]?.x;
  return typeof v === "number" ? v : null;
}

/** A single Plotly chart rendered with `Plotly.react` (no wrapper library). */
export const PlotlyChart = forwardRef<ChartHandle, Props>(function PlotlyChart(
  { build, ariaLabel, filename, height = 320, buttons = DEFAULT_BUTTONS, onHoverX },
  ref,
) {
  const host = useRef<HTMLDivElement>(null);
  const theme = useUi((s) => s.theme);
  const hoverCb = useRef(onHoverX);
  const buildRef = useRef(build);
  const attached = useRef(false);

  useEffect(() => {
    hoverCb.current = onHoverX;
    buildRef.current = build;
  });

  useImperativeHandle(
    ref,
    () => ({
      hoverAt: (x) => {
        const el = host.current;
        if (!el || !attached.current) return;
        void loadPlotly().then((P) => P.Fx.hover(el, { xval: x }, "x2y"));
      },
      clearHover: () => {
        const el = host.current;
        if (!el || !attached.current) return;
        void loadPlotly().then((P) => P.Fx.unhover(el));
      },
    }),
    [],
  );

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let cancelled = false;
    void loadPlotly().then(async (P) => {
      if (cancelled) return;
      const fig = build(theme);
      const pngButton = {
        name: "downloadPngLight",
        title: "Download PNG (light theme, 2x)",
        icon: P.Icons["camera"],
        click: async (gd: HTMLElement) => {
          const light = buildRef.current("light");
          const tmp = document.createElement("div");
          const w = gd.clientWidth || 800;
          const h = gd.clientHeight || height;
          tmp.style.cssText = `position:fixed;left:-10000px;top:0;width:${w}px;height:${h}px`;
          document.body.appendChild(tmp);
          try {
            await P.newPlot(tmp, light.data, { ...light.layout, width: w, height: h }, { staticPlot: true });
            await P.downloadImage(tmp, { format: "png", scale: 2, width: w, height: h, filename });
          } finally {
            P.purge(tmp);
            tmp.remove();
          }
        },
      };
      const gd = await P.react(el, fig.data, fig.layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: "hover",
        modeBarButtons: [[pngButton, ...buttons]],
      });
      if (cancelled || attached.current) return;
      attached.current = true;
      (gd as PlotlyHTMLElement).on("plotly_hover", (e) => hoverCb.current?.(hoveredX(e)));
      (gd as PlotlyHTMLElement).on("plotly_unhover", () => hoverCb.current?.(null));
    });
    return () => {
      cancelled = true;
    };
  }, [build, theme, filename, height, buttons]);

  useEffect(() => {
    const el = host.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    let frame = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (attached.current) void loadPlotly().then((P) => P.Plots.resize(el));
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
    };
  }, []);

  useEffect(() => {
    const el = host.current;
    return () => {
      attached.current = false;
      if (el) void loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  return <div ref={host} className="chart" role="img" aria-label={ariaLabel} style={{ height }} />;
});
