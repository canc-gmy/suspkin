import { useEffect, useRef } from "react";
import { pointerToX, type PixelAxis } from "../lib/readout";
import { cursorShape, readTokens, type PlotlyObject, type Tokens } from "../theme/plotlyTemplate";
import type { Theme } from "../theme/revisions";
import { useUi } from "../state/store";

export interface ChartFigure {
  data: PlotlyObject[];
  layout: PlotlyObject;
}

/** Vertical cursor drawn on a chart (a layout shape): dashed while following, solid when pinned. */
export interface ChartCursor {
  x: number | null;
  pinned: boolean;
}

interface Props {
  /** Builds the figure for a theme; called with the current theme and with "light" for PNG export. */
  build: (theme: Theme) => ChartFigure;
  ariaLabel: string;
  filename: string;
  height?: number;
  /** Extra modebar buttons after the PNG download (default: reset axes). */
  buttons?: string[];
  /** Cursor line to draw; omit for charts that do not show one. */
  cursor?: ChartCursor | null;
  /** Data x under the pointer (null when it leaves the plot area). Set it to track the pointer. */
  onPointerX?: (x: number | null) => void;
  /** A click (not a drag) at data x. */
  onPointerClick?: (x: number) => void;
}

const DEFAULT_BUTTONS = ["resetScale2d"];
/** Pointer travel (px) between press and release above which a click counts as a drag (zoom). */
const DRAG_PX = 4;

type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule["default"]> | null = null;

/** Plotly is large: load it on first use so the app shell appears immediately. */
function loadPlotly(): Promise<PlotlyModule["default"]> {
  plotlyPromise ??= import("plotly.js-dist-min").then((m) => m.default);
  return plotlyPromise;
}

interface FullLayoutHost extends HTMLElement {
  _fullLayout?: { xaxis?: PixelAxis; yaxis?: PixelAxis };
}

/** Data x under a mouse event, read from the chart's current axes (so it follows zoom and resize). */
function eventX(el: FullLayoutHost, e: MouseEvent): number | null {
  const full = el._fullLayout;
  if (!full?.xaxis || !full.yaxis) return null;
  return pointerToX(e.clientX, e.clientY, el.getBoundingClientRect(), full.xaxis, full.yaxis);
}

/** A single Plotly chart rendered with `Plotly.react` (no wrapper library). */
export function PlotlyChart({
  build,
  ariaLabel,
  filename,
  height = 320,
  buttons = DEFAULT_BUTTONS,
  cursor,
  onPointerX,
  onPointerClick,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const theme = useUi((s) => s.theme);
  const buildRef = useRef(build);
  const cursorRef = useRef(cursor);
  const tokens = useRef<Tokens | null>(null);
  const moveCb = useRef(onPointerX);
  const clickCb = useRef(onPointerClick);
  const attached = useRef(false);

  useEffect(() => {
    buildRef.current = build;
    cursorRef.current = cursor;
    moveCb.current = onPointerX;
    clickCb.current = onPointerClick;
  });

  const applyCursor = (P: PlotlyModule["default"], el: HTMLElement) => {
    const c = cursorRef.current;
    const t = tokens.current;
    if (!c || !t || !attached.current) return;
    void P.relayout(el, { shapes: c.x === null ? [] : [cursorShape(t, c.x, c.pinned)] });
  };

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let cancelled = false;
    void loadPlotly().then(async (P) => {
      if (cancelled) return;
      const fig = build(theme);
      tokens.current = readTokens(theme);
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
      await P.react(el, fig.data, fig.layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: "hover",
        modeBarButtons: [[pngButton, ...buttons]],
      });
      if (cancelled) return;
      attached.current = true;
      applyCursor(P, el); // `react` replaces the layout, so the cursor shape is drawn again
    });
    return () => {
      cancelled = true;
    };
  }, [build, theme, filename, height, buttons]);

  const cx = cursor?.x ?? null;
  const pinned = cursor?.pinned ?? false;
  useEffect(() => {
    const el = host.current;
    if (!el || !attached.current) return;
    void loadPlotly().then((P) => applyCursor(P, el));
  }, [cx, pinned]);

  const tracksPointer = onPointerX !== undefined || onPointerClick !== undefined;
  useEffect(() => {
    const el = host.current;
    if (!el || !tracksPointer) return;
    let down: { x: number; y: number } | null = null;
    const move = (e: MouseEvent) => moveCb.current?.(eventX(el, e));
    const leave = () => moveCb.current?.(null);
    const press = (e: MouseEvent) => {
      down = { x: e.clientX, y: e.clientY };
    };
    const click = (e: MouseEvent) => {
      const moved = down ? Math.hypot(e.clientX - down.x, e.clientY - down.y) : 0;
      down = null;
      if (moved > DRAG_PX) return;
      const x = eventX(el, e);
      if (x !== null) clickCb.current?.(x);
    };
    el.addEventListener("mousemove", move);
    el.addEventListener("mouseleave", leave);
    el.addEventListener("mousedown", press);
    el.addEventListener("click", click);
    return () => {
      el.removeEventListener("mousemove", move);
      el.removeEventListener("mouseleave", leave);
      el.removeEventListener("mousedown", press);
      el.removeEventListener("click", click);
    };
  }, [tracksPointer]);

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
}
