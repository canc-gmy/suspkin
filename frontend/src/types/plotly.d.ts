declare module "plotly.js-dist-min" {
  export interface PlotlyHTMLElement extends HTMLElement {
    on(event: string, handler: (data: unknown) => void): void;
    removeAllListeners(event: string): void;
  }

  interface PlotlyStatic {
    react(
      root: HTMLElement,
      data: Record<string, unknown>[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<PlotlyHTMLElement>;
    newPlot(
      root: HTMLElement,
      data: Record<string, unknown>[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<PlotlyHTMLElement>;
    relayout(root: HTMLElement, update: Record<string, unknown>): Promise<PlotlyHTMLElement>;
    purge(root: HTMLElement): void;
    downloadImage(root: HTMLElement, opts: Record<string, unknown>): Promise<string>;
    Fx: {
      hover(root: HTMLElement, points: Record<string, unknown> | Record<string, unknown>[], subplot?: string): void;
      unhover(root: HTMLElement): void;
    };
    Icons: Record<string, unknown>;
    Plots: { resize(root: HTMLElement): void };
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
