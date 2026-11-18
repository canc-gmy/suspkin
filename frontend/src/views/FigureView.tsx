import { useEffect, useRef } from "react";
import type { DatasetInfo, SpecDoc } from "../api";
import { Segmented } from "../components/Segmented";
import { SpecFigure } from "../components/SpecFigure";
import { figuresFor, groupLabel, hasAxles, type Axle } from "../lib/panels";
import { useUi } from "../state/store";

interface Props {
  group: string;
  spec: SpecDoc;
  datasets: DatasetInfo[];
}

/** One figure group: Front|Rear and Overlay|Δ controls above a grid of panels. */
export function FigureView({ group, spec, datasets }: Props) {
  const axle = useUi((s) => s.axle);
  const setAxle = useUi((s) => s.setAxle);
  const mode = useUi((s) => s.mode);
  const setMode = useUi((s) => s.setMode);
  const reference = useUi((s) => s.reference);
  const refLetter = useUi((s) => (s.reference ? s.letters[s.reference] : undefined));

  const toolbar = useRef<HTMLDivElement>(null);
  const section = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = toolbar.current;
    const host = section.current;
    if (!el || !host || typeof ResizeObserver === "undefined") return;
    const set = () => host.style.setProperty("--toolbar-h", `${el.offsetHeight}px`);
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const all = spec.figures.filter((f) => f.group === group);
  const figs = figuresFor(spec.figures, group, axle as Axle);
  const axles = hasAxles(all);

  return (
    <section aria-label={groupLabel(group)} ref={section}>
      <div className="toolbar figure-toolbar" ref={toolbar}>
        <h2 className="view-title">{groupLabel(group)}</h2>
        {axles && (
          <Segmented<Axle>
            label="Axle"
            value={axle}
            onChange={setAxle}
            options={[
              { value: "front", label: "Front" },
              { value: "rear", label: "Rear" },
            ]}
          />
        )}
        <span className="spacer" />
        <Segmented<"overlay" | "delta">
          label="Comparison"
          value={mode}
          onChange={setMode}
          options={[
            { value: "overlay", label: "Overlay" },
            { value: "delta", label: reference ? `Δ to rev ${refLetter ?? "?"}` : "Δ to reference" },
          ]}
        />
      </div>
      {figs.map((fig) => (
        <SpecFigure key={fig.id} fig={fig} spec={spec} datasets={datasets} showTitle={figs.length > 1} />
      ))}
    </section>
  );
}
