import { useEffect, useRef } from "react";
import type { DatasetInfo, SpecDoc } from "../api";
import { Segmented } from "../components/Segmented";
import { SpecFigure } from "../components/SpecFigure";
import { figureColumns, groupLabel, hasAxles, type AxleChoice } from "../lib/panels";
import { useUi } from "../state/store";

interface Props {
  group: string;
  spec: SpecDoc;
  datasets: DatasetInfo[];
}

/** One figure group: Front|Rear|Both and Overlay|Δ controls above a grid of panels. */
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
  const columns = figureColumns(spec.figures, group, axle);
  const axles = hasAxles(all);
  const both = columns.length > 1;
  const showTitle = both || columns.some((c) => c.length > 1);

  return (
    <section aria-label={groupLabel(group)} ref={section}>
      <div className="toolbar figure-toolbar" ref={toolbar}>
        <h2 className="view-title">{groupLabel(group)}</h2>
        {axles && (
          <Segmented<AxleChoice>
            label="Axle"
            value={axle}
            onChange={setAxle}
            options={[
              { value: "front", label: "Front" },
              { value: "rear", label: "Rear" },
              { value: "both", label: "Both" },
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
      <div className={both ? "figure-columns both" : "figure-columns"}>
        {columns.map((figs) => (
          <div key={figs.map((f) => f.id).join("|")} className="figure-column">
            {figs.map((fig) => (
              <SpecFigure key={fig.id} fig={fig} spec={spec} datasets={datasets} showTitle={showTitle} />
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
