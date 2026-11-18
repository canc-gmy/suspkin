import { useCallback, useMemo, useState } from "react";
import type { DatasetInfo } from "../api";
import { useHistory } from "../api/hooks";
import { PlotlyChart } from "../components/PlotlyChart";
import { Segmented } from "../components/Segmented";
import { OptimizationStats } from "./OptimizationStats";
import {
  convergenceTraces,
  marginRows,
  marginTraces,
  mapHistory,
  paretoTraces,
  parcoordsTrace,
  isLogScale,
  type HistoryView,
  type OptColors,
} from "../lib/optimization";
import { useUi } from "../state/store";
import { chartLayout, plainLayout, readTokens, type PlotlyObject, type Tokens } from "../theme/plotlyTemplate";
import type { Theme } from "../theme/revisions";

const colorsOf = (t: Tokens): OptColors => ({
  accent: t.layoutBlue,
  ink: t.ink,
  muted: t.inkMuted,
  danger: t.danger,
  baseline: t.inkMuted,
});

const label = (objective: string) => objective.replace(/^f_/, "");

function ConvergenceChart({ view }: { view: HistoryView }) {
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      return {
        data: convergenceTraces(view, colorsOf(t)),
        layout: chartLayout(t, {
          xTitle: "Evaluation",
          yTitle: "f_total",
          legend: true,
          yZero: false,
          xZero: false,
          hover: "closest",
          yType: isLogScale(view.fTotal) ? "log" : "linear",
        }),
      };
    },
    [view],
  );
  return <PlotlyChart build={build} ariaLabel="Convergence: f_total against evaluation" filename="opt_convergence" />;
}

function ParetoChart({ view }: { view: HistoryView }) {
  const [xObj, setX] = useState(view.objectives[0] ?? "");
  const [yObj, setY] = useState(view.objectives[1] ?? "");
  const x = view.objectives.includes(xObj) ? xObj : (view.objectives[0] ?? "");
  const y = view.objectives.includes(yObj) ? yObj : (view.objectives[1] ?? "");
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      return {
        data: paretoTraces(view, x, y, colorsOf(t)),
        layout: chartLayout(t, {
          xTitle: label(x),
          yTitle: label(y),
          legend: true,
          yZero: false,
          xZero: false,
          hover: "closest",
        }),
      };
    },
    [view, x, y],
  );
  return (
    <>
      {view.objectives.length > 2 && (
        <div className="toolbar">
          {(
            [
              ["Horizontal objective", x, setX],
              ["Vertical objective", y, setY],
            ] as const
          ).map(([name, value, set]) => (
            <label key={name} className="field">
              <span>{name}</span>
              <select value={value} onChange={(e) => set(e.target.value)}>
                {view.objectives.map((o) => (
                  <option key={o} value={o}>
                    {label(o)}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}
      <PlotlyChart build={build} ariaLabel={`Pareto front: ${label(y)} against ${label(x)}`} filename="opt_pareto" />
    </>
  );
}

function ParcoordsChart({ view }: { view: HistoryView }) {
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const trace = parcoordsTrace(view, colorsOf(t));
      return { data: trace ? [trace] : ([] as PlotlyObject[]), layout: plainLayout(t) };
    },
    [view],
  );
  return <PlotlyChart build={build} ariaLabel="Design variables of the feasible evaluations" filename="opt_parallel" height={420} />;
}

function MarginChart({ result }: { result: Record<string, unknown> }) {
  const rows = useMemo(() => marginRows(result), [result]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, {
        xTitle: "g (0 or below satisfied, above 0 violated; clipped at ±50)",
        yTitle: "",
        legend: true,
        yZero: false,
        hover: "closest",
      });
      return {
        data: marginTraces(rows, colorsOf(t)),
        layout: {
          ...base,
          barmode: "group",
          margin: { ...(base["margin"] as PlotlyObject), l: 170 },
          yaxis: { ...(base["yaxis"] as PlotlyObject), autorange: "reversed" },
        },
      };
    },
    [rows],
  );
  return (
    <PlotlyChart
      build={build}
      ariaLabel="Constraint margins of the baseline and the best design"
      filename="opt_margins"
      height={Math.max(320, rows.length * 34 + 120)}
    />
  );
}

/** Optimizer history of one run: convergence, Pareto front, design variables, constraint margins. */
export function OptimizationView({ datasets }: { datasets: DatasetInfo[] }) {
  const setOpenDialog = useUi((s) => s.setOpenDialog);
  const runs = useMemo(() => {
    const seen = new Map<string, string>();
    for (const d of datasets) {
      if (d.run_id && !seen.has(d.run_id)) {
        const parts = d.source.split("/");
        seen.set(d.run_id, parts.length > 1 ? (parts[parts.length - 2] as string) : d.source);
      }
    }
    return [...seen].map(([runId, name]) => ({ runId, name }));
  }, [datasets]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [sub, setSub] = useState<"history" | "statistics">("history");
  const runId = runs.find((r) => r.runId === chosen)?.runId ?? runs[0]?.runId ?? null;
  const history = useHistory(runId);
  const view = useMemo(() => (history.data ? mapHistory(history.data) : null), [history.data]);

  if (runs.length === 0) {
    return (
      <section aria-label="Optimization">
        <h2 className="view-title">Optimization</h2>
        <p className="muted">Open an optimizer run folder to see its history.</p>
        <button type="button" className="btn secondary" onClick={() => setOpenDialog(true)}>
          Open files
        </button>
      </section>
    );
  }

  return (
    <section aria-label="Optimization">
      <div className="toolbar">
        <h2 className="view-title">Optimization</h2>
        {runs.length > 1 && (
          <label className="field">
            <span>Run</span>
            <select value={runId ?? ""} onChange={(e) => setChosen(e.target.value)}>
              {runs.map((r) => (
                <option key={r.runId} value={r.runId}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <Segmented
          label="Optimization view"
          value={sub}
          options={[
            { value: "history", label: "History" },
            { value: "statistics", label: "Statistics" },
          ]}
          onChange={setSub}
        />
      </div>
      {history.isError && (
        <p className="error" role="alert">
          Could not load the optimizer history: {history.error.message}
        </p>
      )}
      {history.isPending && <p className="muted">Loading history.</p>}
      {history.data && sub === "statistics" && <OptimizationStats history={history.data} />}
      {history.data && view && sub === "history" && (
        <div className="panel-grid">
          <section className="panel" aria-label="Convergence">
            <h3 className="panel-title">Convergence</h3>
            <ConvergenceChart view={view} />
          </section>
          {view.objectives.length >= 2 && (
            <section className="panel" aria-label="Pareto front">
              <h3 className="panel-title">Pareto front</h3>
              <ParetoChart view={view} />
            </section>
          )}
          <section className="panel" aria-label="Design variables">
            <h3 className="panel-title">Design variables</h3>
            <ParcoordsChart view={view} />
          </section>
          <section className="panel" aria-label="Constraint margins">
            <h3 className="panel-title">Constraint margins</h3>
            <MarginChart result={history.data.result} />
          </section>
        </div>
      )}
    </section>
  );
}
