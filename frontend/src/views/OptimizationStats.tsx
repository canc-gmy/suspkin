import { useCallback, useMemo, useState } from "react";
import type { HistoryData } from "../api/types";
import { PlotlyChart } from "../components/PlotlyChart";
import { Segmented } from "../components/Segmented";
import {
  F_FAIL,
  bestRow,
  constraintStats,
  defaultPhase,
  evolutionBands,
  filterPhase,
  mapStats,
  objectiveStats,
  paramStats,
  phasesOf,
  scoreHistogram,
  summarize,
  windowSize,
  type Describe,
  type StatsData,
} from "../lib/historyStats";
import { boxTraces, evolutionTraces, histogramTraces, paramTraces, passRateTraces } from "../lib/historyStatsCharts";
import type { OptColors } from "../lib/optimization";
import { chartLayout, readTokens, type PlotlyObject, type Tokens } from "../theme/plotlyTemplate";
import type { Theme } from "../theme/revisions";
import "./OptimizationStats.css";

const colorsOf = (t: Tokens): OptColors => ({
  accent: t.layoutBlue,
  ink: t.ink,
  muted: t.inkMuted,
  danger: t.danger,
  baseline: t.inkMuted,
});

/** Fixed decimals, "-" for a missing value. */
export const fmt = (v: number | null, digits = 3): string => (v === null ? "-" : v.toFixed(digits));
/** "mean ± std", "-" when there is no mean. */
export const fmtPm = (m: number | null, s: number | null, digits = 3): string =>
  m === null ? "-" : s === null ? fmt(m, digits) : `${fmt(m, digits)} ± ${fmt(s, digits)}`;

const NO_FEASIBLE = "No feasible evaluation yet, so there are no statistics to show.";

function SummaryTable({ d }: { d: StatsData }) {
  const s = useMemo(() => summarize(d), [d]);
  const row = (name: string, a: Describe) => (
    <tr key={name}>
      <th scope="row">{name}</th>
      <td className="num">{fmt(a.min)}</td>
      <td className="num">{fmt(a.median)}</td>
      <td className="num">{fmt(a.mean)}</td>
      <td className="num">{fmt(a.std)}</td>
      <td className="num">{a.n}</td>
    </tr>
  );
  const count = (pairs: [string, number][]) => pairs.map(([k, v]) => `${k} ${v}`).join(", ");
  return (
    <>
      <div className="table-scroll">
        <table className="ruled stats-table" aria-label="Evaluation counts">
          <tbody>
            <tr>
              <th scope="row">Evaluations</th>
              <td className="num">{s.total}</td>
            </tr>
            <tr>
              <th scope="row">By status</th>
              <td className="num">{count(s.byStatus) || "-"}</td>
            </tr>
            <tr>
              <th scope="row">By phase</th>
              <td className="num">{count(s.bySource) || "-"}</td>
            </tr>
            <tr>
              <th scope="row">Feasible share</th>
              <td className="num">
                {s.feasibleShare === null ? "-" : `${s.feasibleShare.toFixed(1)} % (${s.feasibleCount} of ${s.total})`}
              </td>
            </tr>
            <tr>
              <th scope="row">Time per evaluation</th>
              <td className="num">{s.seconds.mean === null ? "-" : `${fmt(s.seconds.mean, 2)} s`}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="table-scroll">
        <table className="ruled stats-table" aria-label="f_total statistics">
          <thead>
            <tr>
              <th scope="col">f_total</th>
              <th scope="col" className="num">
                Best
              </th>
              <th scope="col" className="num">
                Median
              </th>
              <th scope="col" className="num">
                Mean
              </th>
              <th scope="col" className="num">
                Std
              </th>
              <th scope="col" className="num">
                n
              </th>
            </tr>
          </thead>
          <tbody>
            {row("Feasible", s.fFeasible)}
            {row("All ok", s.fOk)}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ScoreChart({ d }: { d: StatsData }) {
  const h = useMemo(() => scoreHistogram(d), [d]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, { xTitle: "f_total", yTitle: "Samples", legend: true, yZero: false, xZero: false, hover: "closest" });
      return { data: h ? histogramTraces(h, colorsOf(t)) : ([] as PlotlyObject[]), layout: { ...base, barmode: "overlay" } };
    },
    [h],
  );
  if (!h) return <p className="muted">No evaluation produced a score.</p>;
  return (
    <>
      <p className="stats-note">
        {h.excluded === 0
          ? "No failed evaluations."
          : `${h.excluded} failed evaluation${h.excluded === 1 ? "" : "s"} not shown.`}
      </p>
      <PlotlyChart build={build} ariaLabel="Distribution of f_total" filename="opt_stats_score" />
    </>
  );
}

function ObjectivesPanel({ d }: { d: StatsData }) {
  const rows = useMemo(() => objectiveStats(d), [d]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, { xTitle: "", yTitle: "f (normalised)", legend: false, yZero: false, xZero: false, hover: "closest" });
      return { data: boxTraces(rows, colorsOf(t)), layout: base };
    },
    [rows],
  );
  if (rows.length === 0) return <p className="muted">This run stores no per-objective columns.</p>;
  return (
    <>
      <p className="stats-note">Feasible samples. Each f is normalised by its scale, as stored.</p>
      <PlotlyChart build={build} ariaLabel="Box plot of each objective" filename="opt_stats_objectives" />
      <div className="table-scroll">
        <table className="ruled stats-table" aria-label="Objective statistics">
          <thead>
            <tr>
              <th scope="col">Objective</th>
              <th scope="col" className="num">
                Mean ± std
              </th>
              <th scope="col" className="num">
                Min
              </th>
              <th scope="col" className="num">
                Median
              </th>
              <th scope="col" className="num">
                Best design
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <th scope="row">{r.name.replace(/^f_/, "")}</th>
                <td className="num">{fmtPm(r.mean, r.std)}</td>
                <td className="num">{fmt(r.min)}</td>
                <td className="num">{fmt(r.median)}</td>
                <td className="num">{fmt(r.bestDesign)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ParamsPanel({ d }: { d: StatsData }) {
  const rows = useMemo(() => paramStats(d), [d]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, {
        xTitle: "Position in explored range (0 low, 1 high)",
        yTitle: "",
        legend: true,
        yZero: false,
        xZero: false,
      });
      return {
        data: paramTraces(rows, colorsOf(t)),
        layout: {
          ...base,
          margin: { ...(base["margin"] as PlotlyObject), l: 190 },
          xaxis: { ...(base["xaxis"] as PlotlyObject), range: [-0.05, 1.05] },
          yaxis: {
            ...(base["yaxis"] as PlotlyObject),
            tickmode: "array",
            tickvals: rows.map((_, i) => i),
            ticktext: rows.map((r) => r.name),
            range: [rows.length - 0.5, -0.5],
          },
        },
      };
    },
    [rows],
  );
  const dec = d.paramUnit === "mm" ? 1 : 3;
  if (rows.length === 0) return <p className="muted">No design parameter varied.</p>;
  return (
    <>
      <p className="stats-note">
        Values in {d.paramUnit}. Top 10 % is the best tenth of the distinct feasible designs by f_total (at least 5). The chart scales each parameter to
        the range explored.
      </p>
      <PlotlyChart
        build={build}
        ariaLabel="Design parameters of all feasible and of the best samples"
        filename="opt_stats_params"
        height={Math.max(320, rows.length * 34 + 110)}
      />
      <div className="table-scroll">
        <table className="ruled stats-table" aria-label="Design parameter statistics">
          <thead>
            <tr>
              <th scope="col">Parameter</th>
              <th scope="col" className="num">
                All feasible, mean ± std
              </th>
              <th scope="col" className="num">
                Top 10 %, mean ± std
              </th>
              <th scope="col" className="num">
                Best design
              </th>
              <th scope="col" className="num">
                Baseline
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <th scope="row">{r.name}</th>
                <td className="num">{fmtPm(r.all.mean, r.all.std, dec)}</td>
                <td className="num">{r.top ? fmtPm(r.top.mean, r.top.std, dec) : "-"}</td>
                <td className="num">{fmt(r.best, dec)}</td>
                <td className="num">{fmt(r.baseline, dec)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ConstraintsPanel({ d }: { d: StatsData }) {
  const rows = useMemo(() => constraintStats(d), [d]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, { xTitle: "Pass rate [%]", yTitle: "", legend: false, yZero: false, xZero: false, hover: "closest" });
      return {
        data: [passRateTraces(rows, colorsOf(t))],
        layout: {
          ...base,
          margin: { ...(base["margin"] as PlotlyObject), l: 150 },
          xaxis: { ...(base["xaxis"] as PlotlyObject), range: [0, 100] },
          yaxis: { ...(base["yaxis"] as PlotlyObject), autorange: "reversed" },
        },
      };
    },
    [rows],
  );
  if (rows.length === 0) return <p className="muted">This run stores no constraint columns.</p>;
  return (
    <>
      <p className="stats-note">Pass rate is the share of ok samples with g at or below 0.</p>
      <PlotlyChart
        build={build}
        ariaLabel="Pass rate of each constraint"
        filename="opt_stats_constraints"
        height={Math.max(260, rows.length * 28 + 100)}
      />
      <div className="table-scroll">
        <table className="ruled stats-table" aria-label="Constraint statistics">
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Constraint
              </th>
              <th scope="colgroup" colSpan={3} className="num">
                All ok samples
              </th>
              <th scope="colgroup" colSpan={3} className="num">
                Top 10 %
              </th>
            </tr>
            <tr>
              {["Pass %", "Mean g", "Max g", "Pass %", "Mean g", "Max g"].map((h, i) => (
                <th key={i} scope="col" className="num">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <th scope="row">{r.name.replace(/^g_/, "")}</th>
                <td className="num">{fmt(r.all.passRate, 1)}</td>
                <td className="num">{fmt(r.all.meanG)}</td>
                <td className="num">{fmt(r.all.maxG)}</td>
                <td className="num">{fmt(r.top?.passRate ?? null, 1)}</td>
                <td className="num">{fmt(r.top?.meanG ?? null)}</td>
                <td className="num">{fmt(r.top?.maxG ?? null)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function EvolutionPanel({ d }: { d: StatsData }) {
  const bands = useMemo(() => evolutionBands(d), [d]);
  const build = useCallback(
    (theme: Theme) => {
      const t = readTokens(theme);
      const base = chartLayout(t, { xTitle: "Evaluation", yTitle: "f_total", legend: true, yZero: false, xZero: false, hover: "closest" });
      return { data: evolutionTraces(bands, d.evalId, d.best, colorsOf(t)), layout: base };
    },
    [bands, d],
  );
  if (bands.length === 0) return <p className="muted">{NO_FEASIBLE}</p>;
  return (
    <>
      <p className="stats-note">Mean and standard deviation of the feasible samples in windows of {windowSize(d.n)} evaluations.</p>
      <PlotlyChart build={build} ariaLabel="Evolution of f_total over the evaluations" filename="opt_stats_evolution" />
    </>
  );
}

/** "de" reads "DE only", "slsqp" reads "SLSQP only": phase names are short acronyms. */
export const phaseLabel = (p: string): string => `${/^[a-z]{2,6}$/.test(p) ? p.toUpperCase() : p} only`;

/** Statistics of one optimizer history: how all samples scored and where good designs concentrate. */
export function OptimizationStats({ history }: { history: HistoryData }) {
  const all = useMemo(() => mapStats(history), [history]);
  const phases = useMemo(() => phasesOf(all), [all]);
  // a run with several phases starts on its first one, the global search (gradient phases add
  // many near-identical probes that would dominate every statistic); undefined = not chosen yet
  const [chosen, setChosen] = useState<string | null | undefined>(undefined);
  const phase = chosen !== undefined && (chosen === null || phases.includes(chosen)) ? chosen : defaultPhase(all);
  const d = useMemo(() => filterPhase(all, phase), [all, phase]);
  const anyFeasible = useMemo(() => summarize(d).feasibleCount > 0, [d]);
  const best = bestRow(d);
  if (all.n === 0) return <p className="muted">The history is empty.</p>;
  return (
    <>
      {phases.length > 1 && (
        <div className="toolbar stats-samples">
          <span className="muted">Samples</span>
          <Segmented<string>
            label="Samples"
            value={phase ?? "all"}
            options={[{ value: "all", label: "All phases" }, ...phases.map((p) => ({ value: p, label: phaseLabel(p) }))]}
            onChange={(v) => setChosen(v === "all" ? null : v)}
          />
          <span className="muted" role="status">{`${d.n} of ${all.n} evaluations`}</span>
        </div>
      )}
      <p className="stats-note">
        Failed evaluations (status not ok, or f_total at the {F_FAIL} penalty) are counted but left out of every mean, quantile and axis.
        Standard deviations use n - 1. Top 10 % counts each distinct design once, so finite-difference probes do not fill it.
        f is the normalised objective as stored
        {d.mode === "pareto"
          ? "; in pareto mode f_total is only the stored sum of the objectives and the optimizer ranks by the objective vector."
          : "; f_total is the weighted sum of the objectives."}
        {best !== null ? ` Best design: evaluation ${d.evalId[best]}.` : ""}
      </p>
      {d.n === 0 && <p className="muted">No evaluations in this phase. Choose another phase.</p>}
      {d.n > 0 && !anyFeasible && (
        <p className="error" role="status">
          {NO_FEASIBLE}
        </p>
      )}
      {d.n > 0 && (
        <div className="panel-grid">
          <section className="panel" aria-label="Summary">
            <h3 className="panel-title">Summary</h3>
            <SummaryTable d={d} />
          </section>
          <section className="panel" aria-label="Score distribution">
            <h3 className="panel-title">Score distribution</h3>
            <ScoreChart d={d} />
          </section>
          <section className="panel" aria-label="Objectives">
            <h3 className="panel-title">Objectives</h3>
            {anyFeasible ? <ObjectivesPanel d={d} /> : <p className="muted">{NO_FEASIBLE}</p>}
          </section>
          <section className="panel" aria-label="Design parameters">
            <h3 className="panel-title">Design parameters</h3>
            {anyFeasible ? <ParamsPanel d={d} /> : <p className="muted">{NO_FEASIBLE}</p>}
          </section>
          <section className="panel" aria-label="Constraints">
            <h3 className="panel-title">Constraints</h3>
            <ConstraintsPanel d={d} />
          </section>
          <section className="panel" aria-label="Evolution">
            <h3 className="panel-title">Evolution</h3>
            <EvolutionPanel d={d} />
          </section>
        </div>
      )}
    </>
  );
}
