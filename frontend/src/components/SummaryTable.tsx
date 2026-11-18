import { useMemo } from "react";
import type { DatasetInfo, SpecDoc } from "../api";
import { useSummaries } from "../api/hooks";
import { formatDelta, formatValue, lookup, rowPresent, MINUS } from "../lib/format";
import { useUi, visibleIds } from "../state/store";

interface Props {
  spec: SpecDoc;
  datasets: DatasetInfo[];
}

/** Ruled table: one column per visible revision, Δ columns against the reference, unit column. */
export function SummaryTable({ spec, datasets }: Props) {
  const order = useUi((s) => s.order);
  const hidden = useUi((s) => s.hidden);
  const letters = useUi((s) => s.letters);
  const reference = useUi((s) => s.reference);

  const byId = useMemo(() => new Map(datasets.map((d) => [d.id, d])), [datasets]);
  const visible = useMemo(() => visibleIds({ order, hidden }).filter((id) => byId.has(id)), [order, hidden, byId]);
  const refId = reference && byId.has(reference) ? reference : null;
  const ids = useMemo(() => (refId && !visible.includes(refId) ? [...visible, refId] : visible), [visible, refId]);

  const results = useSummaries(ids);
  const summaries = new Map(ids.map((id, i) => [id, results[i]?.data] as const));
  const loading = results.some((r) => r.isPending);
  const failed = results.find((r) => r.isError);
  const deltas = visible.filter((id) => id !== refId);
  const letter = (id: string) => letters[id] ?? "?";

  if (visible.length === 0) return <p className="muted">Show at least one revision to see its summary.</p>;
  if (failed) {
    return (
      <p className="error" role="alert">
        Could not load the summary: {failed.error?.message ?? "unknown error"}
      </p>
    );
  }
  if (loading) return <p className="muted">Loading summary.</p>;

  const rows = spec.summary_rows.filter((r) => rowPresent(r, [...summaries.values()]));

  return (
    <div className="table-scroll">
      <table className="ruled summary">
        <caption className="sr-only">Summary values per revision</caption>
        <thead>
          <tr>
            <th scope="col">Quantity</th>
            <th scope="col">Unit</th>
            {visible.map((id) => (
              <th scope="col" key={id} className="num">
                {letter(id)}
                <span className="th-sub">{byId.get(id)?.label}</span>
              </th>
            ))}
            {refId &&
              deltas.map((id) => (
                <th scope="col" key={`d${id}`} className="num">
                  {`Δ ${letter(id)} ${MINUS} ${letter(refId)}`}
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.path.join(".")}>
              <th scope="row">{row.label}</th>
              <td className="unit">{row.unit}</td>
              {visible.map((id) => (
                <td key={id} className="num">
                  {formatValue(lookup(summaries.get(id), row.path), row.factor, row.decimals)}
                </td>
              ))}
              {refId &&
                deltas.map((id) => (
                  <td key={`d${id}`} className="num">
                    {formatDelta(
                      lookup(summaries.get(id), row.path),
                      lookup(summaries.get(refId), row.path),
                      row.factor,
                      row.decimals,
                    )}
                  </td>
                ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
