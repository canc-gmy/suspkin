import { useMemo, useState, type ReactNode } from "react";
import type { DatasetInfo } from "../api";
import { useHardpoints } from "../api/hooks";
import { Segmented } from "../components/Segmented";
import { buildHardpointRows, filterRows, type HardpointRow, type Vec3 } from "../lib/hardpoints";
import { fixed, MINUS, MISSING } from "../lib/format";
import { signed } from "../lib/readout";
import { useUi, visibleIds } from "../state/store";

type Show = "all" | "differing";

const AXLE_TITLE = { front: "Front axle", rear: "Rear axle" } as const;
const AXES = ["x", "y", "z"] as const;

/** Per-axis delta line: "x +0.00 y −2.00 z +0.00". */
const axisText = (d: Vec3) => AXES.map((a, i) => `${a} ${signed(d[i] ?? null, 2)}`).join(" ");

/** Every hardpoint of every visible revision in mm, with the delta to the reference. */
export function HardpointsView({ datasets }: { datasets: DatasetInfo[] }) {
  const order = useUi((s) => s.order);
  const hidden = useUi((s) => s.hidden);
  const letters = useUi((s) => s.letters);
  const reference = useUi((s) => s.reference);
  const [show, setShow] = useState<Show>("all");
  const [text, setText] = useState("");

  const byId = useMemo(() => new Map(datasets.map((d) => [d.id, d])), [datasets]);
  const visible = useMemo(() => visibleIds({ order, hidden }).filter((id) => byId.has(id)), [order, hidden, byId]);
  const refId = reference && byId.has(reference) ? reference : null;
  const ids = useMemo(() => (refId && !visible.includes(refId) ? [...visible, refId] : visible), [visible, refId]);
  const results = useHardpoints(ids);
  const loading = results.some((r) => r.isPending);
  const sig = results.map((r) => r.dataUpdatedAt).join(",");

  const all = useMemo(() => {
    const points = Object.fromEntries(ids.map((id, i) => [id, results[i]?.data ?? null]));
    return buildHardpointRows(points, ids, refId);
    // `sig` changes exactly when a result arrives
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, refId, sig]);
  const rows = useMemo(() => filterRows(all, { differingOnly: show === "differing", text }), [all, show, text]);

  const letter = (id: string) => letters[id] ?? "?";
  const deltas = visible.filter((id) => id !== refId);

  if (visible.length === 0) return <p className="muted">Show at least one revision to see its hardpoints.</p>;
  if (loading) return <p className="muted">Loading hardpoints.</p>;

  const groups = (["front", "rear"] as const)
    .map((axle) => ({ axle, rows: rows.filter((r) => r.axle === axle) }))
    .filter((g) => g.rows.length > 0);
  const colSpan = 2 + visible.length * 3 + (refId ? deltas.length : 0);

  const cells = (r: HardpointRow, id: string) => {
    const v = r.values[id];
    return AXES.map((a, i) => (
      <td key={`${id}${a}`} className={i === 0 ? "num group-start" : "num"}>
        {v ? fixed(v[i] ?? null, 1) : MISSING}
      </td>
    ));
  };

  return (
    <div className="hardpoints">
      <div className="toolbar hardpoints-bar">
        <Segmented<Show>
          label="Rows"
          value={show}
          onChange={setShow}
          options={[
            { value: "all", label: "All" },
            { value: "differing", label: "Differing only" },
          ]}
        />
        <label className="field">
          <span>Filter points</span>
          <input type="text" value={text} placeholder="uca_outer" onChange={(e) => setText(e.target.value)} />
        </label>
        <span className="muted" role="status">{`${rows.length} of ${all.length} rows`}</span>
      </div>
      {rows.length === 0 ? (
        <p className="muted">No hardpoints match. Clear the filter or choose All.</p>
      ) : (
        <div className="table-scroll">
          <table className="ruled summary hardpoint-table">
            <caption className="sr-only">Hardpoint positions in mm per revision, with the difference to the reference</caption>
            <thead>
              <tr>
                <th scope="col" rowSpan={2} className="hp-point">
                  Point
                </th>
                <th scope="col" rowSpan={2} className="hp-side">
                  Side
                </th>
                {visible.map((id) => (
                  <th scope="colgroup" colSpan={3} key={id} className="num group-start">
                    {letter(id)}
                    <span className="th-sub">{byId.get(id)?.label}</span>
                  </th>
                ))}
                {refId &&
                  deltas.map((id) => (
                    <th scope="col" rowSpan={2} key={`d${id}`} className="num group-start">
                      {`Δ ${letter(id)} ${MINUS} ${letter(refId)}`}
                      <span className="th-sub">|Δ| in mm</span>
                    </th>
                  ))}
              </tr>
              <tr>
                {visible.flatMap((id) =>
                  AXES.map((a, i) => (
                    <th scope="col" key={`${id}${a}`} className={i === 0 ? "num sub group-start" : "num sub"}>
                      {a}
                    </th>
                  )),
                )}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <GroupRows key={g.axle} title={AXLE_TITLE[g.axle]} colSpan={colSpan}>
                  {g.rows.map((r) => (
                    <tr key={r.key}>
                      <th scope="row" className="hp-point">
                        {r.point}
                      </th>
                      <td className="hp-side">
                        {r.side}
                        {r.overridden && <span className="th-sub">override</span>}
                      </td>
                      {visible.map((id) => cells(r, id))}
                      {refId &&
                        deltas.map((id) => {
                          const d = r.deltas[id];
                          return (
                            <td key={`d${id}`} className="num group-start">
                              {d ? fixed(d.mag, 2) : MISSING}
                              {d && <span className="th-sub">{axisText(d.d)}</span>}
                            </td>
                          );
                        })}
                    </tr>
                  ))}
                </GroupRows>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function GroupRows({ title, colSpan, children }: { title: string; colSpan: number; children: ReactNode }) {
  return (
    <>
      <tr className="group-row">
        <th scope="rowgroup" colSpan={colSpan}>
          {title}
        </th>
      </tr>
      {children}
    </>
  );
}
