import { useEffect, useMemo, useReducer, useState, type ReactNode } from "react";
import type { DatasetInfo } from "../api";
import { useHardpoints, useSource } from "../api/hooks";
import { READ_ONLY_TIP } from "../api/snapshotSource";
import { Segmented } from "../components/Segmented";
import {
  cellChanged,
  changedPoints,
  editReducer,
  effectiveMm,
  isRightOnly,
  yamlFileName,
  type EditState,
} from "../lib/hardpointEdit";
import { buildHardpointRows, filterRows, pointKey, type HardpointRow, type Vec3 } from "../lib/hardpoints";
import { fixed, MINUS, MISSING } from "../lib/format";
import { signed } from "../lib/readout";
import { useUi, visibleIds } from "../state/store";
import { useHardpointSolver, useSaveYaml } from "./useHardpointSolver";

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
  const readOnly = useSource().readOnly;
  const [show, setShow] = useState<Show>("all");
  const [text, setText] = useState("");
  const [edit, dispatch] = useReducer(editReducer, null);
  const [pick, setPick] = useState<string | null>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveId, setSaveId] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");

  const byId = useMemo(() => new Map(datasets.map((d) => [d.id, d])), [datasets]);
  const visible = useMemo(() => visibleIds({ order, hidden }).filter((id) => byId.has(id)), [order, hidden, byId]);
  const refId = reference && byId.has(reference) ? reference : null;
  const editId = edit && byId.has(edit.sourceId) ? edit.sourceId : null;
  // the revision being edited is always a column, even when hidden in the title block
  const shown = useMemo(() => (editId && !visible.includes(editId) ? [...visible, editId] : visible), [visible, editId]);
  const ids = useMemo(() => (refId && !shown.includes(refId) ? [...shown, refId] : shown), [shown, refId]);
  const results = useHardpoints(ids);
  // keep the table (and the focus inside it) while a new revision loads
  const loading = results.length > 0 && results.every((r) => r.isPending);
  const sig = results.map((r) => r.dataUpdatedAt).join(",");

  const all = useMemo(() => {
    const points = Object.fromEntries(ids.map((id, i) => [id, results[i]?.data ?? null]));
    return buildHardpointRows(points, ids, refId);
    // `sig` changes exactly when a result arrives
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, refId, sig]);
  const rows = useMemo(() => filterRows(all, { differingOnly: show === "differing", text }), [all, show, text]);

  const sourcePoints = editId ? (results[ids.indexOf(editId)]?.data ?? null) : null;
  const solver = useHardpointSolver(editId ? edit : null, sourcePoints, dispatch);
  const base = solver.base;
  const editing: EditState | null = editId && base ? edit : null;

  // the edited revision was closed: leave edit mode
  useEffect(() => {
    if (edit && !byId.has(edit.sourceId)) dispatch({ type: "discard" });
  }, [edit, byId]);

  const pickable = shown;
  const pickId = pick && pickable.includes(pick) ? pick : refId && pickable.includes(refId) ? refId : (pickable[0] ?? null);
  const edited = shown.filter((id) => byId.get(id)?.kind === "edited");
  const saveTarget = saveId && edited.includes(saveId) ? saveId : (edited[edited.length - 1] ?? null);
  const saver = useSaveYaml();

  const letter = (id: string) => letters[id] ?? "?";
  const deltas = shown.filter((id) => id !== refId);
  const label = (id: string) => `${letter(id)} ${byId.get(id)?.label ?? ""}`.trim();

  if (shown.length === 0) return <p className="muted">Show at least one revision to see its hardpoints.</p>;
  if (loading) return <p className="muted">Loading hardpoints.</p>;

  const groups = (["front", "rear"] as const)
    .map((axle) => ({ axle, rows: rows.filter((r) => r.axle === axle) }))
    .filter((g) => g.rows.length > 0);
  const colSpan = 2 + shown.length * 3 + (refId ? deltas.length : 0) + (editing ? 1 : 0);

  const plain = (v: Vec3 | null | undefined, id: string, a: string, i: number) => (
    <td key={`${id}${a}`} className={i === 0 ? "num group-start" : "num"}>
      {v ? fixed(v[i] ?? null, 1) : MISSING}
    </td>
  );

  const cells = (r: HardpointRow, id: string) => {
    const v = r.values[id];
    if (!editing || !base || id !== editing.sourceId || !v) return AXES.map((a, i) => plain(v, id, a, i));
    const pt = { axle: r.axle, point: r.point };
    const now = effectiveMm(editing, base, pt, r.side);
    const live = r.side === "left" || isRightOnly(editing, base, pt);
    return AXES.map((a, i) => {
      const changed = cellChanged(editing, base, pt, r.side, a);
      const cls = ["num", i === 0 ? "group-start" : "", changed ? "cell-changed" : ""].filter(Boolean).join(" ");
      return (
        <td key={`${id}${a}`} className={cls}>
          {live ? (
            <CellInput
              value={now?.[i] ?? v[i] ?? 0}
              label={`${r.point} ${r.side} ${a} in mm`}
              onValue={(mm) => dispatch({ type: "set", axle: r.axle, point: r.point, side: r.side, axis: a, mm, base: v })}
            />
          ) : (
            <span className="muted">{fixed(now?.[i] ?? null, 1)}</span>
          )}
        </td>
      );
    });
  };

  const pointEdited = (r: HardpointRow) =>
    !!editing &&
    (!!editing.cells[pointKey({ axle: r.axle, point: r.point, side: "left" })] ||
      !!editing.cells[pointKey({ axle: r.axle, point: r.point, side: "right" })] ||
      editing.rightOnly[`${r.axle}|${r.point}`] !== undefined);

  const unavailable = readOnly ? READ_ONLY_TIP : undefined;
  const nChanged = changedPoints(editing, base ?? new Map());

  const openSave = () => {
    if (!saveTarget) return;
    setSaveId(saveTarget);
    setFileName(yamlFileName(byId.get(saveTarget)?.label ?? ""));
    saver.clear();
    setSaveOpen(true);
  };
  const doSave = async () => {
    if (saveTarget && fileName.trim() && (await saver.save(saveTarget, fileName.trim()))) setSaveOpen(false);
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
      <div className="toolbar edit-bar">
        {editing ? (
          <>
            <span className="edit-title">{`Editing ${label(editing.sourceId)}`}</span>
            <button type="button" className="btn primary" disabled={solver.edits.length === 0 || solver.solving} onClick={solver.solve}>
              Solve
            </button>
            <button type="button" className="btn secondary" onClick={solver.discard}>
              Discard edits
            </button>
            <label className="check">
              <input type="checkbox" checked={solver.auto} onChange={(e) => solver.setAuto(e.target.checked)} />
              <span>Auto-solve</span>
            </label>
            <span className="muted" role="status">
              {solver.solving ? "Solving…" : (solver.status ?? `${nChanged} ${nChanged === 1 ? "point" : "points"} changed`)}
            </span>
          </>
        ) : (
          <>
            <label className="field">
              <span>Edit revision</span>
              <select value={pickId ?? ""} disabled={readOnly || pickable.length === 0} onChange={(e) => setPick(e.target.value)}>
                {pickable.map((id) => (
                  <option key={id} value={id}>
                    {label(id)}
                  </option>
                ))}
              </select>
            </label>
            <span title={unavailable}>
              <button
                type="button"
                className="btn secondary"
                disabled={readOnly || !pickId}
                onClick={() => pickId && dispatch({ type: "start", sourceId: pickId })}
              >
                Edit as new revision
              </button>
            </span>
            {solver.status && (
              <span className="muted" role="status">
                {solver.status}
              </span>
            )}
          </>
        )}
        {edited.length > 0 && (
          <span title={unavailable}>
            <button type="button" className="link-btn" disabled={readOnly} onClick={openSave}>
              Save as YAML
            </button>
          </span>
        )}
        {saver.saved && (
          <span className="muted" role="status">
            {saver.saved}
          </span>
        )}
      </div>
      {editing && (
        <p className="panel-note">
          A left point also moves its right twin unless &ldquo;Right side only&rdquo; is ticked on the right row.
        </p>
      )}
      {saveOpen && saveTarget && (
        <form
          className="toolbar save-line"
          onSubmit={(e) => {
            e.preventDefault();
            void doSave();
          }}
        >
          {edited.length > 1 && (
            <label className="field">
              <span>Revision</span>
              <select
                value={saveTarget}
                onChange={(e) => {
                  setSaveId(e.target.value);
                  setFileName(yamlFileName(byId.get(e.target.value)?.label ?? ""));
                }}
              >
                {edited.map((id) => (
                  <option key={id} value={id}>
                    {label(id)}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="field">
            <span>File name</span>
            <input type="text" value={fileName} onChange={(e) => setFileName(e.target.value)} />
          </label>
          <button type="submit" className="btn secondary" disabled={!fileName.trim()}>
            Save
          </button>
          <button type="button" className="btn secondary" onClick={() => setSaveOpen(false)}>
            Cancel
          </button>
        </form>
      )}
      {saver.error && (
        <p className="error" role="alert">
          {saver.error}
        </p>
      )}
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
                {shown.map((id) => (
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
                {editing && (
                  <th scope="col" rowSpan={2} className="hp-reset">
                    <span className="sr-only">Edit actions</span>
                  </th>
                )}
              </tr>
              <tr>
                {shown.flatMap((id) =>
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
                        {editing && base && r.side === "right" && (
                          <label className="check hp-rightonly">
                            <input
                              type="checkbox"
                              checked={isRightOnly(editing, base, r)}
                              onChange={(e) => dispatch({ type: "rightOnly", axle: r.axle, point: r.point, on: e.target.checked })}
                            />
                            <span>Right side only</span>
                          </label>
                        )}
                      </td>
                      {shown.map((id) => cells(r, id))}
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
                      {editing && (
                        <td className="hp-reset">
                          {pointEdited(r) && (
                            <button
                              type="button"
                              className="link-btn"
                              aria-label={`Reset point ${r.point}`}
                              onClick={() => dispatch({ type: "reset", axle: r.axle, point: r.point })}
                            >
                              Reset point
                            </button>
                          )}
                        </td>
                      )}
                    </tr>
                  ))}
                </GroupRows>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {solver.error && (
        <p className="error" role="alert">
          {solver.error}
        </p>
      )}
    </div>
  );
}

/** A mm input that keeps what the user types until it loses focus. */
function CellInput({ value, label, onValue }: { value: number; label: string; onValue: (mm: number | null) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="number"
      className="cell-input"
      step={0.1}
      aria-label={label}
      value={draft ?? value.toFixed(1)}
      onChange={(e) => {
        const raw = e.target.value;
        setDraft(raw);
        const n = Number(raw);
        if (raw.trim() === "") onValue(null);
        else if (Number.isFinite(n)) onValue(n);
      }}
      onBlur={() => setDraft(null)}
    />
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
