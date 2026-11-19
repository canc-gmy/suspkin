import { useEffect, useId, useState } from "react";
import { parseX, type ReadoutTable } from "../lib/readout";
import { Swatch } from "./TitleBlock";

interface Props {
  /** Heading of the cursor x ("Wheel travel"). */
  xLabel: string;
  xUnit: string;
  x: number | null;
  /** Text of an x value in display units. */
  formatX: (x: number) => string;
  pinned: boolean;
  table: ReadoutTable;
  showPoints: boolean;
  onShowPoints: (show: boolean) => void;
  /** Pin the cursor at a typed x (display units). */
  onPinX: (x: number) => void;
}

/** Delay (ms) before a pinned change is announced, so an arrow held down is read once. */
const ANNOUNCE_MS = 500;

/** One spoken line for the whole table (used for the polite live region). */
function spokenText(p: Pick<Props, "xLabel" | "xUnit" | "x" | "table" | "formatX">): string {
  if (p.x === null) return "";
  const head = `${p.xLabel} ${p.formatX(p.x)} ${p.xUnit}`;
  const rows = p.table.rows.map(
    (r) => `${r.letter}: ${r.cells.map((c, i) => `${p.table.columns[i]?.title ?? ""} ${c.spoken}`).join("; ")}`,
  );
  return [head, ...rows].join(". ");
}

/**
 * The readout strip: cursor x, an x input, and one row per revision with one column per panel.
 * It keeps its height whatever the cursor does, so the plots below never jump.
 */
export function Readout({ xLabel, xUnit, x, formatX, pinned, table, showPoints, onShowPoints, onPinX }: Props) {
  const inputId = useId();
  const statusId = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [announce, setAnnounce] = useState("");

  const spoken = spokenText({ xLabel, xUnit, x, table, formatX });
  useEffect(() => {
    if (!pinned) {
      setAnnounce("");
      return;
    }
    const id = setTimeout(() => setAnnounce(spoken), ANNOUNCE_MS);
    return () => clearTimeout(id);
  }, [pinned, spoken]);

  const shown = draft ?? (x === null ? "" : formatX(x));
  const commit = () => {
    const v = parseX(draft ?? "");
    if (v === null) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    setDraft(null);
    onPinX(v);
  };

  let status = "Move over a plot to read values";
  if (invalid) status = `Enter a number in ${xUnit || "display units"}`;
  else if (pinned) status = "Pinned, Esc releases";
  else if (x !== null) status = "Click a plot to pin";

  return (
    <div className="readout" role="group" aria-label="Readout">
      <div className="readout-head">
        <label className="readout-x" htmlFor={inputId}>
          {xLabel}
        </label>
        <input
          id={inputId}
          type="text"
          inputMode="decimal"
          className="readout-input"
          value={shown}
          placeholder="-"
          aria-label={xUnit ? `${xLabel} [${xUnit}]` : xLabel}
          aria-describedby={statusId}
          aria-invalid={invalid}
          onChange={(e) => {
            setDraft(e.target.value);
            setInvalid(false);
          }}
          onBlur={() => {
            setDraft(null);
            setInvalid(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commit();
            }
          }}
        />
        <span className="readout-unit">{xUnit}</span>
        <span id={statusId} className={invalid ? "readout-status error" : "readout-status muted"} role={invalid ? "alert" : undefined}>{status}</span>
        <span className="spacer" />
        <label className="check">
          <input type="checkbox" aria-label="Show solved points" checked={showPoints} onChange={(e) => onShowPoints(e.target.checked)} />
          Show solved points
        </label>
      </div>
      <div className="readout-scroll">
        <table className="readout-table">
          <thead>
            <tr>
              <th scope="col" className="readout-rev">
                <span className="sr-only">Revision</span>
              </th>
              {table.columns.map((c) => (
                <th key={c.panelId} scope="col" className="num">
                  {c.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r) => (
              <tr key={r.revId}>
                <th scope="row" className="readout-rev">
                  <span className="rev-letter">{r.letter}</span>
                  <Swatch color={r.color} />
                </th>
                {r.cells.map((c, i) => (
                  <td key={table.columns[i]?.panelId ?? i} className="num">
                    {c.parts.map((p, k) => (
                      <span key={k} className="readout-part">
                        {p.prefix && <span className="pfx">{p.prefix}</span>}
                        {p.text || " "}
                      </span>
                    ))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {announce}
      </div>
    </div>
  );
}
