import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { DatasetInfo } from "../api";
import { useCloseDataset, usePatchDataset } from "../api/hooks";
import { useUi } from "../state/store";
import { revisionColor, type Theme } from "../theme/revisions";

interface Props {
  datasets: DatasetInfo[];
  palette: string[];
  theme: Theme;
}

/** Drawn line sample in the revision colour (solid = left / front). */
function Swatch({ color }: { color: string }) {
  return (
    <svg className="swatch" width="28" height="10" viewBox="0 0 28 10" aria-hidden="true">
      <line x1="1" y1="5" x2="27" y2="5" stroke={color} strokeWidth="2.25" strokeLinecap="butt" />
    </svg>
  );
}

function Row({ d, color, error, setError }: { d: DatasetInfo; color: string; error?: string; setError: (m: string | null) => void }) {
  const letter = useUi((s) => s.letters[d.id] ?? "?");
  const visible = useUi((s) => !s.hidden[d.id]);
  const isRef = useUi((s) => s.reference === d.id);
  const fresh = useUi((s) => !!s.fresh[d.id]);
  const toggle = useUi((s) => s.toggleVisible);
  const setReference = useUi((s) => s.setReference);
  const clearFresh = useUi((s) => s.clearFresh);
  const patch = usePatchDataset();
  const close = useCloseDataset();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(d.label);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  const commit = () => {
    const label = draft.trim();
    setEditing(false);
    if (!label || label === d.label) return;
    patch.mutate(
      { id: d.id, patch: { label } },
      { onError: (e) => setError(`Could not rename ${d.label}: ${e.message}`), onSuccess: () => setError(null) },
    );
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") {
      setDraft(d.label);
      setEditing(false);
    }
  };

  return (
    <>
      <tr
        className={[visible ? "" : "off", fresh ? "fresh" : "", isRef ? "is-ref" : ""].join(" ").trim()}
        onClick={() => toggle(d.id)}
        onAnimationEnd={() => clearFresh(d.id)}
      >
        <td className="rev">
          <button
            type="button"
            className="rev-btn"
            aria-pressed={visible}
            aria-label={`${visible ? "Hide" : "Show"} revision ${letter}`}
            onClick={(e) => {
              e.stopPropagation();
              toggle(d.id);
            }}
          >
            <span className="rev-letter">{letter}</span>
            <Swatch color={color} />
          </button>
        </td>
        <td className="design" onClick={(e) => editing && e.stopPropagation()}>
          {editing ? (
            <input
              ref={input}
              className="inline-edit"
              aria-label={`Name of revision ${letter}`}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
              onBlur={commit}
            />
          ) : (
            <span className="design-name">
              <span className="design-label" title={d.source}>
                {d.label}
              </span>
              <button
                type="button"
                className="link-btn"
                aria-label={`Rename revision ${letter}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setDraft(d.label);
                  setEditing(true);
                }}
              >
                Rename
              </button>
              <button
                type="button"
                className="link-btn"
                aria-label={`Close revision ${letter}`}
                onClick={(e) => {
                  e.stopPropagation();
                  close.mutate(d.id, { onError: (err) => setError(`Could not close ${d.label}: ${err.message}`) });
                }}
              >
                Close
              </button>
            </span>
          )}
        </td>
        <td className="ref" onClick={(e) => e.stopPropagation()}>
          <input
            type="radio"
            name="reference"
            checked={isRef}
            onChange={() => setReference(d.id)}
            aria-label={`Use revision ${letter} as the reference`}
          />
        </td>
      </tr>
      {error && (
        <tr>
          <td colSpan={3} className="error" role="alert">
            {error}
          </td>
        </tr>
      )}
    </>
  );
}

/** The drawing title block: a ruled table of revisions (letter, design, reference). */
export function TitleBlock({ datasets, palette, theme }: Props) {
  const setOpenDialog = useUi((s) => s.setOpenDialog);
  const [errors, setErrors] = useState<Record<string, string | undefined>>({});

  return (
    <div className="titleblock-inner">
      <table className="ruled titleblock-table">
        <caption className="sr-only">Revisions</caption>
        <thead>
          <tr>
            <th scope="col">Rev</th>
            <th scope="col">Design</th>
            <th scope="col" className="ref">
              Ref
            </th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((d) => (
            <Row
              key={d.id}
              d={d}
              color={revisionColor(d.color, palette, theme)}
              error={errors[d.id]}
              setError={(m) => setErrors((e) => ({ ...e, [d.id]: m ?? undefined }))}
            />
          ))}
        </tbody>
      </table>
      <button type="button" className="btn secondary" onClick={() => setOpenDialog(true)}>
        Open files
      </button>
      <p className="legend-note">Solid left/front, dashed right/rear</p>
    </div>
  );
}
