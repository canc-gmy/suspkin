import { useEffect, useId, useRef, useState } from "react";
import { toggleCentreKind, toggleRegionRole } from "../lib/centresMenu";
import {
  CENTRE_KINDS,
  CENTRE_LABELS,
  REGION_LABELS,
  REGION_ROLES,
  type CentreKind,
  type RegionRole,
} from "../theme/scene3d";

/**
 * "Overlays" toggle button with a small panel: one checkbox per kind of rotation centre, and one
 * per kind of study region (disabled when the revision has no optimization study).
 */
export function OverlaysMenu({
  centres,
  onCentres,
  regions,
  onRegions,
  hasStudy,
}: {
  centres: ReadonlySet<CentreKind>;
  onCentres: (next: Set<CentreKind>) => void;
  regions: ReadonlySet<RegionRole>;
  onRegions: (next: Set<RegionRole>) => void;
  hasStudy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const count = centres.size + (hasStudy ? regions.size : 0);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      button.current?.focus();
    };
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  return (
    <div className="centres-menu" ref={root}>
      <button
        ref={button}
        type="button"
        className="btn secondary"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen(!open)}
      >
        {count > 0 ? `Overlays (${count})` : "Overlays"}
      </button>
      {open && (
        <div id={panelId} className="centres-panel">
          <div role="group" aria-label="Rotation centres" className="centres-group">
            <span className="centres-heading">Rotation centres</span>
            {CENTRE_KINDS.map((k) => (
              <label key={k} className="field">
                <input
                  type="checkbox"
                  checked={centres.has(k)}
                  onChange={(e) => onCentres(toggleCentreKind(centres, k, e.target.checked))}
                />
                <span>{CENTRE_LABELS[k]}</span>
              </label>
            ))}
          </div>
          <div
            role="group"
            aria-label="Study regions"
            className="centres-group"
            title={hasStudy ? undefined : "No optimization study attached to this revision"}
          >
            <span className="centres-heading">Study regions</span>
            {REGION_ROLES.map((k) => (
              <label key={k} className="field">
                <input
                  type="checkbox"
                  checked={hasStudy && regions.has(k)}
                  disabled={!hasStudy}
                  onChange={(e) => onRegions(toggleRegionRole(regions, k, e.target.checked))}
                />
                <span>{REGION_LABELS[k]}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
