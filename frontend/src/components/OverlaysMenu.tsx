import { Disclosure } from "./Disclosure";
import { NO_STUDY_HINT, type RegionSource } from "../lib/regionSource";
import { toggleCentreKind, toggleRegionRole } from "../lib/centresMenu";
import {
  CENTRE_KINDS,
  CENTRE_LABELS,
  REGION_LABELS,
  REGION_ROLES,
  type CentreKind,
  type RegionRole,
} from "../theme/scene3d";

const HINT_ID = "regions-hint";

/**
 * "Overlays" toggle button with a small panel: one checkbox per kind of rotation centre, and one
 * per kind of study region. The regions come from a study picked independently of the selected
 * revision; with no open study the region boxes are disabled and a hint says how to get one.
 */
export function OverlaysMenu({
  centres,
  onCentres,
  regions,
  onRegions,
  sources,
  source,
  onSource,
  onOpenFiles,
}: {
  centres: ReadonlySet<CentreKind>;
  onCentres: (next: Set<CentreKind>) => void;
  regions: ReadonlySet<RegionRole>;
  onRegions: (next: Set<RegionRole>) => void;
  /** Open revisions that have a study. */
  sources: readonly RegionSource[];
  /** The study whose regions are drawn (null when `sources` is empty). */
  source: string | null;
  onSource: (id: string) => void;
  onOpenFiles: () => void;
}) {
  const hasStudy = sources.length > 0;
  const count = centres.size + (hasStudy ? regions.size : 0);

  return (
    <Disclosure label={count > 0 ? `Overlays (${count})` : "Overlays"}>
      {() => (
        <>
          <div role="group" aria-label="Rotation centres" className="centres-group">
            <span className="centres-heading">Rotation centres</span>
            {CENTRE_KINDS.map((k) => (
              <label key={k} className="field">
                <input
                  type="checkbox"
                  aria-label={CENTRE_LABELS[k]}
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
          >
            <span className="centres-heading">Study regions</span>
            {hasStudy ? (
              <>
                <label className="field">
                  <span>Study</span>
                  <select aria-label="Study" value={source ?? ""} onChange={(e) => onSource(e.target.value)}>
                    {sources.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="muted">Shown as the study defines them, relative to its baseline car.</p>
              </>
            ) : (
              <p id={HINT_ID} className="muted">
                {NO_STUDY_HINT}{" "}
                <button type="button" className="link-btn" onClick={onOpenFiles}>
                  Open files
                </button>
              </p>
            )}
            {REGION_ROLES.map((k) => (
              <label key={k} className="field">
                <input
                  type="checkbox"
                  aria-label={REGION_LABELS[k]}
                  checked={hasStudy && regions.has(k)}
                  disabled={!hasStudy}
                  aria-describedby={hasStudy ? undefined : HINT_ID}
                  onChange={(e) => onRegions(toggleRegionRole(regions, k, e.target.checked))}
                />
                <span>{REGION_LABELS[k]}</span>
              </label>
            ))}
          </div>
        </>
      )}
    </Disclosure>
  );
}
