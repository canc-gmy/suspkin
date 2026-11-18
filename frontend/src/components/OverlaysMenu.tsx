import { Disclosure } from "./Disclosure";
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
            title={hasStudy ? undefined : "No optimization study attached to this revision"}
          >
            <span className="centres-heading">Study regions</span>
            {REGION_ROLES.map((k) => (
              <label key={k} className="field">
                <input
                  type="checkbox"
                  aria-label={REGION_LABELS[k]}
                  checked={hasStudy && regions.has(k)}
                  disabled={!hasStudy}
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
