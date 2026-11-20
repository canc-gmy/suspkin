"""Optimization regions of a study as wireframes for the 3D view (SI, vehicle frame).

The geometry comes from ``suspkin.plotting.regions`` (the same helpers ``plot_study_3d`` uses),
so the web view and the static figure cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from suspkin.app.models import RegionOut
from suspkin.optimize.design import DesignSpace
from suspkin.optimize.spec import load_study
from suspkin.optimize.spec.criteria import InsideConstraint, OutsideConstraint
from suspkin.optimize.spec.study import region_centroid
from suspkin.optimize.spec.variables import BoxRegion, HexahedronRegion, OffsetRegion, SphereRegion
from suspkin.plotting.regions import (
    _domain_vertices,
    _hex_lines,
    _is_right,
    _mirror,
    _region_lines,
)

__all__ = ["is_study_data", "study_regions"]


def is_study_data(doc: Any) -> bool:
    """Light check of parsed YAML: ``schema`` plus ``variables`` or ``constraints``."""
    return (
        isinstance(doc, dict) and "schema" in doc and ("variables" in doc or "constraints" in doc)
    )


def _lines(lines: list[Any]) -> list[list[tuple[float, float, float]]]:
    return [[(float(p[0]), float(p[1]), float(p[2])) for p in np.asarray(ln)] for ln in lines]


def _shape(region: Any) -> str:
    if isinstance(region, BoxRegion):
        return "box"
    if isinstance(region, HexahedronRegion):
        return "hexahedron"
    if isinstance(region, SphereRegion):
        return "sphere"
    if isinstance(region, OffsetRegion):
        return "offset"
    return "cylinder"


def study_regions(study_path: Path) -> list[RegionOut]:
    """Domains and constraint regions of the study at ``study_path`` (raises StudyConfigError)."""
    study, vehicle = load_study(study_path)
    space = DesignSpace(study, vehicle)
    out: list[RegionOut] = []
    for var in study.variables:
        verts = _domain_vertices(space, var)
        shape = _shape(var.domain)
        side = "right" if _is_right(var) else "left"
        out.append(
            RegionOut(
                name=var.point,
                role="domain",
                point=var.point,
                side=side,
                shape=shape,
                lines=_lines(_hex_lines(verts)),
            )
        )
        if not var.asymmetric:
            out.append(
                RegionOut(
                    name=var.point,
                    role="domain",
                    point=var.point,
                    side="right",
                    shape=shape,
                    mirrored=True,
                    lines=_lines(_hex_lines(_mirror(verts))),
                )
            )
    for c in study.constraints:
        if not isinstance(c, InsideConstraint | OutsideConstraint):
            continue
        over = c.over
        out.append(
            RegionOut(
                name=c.name,
                role="inside" if isinstance(c, InsideConstraint) else "outside",
                point=c.point,
                side="right" if region_centroid(c.region)[1] < 0.0 else "left",
                shape=_shape(c.region),
                lines=_lines(_region_lines(c.region)),
                over=over,
                note=(
                    f"Checked over sweep {over}: the point moves with the wheel, the region "
                    "stays fixed in the body frame."
                    if over
                    else None
                ),
            )
        )
    return out
