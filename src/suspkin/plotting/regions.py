"""3D view of an optimisation study: vehicle, variable domains and constraint regions.

Reads the study spec and the design space only; the only kinematic work is the static pose
drawn by :func:`~suspkin.plotting.geometry3d.plot_vehicle_3d`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray

# optimize.spec is imported inside the functions: optimize -> plotting -> optimize otherwise
from suspkin.plotting.geometry3d import plot_vehicle_3d

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from suspkin.optimize.design import DesignSpace

F64 = NDArray[np.float64]

_GREEN = "#2ca02c"
_RED = "#d62728"
# hexahedron edges: bottom face, top face, verticals (vertex order: see optimize.regions)
_HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)  # fmt: skip
_BOX_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float64,
)


def _mirror(a: ArrayLike) -> F64:
    out = np.array(a, dtype=np.float64)
    out[..., 1] = -out[..., 1]
    return out


def _box_vertices(lo: ArrayLike, hi: ArrayLike) -> F64:
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    out: F64 = lo_a + _BOX_CORNERS * (hi_a - lo_a)
    return out


def _hex_lines(v: F64) -> list[F64]:
    return [np.vstack([v[a], v[b]]) for a, b in _HEX_EDGES]


def _circle(center: F64, e1: F64, e2: F64, radius: float, n: int = 48) -> F64:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    out: F64 = center + radius * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    return out


def _sphere_lines(center: ArrayLike, radius: float) -> list[F64]:
    c = np.asarray(center, dtype=np.float64)
    eye = np.eye(3)
    return [_circle(c, eye[i], eye[j], radius) for i, j in ((0, 1), (0, 2), (1, 2))]


def _cylinder_lines(p0: ArrayLike, p1: ArrayLike, radius: float) -> list[F64]:
    a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    axis = (b - a) / np.linalg.norm(b - a)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    lines = [_circle(a, e1, e2, radius), _circle(b, e1, e2, radius)]
    for e in (e1, e2, -e1, -e2):
        lines.append(np.vstack([a + radius * e, b + radius * e]))
    return lines


def _region_lines(region: Any) -> list[F64]:
    from suspkin.optimize.spec.variables import (
        BoxRegion,
        CylinderRegion,
        HexahedronRegion,
        SphereRegion,
    )

    if isinstance(region, BoxRegion):
        return _hex_lines(_box_vertices(region.lo, region.hi))
    if isinstance(region, HexahedronRegion):
        return _hex_lines(np.array(region.vertices, dtype=np.float64))
    if isinstance(region, SphereRegion):
        return _sphere_lines(region.center, region.radius)
    assert isinstance(region, CylinderRegion)
    return _cylinder_lines(region.p0, region.p1, region.radius)


def _baseline_point(space: DesignSpace, axle: str, name: str, right: bool) -> F64:
    ax = getattr(space.baseline, axle)
    left = np.array(getattr(ax.hardpoints, name), dtype=np.float64)
    if not right:
        return left
    ov = ax.right_overrides
    return np.array(ov[name], dtype=np.float64) if name in ov else _mirror(left)


def _is_right(var: Any) -> bool:
    return bool(var.asymmetric and var.effective_side == "right")


def _domain_vertices(space: DesignSpace, var: Any) -> F64:
    """The 8 vertices of a variable's domain (right-side coordinates for ``side: right``)."""
    from suspkin.optimize.spec.variables import (
        BoxRegion,
        HexahedronRegion,
        OffsetRegion,
        parse_point_ref,
    )

    d = var.domain
    if isinstance(d, HexahedronRegion):
        return np.array(d.vertices, dtype=np.float64)
    if isinstance(d, BoxRegion):
        return _box_vertices(d.lo, d.hi)
    assert isinstance(d, OffsetRegion)
    axle, name = parse_point_ref(var.point)
    base = _baseline_point(space, axle, name, _is_right(var))
    lo, hi = np.zeros(3), np.zeros(3)
    for i, rng in enumerate((d.x, d.y, d.z)):
        if rng is not None:
            lo[i], hi[i] = rng
    return _box_vertices(base + lo, base + hi)


def plot_study_3d(
    space: DesignSpace,
    x: ArrayLike | None = None,
    ax: Axes3D | None = None,
) -> Figure:
    """Draw the vehicle with the study's variable domains and constraint regions.

    Variable domains are green wireframes (symmetric ones also mirrored to the right side,
    dashed); ``inside`` regions are green dotted, ``outside`` (keep-out) regions red.
    Variable points are marked for ``x`` (baseline when ``x`` is None).

    Parameters
    ----------
    space : DesignSpace
    x : array_like, optional
        Normalised design vector; the baseline vehicle is drawn if omitted.
    ax : Axes3D, optional
        Existing 3D axes; a new figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from suspkin.optimize.spec.constraints import InsideConstraint, OutsideConstraint
    from suspkin.optimize.spec.variables import parse_point_ref

    variables = space.study.variables
    if x is None:
        vehicle = space.baseline
        marks = [
            (_baseline_point(space, *parse_point_ref(v.point), _is_right(v)), not v.asymmetric)
            for v in variables
        ]
    else:
        vehicle = space.to_vehicle(x)
        pts = space.points(x)
        marks = [
            (pts[f"{v.point}@right" if _is_right(v) else v.point], not v.asymmetric)
            for v in variables
        ]
    fig = plot_vehicle_3d(vehicle, ax)
    axes: Any = fig.axes[0] if ax is None else ax
    drawn: list[F64] = []

    def draw(lines: list[F64], **kw: Any) -> None:
        for ln in lines:
            axes.plot(ln[:, 0], ln[:, 1], ln[:, 2], **kw)
            drawn.append(ln)

    for var in variables:
        verts = _domain_vertices(space, var)
        draw(_hex_lines(verts), color=_GREEN, lw=1.0, ls="-")
        if not var.asymmetric:
            draw(_hex_lines(_mirror(verts)), color=_GREEN, lw=0.8, ls="--")
    for c in space.study.constraints:
        if isinstance(c, InsideConstraint):
            draw(_region_lines(c.region), color=_GREEN, lw=0.9, ls=":")
        elif isinstance(c, OutsideConstraint):
            draw(_region_lines(c.region), color=_RED, lw=0.9, ls="-")
    for p, sym in marks:
        pp = np.vstack([p, _mirror(p)]) if sym else p[None, :]
        axes.scatter(pp[:, 0], pp[:, 1], pp[:, 2], color=_GREEN, s=14, depthshade=False)
        drawn.append(pp)

    if drawn:
        pts_all = np.vstack(drawn)
        lims = (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())
        new = [
            (
                min(lims[i][0], float(np.min(pts_all[:, i]))),
                max(lims[i][1], float(np.max(pts_all[:, i]))),
            )
            for i in range(3)
        ]
        axes.set_xlim(*new[0])
        axes.set_ylim(*new[1])
        axes.set_zlim(*new[2])
        axes.set_box_aspect(tuple(max(hi - lo, 1e-6) for lo, hi in new))
    leg = axes.get_legend()
    handles = list(leg.legend_handles) if leg is not None else []
    handles += [
        Line2D([0], [0], color=_GREEN, lw=1.2, label="variable domain"),
        Line2D([0], [0], color=_GREEN, lw=1.2, ls="--", label="mirrored domain"),
        Line2D([0], [0], color=_GREEN, lw=1.2, ls=":", label="inside region"),
        Line2D([0], [0], color=_RED, lw=1.2, label="keep-out region"),
    ]
    axes.legend(handles=handles, loc="upper left", fontsize=7)
    return fig
