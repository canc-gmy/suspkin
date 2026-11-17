"""Static 3D view of the vehicle suspension geometry (matplotlib only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from numpy.typing import NDArray

from suspkin.geometry import Corner, CornerId, Vehicle, arb_points, build_corners

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

GROUP_COLORS: dict[str, str] = {
    "wishbones": "#1f77b4",
    "tie rod": "#d62728",
    "upright": "#7f7f7f",
    "pushrod": "#ff7f0e",
    "rocker": "#9467bd",
    "coilover": "#2ca02c",
    "ARB": "#e377c2",
    "wheel": "#222222",
}


def wheel_spin_axis(
    side: Literal["left", "right"], camber: float, toe: float
) -> NDArray[np.float64]:
    """Unit wheel spin axis in the ISO 8855 frame, pointing outboard.

    Zero camber and toe give ``(0, 1, 0)`` for the left wheel and ``(0, -1, 0)``
    for the right wheel. Conventions (DESIGN_PLAN section 2): negative camber
    means the top of the wheel leans toward the car centre; positive toe is
    toe-in (front of the wheel points toward the car centre). The right-side
    axis is the y-mirror of the left one.

    Parameters
    ----------
    side : {"left", "right"}
    camber, toe : float
        Static camber and toe [rad].
    """
    n = np.array([0.0, np.cos(camber), -np.sin(camber)])
    c, s = np.cos(-toe), np.sin(-toe)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    n = rz @ n
    if side == "right":
        n = n * np.array([1.0, -1.0, 1.0])
    return np.asarray(n / np.linalg.norm(n), dtype=np.float64)


def _wheel_basis(
    axis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """In-plane unit vectors: ``up`` (towards the top) and ``u`` (completing the frame)."""
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, axis) * axis
    up = up / np.linalg.norm(up)
    return up, np.cross(axis, up)


def _line(ax: Axes3D, pts: list[NDArray[np.float64]], color: str, lw: float = 1.5) -> None:
    a = np.array(pts)
    ax.plot(a[:, 0], a[:, 1], a[:, 2], color=color, lw=lw)


def _poly(ax: Axes3D, pts: list[NDArray[np.float64]], color: str) -> None:
    ax.add_collection3d(
        Poly3DCollection([np.array(pts)], facecolors=color, edgecolors=color, alpha=0.25)
    )


def _draw_corner(ax: Axes3D, c: Corner, show_labels: bool) -> list[NDArray[np.float64]]:
    p = c.points
    col = GROUP_COLORS
    for a, b in (
        ("uca_front", "uca_outer"),
        ("uca_rear", "uca_outer"),
        ("lca_front", "lca_outer"),
        ("lca_rear", "lca_outer"),
    ):
        _line(ax, [p[a], p[b]], col["wishbones"])
    _line(ax, [p["tie_inner"], p["tie_outer"]], col["tie rod"])
    _poly(ax, [p["uca_outer"], p["lca_outer"], p["tie_outer"]], col["upright"])
    _line(ax, [p["uca_outer"], p["wheel_center"]], col["upright"])
    _line(ax, [p["pushrod_outer"], p["pushrod_inner"]], col["pushrod"])
    rocker = [p["rocker_pivot"], p["pushrod_inner"], p["rocker_spring"]]
    if "rocker_arb" in p:
        rocker.append(p["rocker_arb"])
    _poly(ax, rocker, col["rocker"])
    half = 0.03 * c.rocker_axis / np.linalg.norm(c.rocker_axis)
    _line(ax, [p["rocker_pivot"] - half, p["rocker_pivot"] + half], col["rocker"], 2.0)
    _line(ax, [p["rocker_spring"], p["spring_chassis"]], col["coilover"], 3.0)

    ab = arb_points(c)
    markers = list(p.values())
    if ab:
        root = ab["bar_pivot"] if "bar_pivot" in ab else ab["blade_root"]
        if "rocker_arb" in p:
            _line(ax, [p["rocker_arb"], ab["arm_end"]], col["ARB"])
        _line(ax, [ab["arm_end"], root], col["ARB"])
        markers += [ab["arm_end"], root]

    wheel = c.axle_cfg.wheel
    axis = wheel_spin_axis(c.side, wheel.static_camber, wheel.static_toe)
    up, u = _wheel_basis(axis)
    t = np.linspace(0.0, 2.0 * np.pi, 73)
    ctr = p["wheel_center"]
    circle = ctr + wheel.tyre_radius * (np.outer(np.cos(t), up) + np.outer(np.sin(t), u))
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color=col["wheel"], lw=1.2)
    contact = ctr - wheel.tyre_radius * up
    ax.scatter(*contact, color=col["wheel"], marker="x", s=30)

    for q in markers:
        ax.scatter(*q, color="k", s=6)
    if show_labels:
        for k, v in p.items():
            ax.text(v[0], v[1], v[2], f"{c.id}:{k}", fontsize=5)
    return [*markers, circle.min(axis=0), circle.max(axis=0), contact]


def plot_vehicle_3d(
    vehicle: Vehicle, ax: Axes3D | None = None, *, show_labels: bool = False
) -> Figure:
    """Draw the static 3D suspension geometry of all four corners.

    Parameters
    ----------
    vehicle : Vehicle
    ax : Axes3D, optional
        Existing 3D axes to draw into; a new figure is created if omitted.
    show_labels : bool
        Annotate hardpoints with their names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig = Figure(figsize=(9, 6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.get_figure()
    corners = build_corners(vehicle)
    pts: list[NDArray[np.float64]] = []
    for c in corners.values():
        pts += _draw_corner(ax, c, show_labels)

    pairs: list[tuple[CornerId, CornerId]] = [("FL", "FR"), ("RL", "RR")]
    for left, right in pairs:
        al, ar = arb_points(corners[left]), arb_points(corners[right])
        if "bar_pivot" in al:
            _line(ax, [al["bar_pivot"], ar["bar_pivot"]], GROUP_COLORS["ARB"], 2.5)

    allp = np.array(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    lo[2] = min(lo[2], 0.0)
    ground = np.array(
        [[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0], [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0]]
    )
    ax.add_collection3d(
        Poly3DCollection([ground], facecolors="#cccccc", edgecolors="none", alpha=0.25)
    )
    span = np.maximum(hi - lo, 1e-6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(span))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(vehicle.vehicle.name)
    handles = [Line2D([0], [0], color=c, lw=2, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    return fig
