"""3D view of the vehicle suspension geometry (matplotlib; scene builder shared by all views).

``vehicle_segments`` is a pure builder: it turns a vehicle (and optionally a solved
``VehicleState``) into a ``VehicleScene`` of line segments, polygons, wheel circles and
marker points. The matplotlib drawing here, the GIF animation and the plotly view all
consume that one scene.

Body-fixed frame: the solver works in the body frame, so the car body (chassis points,
rocker pivots, ARB pivots) never moves and the *ground* moves. In a solved pose the
ground plane is drawn tilted by ``state.ground_normal`` through the contact points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]
from numpy.typing import NDArray

from suspkin.geom_math.vectors import wheel_spin_axis as wheel_spin_axis
from suspkin.geometry import Corner, CornerId, Vehicle, arb_points, build_corners

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from suspkin.kinematics.vehicle import VehicleState

Vec = NDArray[np.float64]
Segment = tuple[Vec, Vec]

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

#: Line width per group (shared by matplotlib and plotly).
GROUP_WIDTHS: dict[str, float] = {
    "wishbones": 1.5,
    "tie rod": 1.5,
    "upright": 1.5,
    "pushrod": 1.5,
    "rocker": 2.0,
    "coilover": 3.0,
    "ARB": 2.0,
    "wheel": 1.2,
}


@dataclass(frozen=True)
class VehicleScene:
    """Drawable description of a vehicle pose (body frame, metres)."""

    segments: dict[str, list[Segment]]
    wheels: list[Vec]
    polygons: list[tuple[str, Vec]]
    points: Vec
    contacts: Vec
    labels: list[tuple[str, Vec]]
    ground_normal: Vec
    ground_point: Vec

    def extent(self) -> tuple[Vec, Vec]:
        """Axis-aligned bounds of all points, wheel circles and contacts."""
        allp = np.vstack([self.points, self.contacts, *self.wheels])
        lo: Vec = allp.min(axis=0)
        hi: Vec = allp.max(axis=0)
        return lo, hi


def ground_quad(scene: VehicleScene, lo: Vec, hi: Vec) -> Vec:
    """Corners (4, 3) of the ground plane above the xy rectangle ``lo``..``hi``."""
    n, p0 = scene.ground_normal, scene.ground_point
    xy = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    z = p0[2] - (n[0] * (xy[:, 0] - p0[0]) + n[1] * (xy[:, 1] - p0[1])) / n[2]
    out: Vec = np.column_stack([xy, z])
    return out


def scene_bounds(scenes: Sequence[VehicleScene]) -> tuple[Vec, Vec]:
    """Common bounds of several scenes, with each ground plane inside the z range."""
    ext = [s.extent() for s in scenes]
    lo: Vec = np.array([e[0] for e in ext], dtype=np.float64).min(axis=0)
    hi: Vec = np.array([e[1] for e in ext], dtype=np.float64).max(axis=0)
    lo_z, hi_z = float(lo[2]), float(hi[2])
    for s in scenes:
        gz = ground_quad(s, lo, hi)[:, 2]
        lo_z, hi_z = min(lo_z, float(np.min(gz))), max(hi_z, float(np.max(gz)))
    lo[2], hi[2] = lo_z, hi_z
    return lo, hi


def _wheel_basis(axis: Vec, vertical: Vec | None = None) -> tuple[Vec, Vec]:
    """In-plane unit vectors: ``up`` (towards the top) and ``u`` (completing the frame)."""
    z = np.array([0.0, 0.0, 1.0]) if vertical is None else vertical
    up = z - np.dot(z, axis) * axis
    up = up / np.linalg.norm(up)
    return up, np.cross(axis, up)


def _add_corner(
    scene: dict[str, object],
    c: Corner,
    pts_in: Mapping[str, Vec],
    spin: Vec,
    vertical: Vec | None,
    arm_end: Vec | None,
    beam_heave: float,
    contact: Vec | None,
) -> None:
    seg: dict[str, list[Segment]] = scene["seg"]  # type: ignore[assignment]
    polys: list[tuple[str, Vec]] = scene["polys"]  # type: ignore[assignment]
    wheels: list[Vec] = scene["wheels"]  # type: ignore[assignment]
    markers: list[Vec] = scene["markers"]  # type: ignore[assignment]
    labels: list[tuple[str, Vec]] = scene["labels"]  # type: ignore[assignment]
    contacts: list[Vec] = scene["contacts"]  # type: ignore[assignment]

    p = {k: np.asarray(v, dtype=np.float64) for k, v in pts_in.items()}
    for a, b in (
        ("uca_front", "uca_outer"),
        ("uca_rear", "uca_outer"),
        ("lca_front", "lca_outer"),
        ("lca_rear", "lca_outer"),
    ):
        seg["wishbones"].append((p[a], p[b]))
    seg["tie rod"].append((p["tie_inner"], p["tie_outer"]))
    polys.append(("upright", np.array([p["uca_outer"], p["lca_outer"], p["tie_outer"]])))
    seg["upright"].append((p["uca_outer"], p["wheel_center"]))
    seg["pushrod"].append((p["pushrod_outer"], p["pushrod_inner"]))
    rocker = [p["rocker_pivot"], p["pushrod_inner"], p["rocker_spring"]]
    if "rocker_arb" in p:
        rocker.append(p["rocker_arb"])
    polys.append(("rocker", np.array(rocker)))
    half = 0.03 * c.rocker_axis / np.linalg.norm(c.rocker_axis)
    seg["rocker"].append((p["rocker_pivot"] - half, p["rocker_pivot"] + half))
    seg["coilover"].append((p["rocker_spring"], p["spring_chassis"]))

    markers.extend(p.values())
    ab = arb_points(c)
    if ab:
        end = ab["arm_end"] if arm_end is None else arm_end
        # torsional/blade: arm from bar pivot to arm end; beam: half beam from centre to end
        root = ab["bar_pivot"] if "bar_pivot" in ab else ab["beam_center"]
        if "bar_pivot" not in ab:
            root = root + np.array([0.0, 0.0, beam_heave])
        if "rocker_arb" in p:
            seg["ARB"].append((p["rocker_arb"], end))
        seg["ARB"].append((end, root))
        markers.extend([end, root])

    wheel = c.axle_cfg.wheel
    up, u = _wheel_basis(spin, vertical)
    t = np.linspace(0.0, 2.0 * np.pi, 73)
    ctr = p["wheel_center"]
    wheels.append(ctr + wheel.tyre_radius * (np.outer(np.cos(t), up) + np.outer(np.sin(t), u)))
    contacts.append(ctr - wheel.tyre_radius * up if contact is None else contact)
    labels.extend((f"{c.id}:{k}", v) for k, v in p.items())


def vehicle_segments(vehicle: Vehicle, state: VehicleState | None = None) -> VehicleScene:
    """Build the drawable scene of a vehicle, static or at a solved pose.

    Parameters
    ----------
    vehicle : Vehicle
    state : VehicleState, optional
        Solved pose. Moved points come from each ``CornerState`` and the ARB from
        ``ArbState``; the wheel circle uses ``spin_axis`` and the ground plane is
        tilted by ``state.ground_normal`` through the contact points. Without it the
        static hardpoints and the plane z = 0 are used. The body frame is fixed.

    Returns
    -------
    VehicleScene
        ``segments`` maps each group of ``GROUP_COLORS`` (except ``"wheel"``) to a list
        of ``(p0, p1)`` pairs; wheel circles are polylines in ``wheels``.
    """
    corners = build_corners(vehicle)
    acc: dict[str, object] = {
        "seg": {g: [] for g in GROUP_COLORS if g != "wheel"},
        "polys": [],
        "wheels": [],
        "markers": [],
        "labels": [],
        "contacts": [],
    }
    for cid, c in corners.items():
        if state is None:
            wheel = c.axle_cfg.wheel
            spin = wheel_spin_axis(c.side, wheel.static_camber, wheel.static_toe)
            _add_corner(acc, c, c.points, spin, None, None, 0.0, None)
        else:
            st = state.corners[cid]
            arb = state.arb.get(c.axle)
            arm = None
            heave = 0.0
            if arb is not None:
                arm = arb.arm_end_left if c.side == "left" else arb.arm_end_right
                heave = arb.heave if arb.type == "beam" else 0.0
            _add_corner(
                acc,
                c,
                st.points,
                np.asarray(st.spin_axis, dtype=np.float64),
                np.asarray(state.ground_normal, dtype=np.float64),
                arm,
                heave,
                np.asarray(state.contact_points[cid], dtype=np.float64),
            )

    seg: dict[str, list[Segment]] = acc["seg"]  # type: ignore[assignment]
    pairs: list[tuple[CornerId, CornerId]] = [("FL", "FR"), ("RL", "RR")]
    for left, right in pairs:
        al, ar = arb_points(corners[left]), arb_points(corners[right])
        if "bar_pivot" in al:  # torsional / blade: tube between the pivots
            seg["ARB"].append((al["bar_pivot"], ar["bar_pivot"]))

    contact_arr = np.array(acc["contacts"])
    if state is None:
        gp = np.zeros(3)
        gn = np.array([0.0, 0.0, 1.0])
    else:
        gp = contact_arr.mean(axis=0)
        gn = np.asarray(state.ground_normal, dtype=np.float64)
    return VehicleScene(
        segments=seg,
        wheels=acc["wheels"],  # type: ignore[arg-type]
        polygons=acc["polys"],  # type: ignore[arg-type]
        points=np.array(acc["markers"]),
        contacts=contact_arr,
        labels=acc["labels"],  # type: ignore[arg-type]
        ground_normal=gn,
        ground_point=gp,
    )


def draw_scene(
    ax: Axes3D,
    scene: VehicleScene,
    bounds: tuple[Vec, Vec] | None = None,
    *,
    show_labels: bool = False,
    title: str = "",
) -> None:
    """Draw a scene into 3D axes with limits from ``bounds`` (default: the scene's own)."""
    for group, items in scene.segments.items():
        for a, b in items:
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color=GROUP_COLORS[group],
                lw=GROUP_WIDTHS[group],
            )
    for group, pts in scene.polygons:
        col = GROUP_COLORS[group]
        ax.add_collection3d(Poly3DCollection([pts], facecolors=col, edgecolors=col, alpha=0.25))
    for circle in scene.wheels:
        ax.plot(
            circle[:, 0],
            circle[:, 1],
            circle[:, 2],
            color=GROUP_COLORS["wheel"],
            lw=GROUP_WIDTHS["wheel"],
        )
    for q in scene.contacts:
        ax.scatter(*q, color=GROUP_COLORS["wheel"], marker="x", s=30)
    for q in scene.points:
        ax.scatter(*q, color="k", s=6)
    if show_labels:
        for text, v in scene.labels:
            ax.text(v[0], v[1], v[2], text, fontsize=5)

    lo, hi = scene_bounds([scene]) if bounds is None else bounds
    ax.add_collection3d(
        Poly3DCollection(
            [ground_quad(scene, lo, hi)], facecolors="#cccccc", edgecolors="none", alpha=0.25
        )
    )
    span = np.maximum(hi - lo, 1e-6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(span))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    handles = [Line2D([0], [0], color=c, lw=2, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=7)


def plot_vehicle_3d(
    vehicle: Vehicle,
    ax: Axes3D | None = None,
    *,
    state: VehicleState | None = None,
    show_labels: bool = False,
) -> Figure:
    """Draw the 3D suspension geometry of all four corners, static or at a solved pose.

    The body frame stays fixed; with ``state`` the links, wheels, rockers, coilovers and
    ARBs are drawn at their solved positions and the ground plane is tilted/shifted.

    Parameters
    ----------
    vehicle : Vehicle
    ax : Axes3D, optional
        Existing 3D axes to draw into; a new figure is created if omitted.
    state : VehicleState, optional
        Solved pose from ``solve_vehicle``; static geometry if omitted.
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
    draw_scene(
        ax, vehicle_segments(vehicle, state), show_labels=show_labels, title=vehicle.vehicle.name
    )
    return fig
