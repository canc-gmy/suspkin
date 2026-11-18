"""Interactive plotly 3D view of the vehicle (optional ``viz`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from suspkin.geometry import Vehicle
from suspkin.kinematics.vehicle import VehicleState
from suspkin.plotting.geometry3d import (
    GROUP_COLORS,
    GROUP_WIDTHS,
    ground_quad,
    scene_bounds,
    vehicle_segments,
)


def _import_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the interactive view: install the 'viz' extra "
            "(pip install 'suspkin[viz]')"
        ) from exc
    return go


def interactive_view(
    vehicle: Vehicle,
    state: VehicleState | None = None,
    path: str | Path = "vehicle_3d.html",
    *,
    inline: bool = False,
) -> Path:
    """Write a self-contained interactive plotly HTML of the vehicle (static or solved pose).

    Parameters
    ----------
    vehicle : Vehicle
    state : VehicleState, optional
        Solved pose; static geometry if omitted. Body frame fixed, ground moves.
    path : str or Path
        Output HTML file.
    inline : bool
        Embed plotly.js in the file (offline use, ~4 MB). Default loads it from the CDN.

    Returns
    -------
    Path

    Raises
    ------
    ImportError
        If plotly (the ``viz`` extra) is not installed.
    """
    fig = interactive_figure(vehicle, state)
    out = Path(path)
    fig.write_html(str(out), include_plotlyjs=True if inline else "cdn", full_html=True)
    return out


def interactive_figure(vehicle: Vehicle, state: VehicleState | None = None) -> Any:
    """Plotly ``Figure`` of the vehicle (static or solved pose), not written anywhere.

    Parameters
    ----------
    vehicle : Vehicle
    state : VehicleState, optional
        Solved pose; static geometry if omitted. Body frame fixed, ground moves.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If plotly (the ``viz`` extra) is not installed.
    """
    go = _import_plotly()
    scene = vehicle_segments(vehicle, state)
    traces: list[Any] = []

    def lines(name: str, color: str, width: float, polylines: list[Any]) -> None:
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for pl in polylines:
            for q in pl:
                xs.append(float(q[0]))
                ys.append(float(q[1]))
                zs.append(float(q[2]))
            xs.append(None)
            ys.append(None)
            zs.append(None)
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=name, line={"color": color, "width": width * 2}
            )
        )

    for group, items in scene.segments.items():
        lines(group, GROUP_COLORS[group], GROUP_WIDTHS[group], [list(s) for s in items])
    lines("wheel", GROUP_COLORS["wheel"], GROUP_WIDTHS["wheel"], list(scene.wheels))
    for group in ("upright", "rocker"):
        polys = [p for g, p in scene.polygons if g == group]
        verts = np.vstack(polys)
        ii: list[int] = []
        jj: list[int] = []
        kk: list[int] = []
        off = 0
        for p in polys:
            for t in range(1, len(p) - 1):  # triangle fan
                ii.append(off)
                jj.append(off + t)
                kk.append(off + t + 1)
            off += len(p)
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=ii,
                j=jj,
                k=kk,
                color=GROUP_COLORS[group],
                opacity=0.3,
                name=f"{group} plate",
                showlegend=False,
            )
        )
    pts = scene.points
    traces.append(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name="hardpoints",
            marker={"size": 2, "color": "black"},
        )
    )
    c = scene.contacts
    traces.append(
        go.Scatter3d(
            x=c[:, 0],
            y=c[:, 1],
            z=c[:, 2],
            mode="markers",
            name="contact patches",
            marker={"size": 4, "color": GROUP_COLORS["wheel"], "symbol": "x"},
        )
    )
    lo, hi = scene_bounds([scene])
    g = ground_quad(scene, lo, hi)
    traces.append(
        go.Mesh3d(
            x=g[:, 0],
            y=g[:, 1],
            z=g[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#cccccc",
            opacity=0.3,
            name="ground",
            showlegend=False,
        )
    )
    fig = go.Figure(traces)
    span = np.maximum(hi - lo, 1e-6)
    span = span / span.max()
    fig.update_layout(
        title=vehicle.vehicle.name,
        scene={
            "xaxis": {"title": "x [m]", "range": [float(lo[0]), float(hi[0])]},
            "yaxis": {"title": "y [m]", "range": [float(lo[1]), float(hi[1])]},
            "zaxis": {"title": "z [m]", "range": [float(lo[2]), float(hi[2])]},
            "aspectmode": "manual",
            "aspectratio": {"x": float(span[0]), "y": float(span[1]), "z": float(span[2])},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig
