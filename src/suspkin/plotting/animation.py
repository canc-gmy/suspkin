"""Animated 3D view of a vehicle pose sweep (GIF via Pillow, MP4 via ffmpeg)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from suspkin.geometry import Vehicle
from suspkin.kinematics.vehicle import VehicleState
from suspkin.plotting.geometry3d import draw_scene, scene_bounds, vehicle_segments


def animate_sweep(
    vehicle: Vehicle,
    states: Sequence[VehicleState],
    path: str | Path,
    *,
    fps: int = 15,
    view: tuple[float, float] = (22.0, -60.0),
) -> Path:
    """Render a sweep of solved poses as an animation (body frame fixed, ground moves).

    Args:
        vehicle (Vehicle):
        states (sequence of VehicleState): One frame per state; axis limits are fixed over the whole
            sweep.
        path (str or Path): Output file; ``.gif`` (Pillow) or ``.mp4`` (needs ffmpeg on PATH).
        fps (int): Frames per second.
        view ((float, float)): Camera ``(elevation, azimuth)`` in degrees.

    Returns:
        Path: The written file.

    Raises:
        ValueError: If ``states`` is empty or the suffix is neither ``.gif`` nor ``.mp4``.
        RuntimeError: For ``.mp4`` when ffmpeg is not available.
    """
    out = Path(path)
    suffix = out.suffix.lower()
    if not states:
        raise ValueError("states must not be empty")
    if suffix == ".gif":
        writer: PillowWriter | FFMpegWriter = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg not found: install ffmpeg to write .mp4, or use a .gif path")
        writer = FFMpegWriter(fps=fps)
    else:
        raise ValueError(f"unsupported animation suffix {suffix!r}; use .gif or .mp4")

    scenes = [vehicle_segments(vehicle, s) for s in states]
    bounds = scene_bounds(scenes)
    fig = Figure(figsize=(8, 5.5))
    ax = fig.add_subplot(projection="3d")

    def frame(i: int) -> None:
        ax.clear()
        draw_scene(ax, scenes[i], bounds, title=f"{vehicle.vehicle.name}  [{i + 1}/{len(scenes)}]")
        ax.view_init(elev=view[0], azim=view[1])

    anim = FuncAnimation(fig, frame, frames=len(scenes), blit=False)
    anim.save(str(out), writer=writer, dpi=80)
    return out
