"""Plotting helpers (read-only consumers of geometry and result tables).

``plot_study_3d`` is imported lazily: ``suspkin.optimize`` imports this package, and
:mod:`suspkin.plotting.regions` needs ``suspkin.optimize.spec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from suspkin.plotting.animation import animate_sweep
from suspkin.plotting.figures import (
    FIGURE_INPUTS,
    FIGURE_SET,
    fig_actuation,
    fig_corner_alignment,
    fig_corner_centres,
    fig_corner_position,
    fig_heave_pitch,
    fig_roll,
    fig_steering,
    fig_summary_table,
    save_figure,
)
from suspkin.plotting.geometry3d import (
    VehicleScene,
    draw_scene,
    plot_vehicle_3d,
    vehicle_segments,
    wheel_spin_axis,
)
from suspkin.plotting.interactive import interactive_view
from suspkin.plotting.optimization import (
    knee_index,
    pareto_mask,
    plot_constraint_margins,
    plot_convergence,
    plot_parallel_coordinates,
    plot_pareto,
)
from suspkin.plotting.style import (
    PALETTE,
    design_color,
    figure_size,
    side_linestyle,
    use_style,
)

if TYPE_CHECKING:
    from suspkin.plotting.regions import plot_study_3d

__all__ = [
    "FIGURE_INPUTS",
    "FIGURE_SET",
    "PALETTE",
    "VehicleScene",
    "animate_sweep",
    "design_color",
    "draw_scene",
    "fig_actuation",
    "fig_corner_alignment",
    "fig_corner_centres",
    "fig_corner_position",
    "fig_heave_pitch",
    "fig_roll",
    "fig_steering",
    "fig_summary_table",
    "figure_size",
    "interactive_view",
    "knee_index",
    "pareto_mask",
    "plot_constraint_margins",
    "plot_convergence",
    "plot_parallel_coordinates",
    "plot_pareto",
    "plot_study_3d",
    "plot_vehicle_3d",
    "save_figure",
    "side_linestyle",
    "use_style",
    "vehicle_segments",
    "wheel_spin_axis",
]


def __getattr__(name: str) -> Any:
    if name == "plot_study_3d":
        from suspkin.plotting.regions import plot_study_3d

        return plot_study_3d
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
