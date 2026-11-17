"""Plotting helpers (read-only consumers of geometry and result tables)."""

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
from suspkin.plotting.regions import plot_study_3d
from suspkin.plotting.style import (
    PALETTE,
    design_color,
    figure_size,
    side_linestyle,
    use_style,
)

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
    "plot_study_3d",
    "plot_vehicle_3d",
    "save_figure",
    "side_linestyle",
    "use_style",
    "vehicle_segments",
    "wheel_spin_axis",
]
