"""Command-line interface for suspkin."""

from pathlib import Path
from typing import Annotated

import typer

from suspkin import __version__
from suspkin.geometry import VehicleConfigError, load_vehicle

app = typer.Typer(help="suspkin - suspension kinematics toolkit.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """suspkin command group."""


@app.command()
def version() -> None:
    """Print the installed suspkin version."""
    typer.echo(f"suspkin {__version__}")


@app.command()
def show(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path | None, typer.Option("--out", help="Save figure to this path.")] = None,
    labels: Annotated[bool, typer.Option("--labels", help="Label hardpoints.")] = False,
) -> None:
    """Show the static 3D suspension geometry of a vehicle."""
    try:
        vehicle = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    import matplotlib.pyplot as plt

    from suspkin.plotting import plot_vehicle_3d

    if out is not None:
        fig = plot_vehicle_3d(vehicle, show_labels=labels)
        fig.savefig(out, dpi=200)
        typer.echo(str(out))
        return
    fig = plt.figure(figsize=(9, 6))
    plot_vehicle_3d(vehicle, fig.add_subplot(projection="3d"), show_labels=labels)
    plt.show()


@app.command()
def sweep(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    corner: Annotated[str, typer.Option("--corner", help="Corner: FL, FR, RL or RR.")] = "FL",
    travel: Annotated[
        tuple[float, float] | None,
        typer.Option("--travel", help="Wheel travel range MIN MAX [m] (bump positive)."),
    ] = None,
    rack_sweep: Annotated[
        tuple[float, float] | None,
        typer.Option("--rack-sweep", help="Rack range MIN MAX [m] at zero travel."),
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=2, help="Number of sweep points.")] = 21,
    rack: Annotated[
        float, typer.Option("--rack", help="Fixed rack offset [m] (travel sweep).")
    ] = 0.0,
    metrics: Annotated[
        bool, typer.Option("--metrics", help="Add metric and gradient columns (>= 3 steps).")
    ] = False,
) -> None:
    """Sweep one corner over wheel travel or rack and write a CSV table (SI units)."""
    if (travel is None) == (rack_sweep is None):
        typer.echo("error: give exactly one of --travel or --rack-sweep", err=True)
        raise typer.Exit(code=2)
    if corner not in ("FL", "FR", "RL", "RR"):
        typer.echo(f"error: unknown corner {corner!r}", err=True)
        raise typer.Exit(code=2)
    try:
        vehicle = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    import numpy as np

    from suspkin.geometry import build_corners
    from suspkin.kinematics import KinematicsError, sweep_corner, sweep_rack
    from suspkin.sweeps import states_to_frame

    c = build_corners(vehicle)[corner]  # type: ignore[index]
    try:
        if travel is not None:
            states = sweep_corner(c, np.linspace(travel[0], travel[1], steps), rack=rack)
        else:
            assert rack_sweep is not None
            states = sweep_rack(c, np.linspace(rack_sweep[0], rack_sweep[1], steps))
    except (KinematicsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    df = states_to_frame(states, c, metrics=metrics)
    if metrics:
        from suspkin.metrics import add_gradients

        try:
            df = add_gradients(df, "travel" if travel is not None else "rack")
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    df.to_csv(out, index=False)
    typer.echo(str(out))


_ANGLE_COLS = (
    "camber", "toe", "steer_angle", "caster", "kpi",
    "fv_swing_angle", "sv_swing_angle", "sv_wc_angle",
)  # fmt: skip
_ANGLE_RATE_COLS = ("camber_gain", "bump_steer", "caster_change")
_RATIO_COLS = ("track_change_rate", "rc_migration_rate")


def _display(name: str, value: float) -> tuple[float, str]:
    """Convert an SI metric value to display units; return (value, unit)."""
    from suspkin.units import m_to_mm, rad_to_deg

    if name in _ANGLE_COLS:
        return float(rad_to_deg(value)), "deg"
    if name in _ANGLE_RATE_COLS:
        return float(rad_to_deg(value)) / 1000.0, "deg/mm"
    if name in _RATIO_COLS:
        return value, "mm/mm"
    return float(m_to_mm(value)), "mm"


@app.command()
def summary(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the SI static summary as JSON.")
    ] = None,
) -> None:
    """Print static alignment, centre and gradient metrics of FL and RL (display units)."""
    import json

    import numpy as np

    from suspkin.geometry import build_corners
    from suspkin.kinematics import KinematicsError, sweep_corner
    from suspkin.metrics import add_gradients, metrics_frame, static_summary

    try:
        vehicle = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    corners = build_corners(vehicle)
    result: dict[str, dict[str, float]] = {}
    try:
        for cid in ("FL", "RL"):
            c = corners[cid]
            states = sweep_corner(c, np.linspace(-0.01, 0.01, 21))
            result[cid] = static_summary(add_gradients(metrics_frame(states, c)))
    except (KinematicsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    names = list(result["FL"])
    width = max(len(n) for n in names)
    typer.echo(f"{'metric':<{width}}  {'FL':>12}  {'RL':>12}  unit")
    for n in names:
        cells = []
        unit = ""
        for cid in ("FL", "RL"):
            v, unit = _display(n, result[cid][n])
            cells.append(f"{v:12.4f}")
        typer.echo(f"{n:<{width}}  {cells[0]}  {cells[1]}  {unit}")
    if json_out is not None:
        json_out.write_text(json.dumps(result, indent=2))
        typer.echo(str(json_out))
