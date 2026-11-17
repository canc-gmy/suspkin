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
_ACT_ANGLE_COLS = ("pushrod_angle", "pushrod_angle_fv", "rocker_rotation")
_FORCE_COLS = ("spring_force", "wheel_load_spring")
_RATE_COLS = ("spring_rate", "tyre_stiffness", "wheel_rate_linear", "wheel_rate", "ride_rate")


def _display(name: str, value: float) -> tuple[float, str]:
    """Convert an SI metric value to display units; return (value, unit)."""
    from suspkin.units import m_to_mm, rad_to_deg

    if name in _ANGLE_COLS or name in _ACT_ANGLE_COLS:
        return float(rad_to_deg(value)), "deg"
    if name in _FORCE_COLS:
        return value, "N"
    if name in _RATE_COLS:
        return value / 1000.0, "N/mm"
    if name == "mr_spring":
        return value, "-"
    if name == "mr_progression":
        return value / 1000.0, "1/mm"
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


# key -> (scale from SI, unit) for the vehicle summary display
_DEG = 180.0 / 3.141592653589793
_VS_UNITS: dict[str, tuple[float, str]] = {
    "track": (1e3, "mm"),
    "wheel_rate": (1e-3, "N/mm"),
    "arb_wheel_rate_roll": (1e-3, "N/mm"),
    "roll_stiffness_springs": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_arb": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_tyres": (1 / _DEG, "N*m/deg"),
    "roll_stiffness": (1 / _DEG, "N*m/deg"),
    "roll_stiffness_total": (1 / _DEG, "N*m/deg"),
    "rc_height": (1e3, "mm"),
    "roll_camber": (1.0, "deg/deg"),
    "roll_steer": (1.0, "deg/deg"),
    "rc_migration": (1e3 / _DEG, "mm/deg"),
    "rc_height_gradient": (1e3 / _DEG, "mm/deg"),
    "wheelbase": (1e3, "mm"),
    "sprung_mass": (1.0, "kg"),
    "sprung_cg_height": (1e3, "mm"),
    "roll_axis_height_at_cg": (1e3, "mm"),
    "roll_gradient": (_DEG, "deg/g"),
    "ay_ref": (1.0, "m/s^2"),
    "load_transfer_front": (1.0, "N"),
    "load_transfer_rear": (1.0, "N"),
    "tlltd_front": (100.0, "%"),
    "anti_dive_front": (1.0, "%"),
    "anti_lift_rear": (1.0, "%"),
    "anti_squat_rear": (1.0, "%"),
    "anti_lift_front_accel": (1.0, "%"),
    "pitch_centre_x": (1e3, "mm"),
    "pitch_centre_z": (1e3, "mm"),
    "max_steer_inner": (_DEG, "deg"),
    "max_steer_outer": (_DEG, "deg"),
    "ackermann_5deg": (1.0, "%"),
    "ackermann_10deg": (1.0, "%"),
    "ackermann_15deg": (1.0, "%"),
    "steering_ratio": (1.0, "-"),
    "turning_radius": (1.0, "m"),
}


@app.command()
def vehicle(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the SI vehicle summary as JSON.")
    ] = None,
) -> None:
    """Print the vehicle-level summary (rates, roll, anti-features, steering) in display units."""
    import json

    from suspkin.kinematics import KinematicsError
    from suspkin.metrics import vehicle_summary

    try:
        veh = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    try:
        data = vehicle_summary(veh).to_dict()
    except (KinematicsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    rows: list[tuple[str, object]] = []
    for k, v in data.items():
        if isinstance(v, dict):
            rows += [(f"{k}.{kk}", vv) for kk, vv in v.items()]
        else:
            rows.append((k, v))
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        scale, unit = _VS_UNITS.get(k.split(".")[-1], (1.0, ""))
        if v is None:
            typer.echo(f"{k:<{width}}  {'n/a':>12}  {unit}")
        else:
            typer.echo(f"{k:<{width}}  {float(v) * scale:12.4f}  {unit}")  # type: ignore[arg-type]
    if json_out is not None:
        json_out.write_text(json.dumps(data, indent=2))
        typer.echo(str(json_out))


@app.command()
def vsweep(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    heave: Annotated[
        tuple[float, float] | None, typer.Option("--heave", help="Heave range MIN MAX [m].")
    ] = None,
    roll: Annotated[
        tuple[float, float] | None, typer.Option("--roll", help="Roll range MIN MAX [deg].")
    ] = None,
    pitch: Annotated[
        tuple[float, float] | None, typer.Option("--pitch", help="Pitch range MIN MAX [deg].")
    ] = None,
    rack: Annotated[
        tuple[float, float] | None, typer.Option("--rack", help="Rack range MIN MAX [m].")
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=2, help="Number of sweep points.")] = 21,
    base_heave: Annotated[float, typer.Option("--base-heave", help="Fixed heave [m].")] = 0.0,
    base_roll: Annotated[float, typer.Option("--base-roll", help="Fixed roll [deg].")] = 0.0,
    base_pitch: Annotated[float, typer.Option("--base-pitch", help="Fixed pitch [deg].")] = 0.0,
    base_rack: Annotated[float, typer.Option("--base-rack", help="Fixed rack [m].")] = 0.0,
) -> None:
    """Sweep the body pose (heave, roll, pitch or rack) and write a CSV table (SI units).

    Roll and pitch are given in DEGREES on the command line (converted to rad); the CSV
    is in SI (m, rad). Give exactly one of --heave, --roll, --pitch, --rack.
    """
    import numpy as np

    from suspkin.kinematics import KinematicsError
    from suspkin.sweeps import sweep_vehicle, vehicle_frame

    given = {"heave": heave, "roll": roll, "pitch": pitch, "rack": rack}
    chosen = [k for k, v in given.items() if v is not None]
    if len(chosen) != 1:
        typer.echo("error: give exactly one of --heave, --roll, --pitch, --rack", err=True)
        raise typer.Exit(code=2)
    try:
        veh = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    name = chosen[0]
    lo, hi = given[name]  # type: ignore[misc]
    if name in ("roll", "pitch"):
        lo, hi = np.radians(lo), np.radians(hi)
    base = {
        "heave": base_heave,
        "roll": float(np.radians(base_roll)),
        "pitch": float(np.radians(base_pitch)),
        "rack": base_rack,
    }
    base.pop(name)
    try:
        states = sweep_vehicle(veh, **{name: np.linspace(lo, hi, steps)}, base=base)
        df = vehicle_frame(states, veh)
    except (KinematicsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    df.to_csv(out, index=False)
    typer.echo(str(out))
