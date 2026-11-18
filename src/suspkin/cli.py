"""Command-line interface for suspkin."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

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


def _pose_kwargs(
    heave: float | None, roll: float | None, pitch: float | None, rack: float | None
) -> dict[str, float]:
    """CLI pose (mm, deg) -> solver kwargs (m, rad); only the given entries."""
    import math

    out: dict[str, float] = {}
    if heave is not None:
        out["heave"] = heave / 1000.0
    if roll is not None:
        out["roll"] = math.radians(roll)
    if pitch is not None:
        out["pitch"] = math.radians(pitch)
    if rack is not None:
        out["rack"] = rack / 1000.0
    return out


@app.command()
def show(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path | None, typer.Option("--out", help="Save figure to this path.")] = None,
    labels: Annotated[bool, typer.Option("--labels", help="Label hardpoints.")] = False,
    heave: Annotated[float | None, typer.Option("--heave", help="Body heave (mm).")] = None,
    roll: Annotated[float | None, typer.Option("--roll", help="Body roll (deg).")] = None,
    pitch: Annotated[float | None, typer.Option("--pitch", help="Body pitch (deg).")] = None,
    rack: Annotated[float | None, typer.Option("--rack", help="Rack displacement (mm).")] = None,
    html: Annotated[
        bool, typer.Option("--html", help="Write an interactive plotly HTML (needs the viz extra).")
    ] = False,
) -> None:
    """Show the 3D suspension geometry, static or at a solved body pose.

    Heave and rack are in MILLIMETRES and roll and pitch in DEGREES (all commands use
    the same user-facing units; files are SI). With none of them given the static
    geometry is drawn. --html writes an interactive file (to --out, default
    <vehicle>.html in the current directory) instead of a matplotlib figure.
    """
    try:
        vehicle = load_vehicle(vehicle_yaml)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    state = None
    pose = _pose_kwargs(heave, roll, pitch, rack)
    if pose:
        from suspkin.kinematics import KinematicsError, solve_vehicle

        try:
            state = solve_vehicle(vehicle, **pose)  # type: ignore[arg-type]
        except (KinematicsError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if html:
        from suspkin.plotting.interactive import interactive_view

        target = out if out is not None else Path(f"{vehicle_yaml.stem}.html")
        try:
            interactive_view(vehicle, state, target)
        except ImportError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(str(target))
        return

    import matplotlib.pyplot as plt

    from suspkin.plotting import plot_vehicle_3d

    if out is not None:
        fig = plot_vehicle_3d(vehicle, state=state, show_labels=labels)
        fig.savefig(out, dpi=200)
        typer.echo(str(out))
        return
    fig = plt.figure(figsize=(9, 6))
    plot_vehicle_3d(vehicle, fig.add_subplot(projection="3d"), state=state, show_labels=labels)
    plt.show()


@app.command()
def animate(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output .gif (or .mp4 with ffmpeg).")],
    heave: Annotated[
        tuple[float, float] | None, typer.Option("--heave", help="Heave range MIN MAX (mm).")
    ] = None,
    roll: Annotated[
        tuple[float, float] | None, typer.Option("--roll", help="Roll range MIN MAX (deg).")
    ] = None,
    pitch: Annotated[
        tuple[float, float] | None, typer.Option("--pitch", help="Pitch range MIN MAX (deg).")
    ] = None,
    rack: Annotated[
        tuple[float, float] | None, typer.Option("--rack", help="Rack range MIN MAX (mm).")
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=2, help="Number of frames.")] = 15,
    fps: Annotated[int, typer.Option("--fps", min=1, help="Frames per second.")] = 15,
) -> None:
    """Animate a body-pose sweep (heave, roll, pitch or rack) of the 3D geometry.

    Heave and rack in MILLIMETRES, roll and pitch in DEGREES. Give exactly one sweep
    variable.
    """
    import numpy as np

    from suspkin.kinematics import KinematicsError
    from suspkin.sweeps import sweep_vehicle

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
    scale = _pose_kwargs(1.0, 1.0, 1.0, 1.0)[name]
    values = np.linspace(lo, hi, steps) * scale
    from suspkin.plotting.animation import animate_sweep

    try:
        states = sweep_vehicle(veh, **{name: values})  # type: ignore[arg-type]
        animate_sweep(veh, states, out, fps=fps)
    except (KinematicsError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(out))


@app.command()
def sweep(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    corner: Annotated[str, typer.Option("--corner", help="Corner: FL, FR, RL or RR.")] = "FL",
    travel: Annotated[
        tuple[float, float] | None,
        typer.Option("--travel", help="Wheel travel range MIN MAX (mm, bump positive)."),
    ] = None,
    rack_sweep: Annotated[
        tuple[float, float] | None,
        typer.Option("--rack-sweep", help="Rack range MIN MAX (mm) at zero travel."),
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=2, help="Number of sweep points.")] = 21,
    rack: Annotated[
        float, typer.Option("--rack", help="Fixed rack offset (mm, travel sweep).")
    ] = 0.0,
    metrics: Annotated[
        bool, typer.Option("--metrics", help="Add metric and gradient columns (>= 3 steps).")
    ] = False,
) -> None:
    """Sweep one corner over wheel travel or rack (inputs in mm); the CSV is in SI units."""
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
            tr = np.linspace(travel[0], travel[1], steps) / 1000.0
            states = sweep_corner(c, tr, rack=rack / 1000.0)
        else:
            assert rack_sweep is not None
            rk = np.linspace(rack_sweep[0], rack_sweep[1], steps) / 1000.0
            states = sweep_rack(c, rk)
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
    from suspkin.plotting.figures import format_value

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
            cells.append(format_value("{:12.4f}", v))
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
    "rc_migration": (
        1e3 / _DEG,
        "mm/deg (+ = toward left of the car centreline, for left-up roll)",
    ),
    "rc_height_curvature": (1e3 / _DEG**2, "mm/deg^2"),
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
    "steering_ratio": (1.0, "deg/deg (handwheel/road wheel)"),
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
    from suspkin.plotting.figures import format_value

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
            val = format_value("{:12.4f}", float(v) * scale)  # type: ignore[arg-type]
            typer.echo(f"{k:<{width}}  {val}  {unit}")
    if json_out is not None:
        json_out.write_text(json.dumps(data, indent=2))
        typer.echo(str(json_out))


@app.command()
def vsweep(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    heave: Annotated[
        tuple[float, float] | None, typer.Option("--heave", help="Heave range MIN MAX (mm).")
    ] = None,
    roll: Annotated[
        tuple[float, float] | None, typer.Option("--roll", help="Roll range MIN MAX (deg).")
    ] = None,
    pitch: Annotated[
        tuple[float, float] | None, typer.Option("--pitch", help="Pitch range MIN MAX (deg).")
    ] = None,
    rack: Annotated[
        tuple[float, float] | None, typer.Option("--rack", help="Rack range MIN MAX (mm).")
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=2, help="Number of sweep points.")] = 21,
    base_heave: Annotated[float, typer.Option("--base-heave", help="Fixed heave (mm).")] = 0.0,
    base_roll: Annotated[float, typer.Option("--base-roll", help="Fixed roll (deg).")] = 0.0,
    base_pitch: Annotated[float, typer.Option("--base-pitch", help="Fixed pitch (deg).")] = 0.0,
    base_rack: Annotated[float, typer.Option("--base-rack", help="Fixed rack (mm).")] = 0.0,
    pivot: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--pivot", help="Body rotation centre X Y Z (mm; default: the CG)."),
    ] = None,
) -> None:
    """Sweep the body pose (heave, roll, pitch or rack) and write a CSV table (SI units).

    Heave, rack and pivot are given in MILLIMETRES and roll and pitch in DEGREES on the
    command line (converted to SI); the CSV is in SI (m, rad). Give exactly one of --heave,
    --roll, --pitch, --rack.
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
    else:
        lo, hi = lo / 1000.0, hi / 1000.0
    base = {
        "heave": base_heave / 1000.0,
        "roll": float(np.radians(base_roll)),
        "pitch": float(np.radians(base_pitch)),
        "rack": base_rack / 1000.0,
    }
    pivot_m = None if pivot is None else tuple(v / 1000.0 for v in pivot)
    base.pop(name)
    try:
        states = sweep_vehicle(veh, **{name: np.linspace(lo, hi, steps)}, base=base, pivot=pivot_m)
        df = vehicle_frame(states, veh)
    except (KinematicsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    df.to_csv(out, index=False)
    typer.echo(str(out))


@app.command()
def report(
    vehicle_yaml: Annotated[Path, typer.Argument(help="Vehicle YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    compare: Annotated[
        list[Path] | None,
        typer.Option("--compare", help="Further vehicle YAML to overlay (repeatable)."),
    ] = None,
    no_3d: Annotated[bool, typer.Option("--no-3d", help="Skip the 3D view and GIF.")] = False,
    formats: Annotated[
        str, typer.Option("--formats", help="Figure file formats, comma separated.")
    ] = "png,svg",
    steps: Annotated[int, typer.Option("--steps", min=3, help="Points per sweep.")] = 31,
) -> None:
    """Run the standard studies and write a report folder (HTML, Markdown, CSV, figures).

    Designs are labelled by their YAML file name. Tables are SI, figures and the HTML
    summary are in mm and deg.
    """
    from suspkin.kinematics import KinematicsError
    from suspkin.report import build_report, run_standard_studies

    paths = [vehicle_yaml, *(compare or [])]
    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    if not fmts:
        typer.echo("error: --formats is empty", err=True)
        raise typer.Exit(code=2)
    try:
        designs = []
        seen: set[str] = set()
        for p in paths:
            label, k = p.stem, 1
            while label in seen:
                k += 1
                label = f"{p.stem}_{k}"
            seen.add(label)
            designs.append((label, run_standard_studies(load_vehicle(p), steps=steps)))
        html_path = build_report(designs, out, formats=fmts, include_3d=not no_3d)
    except VehicleConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (KinematicsError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(html_path))


def _cell(v: float, fmt: str = "{:.3f}") -> str:
    """Format a table number; ``nan`` (no limit, disabled) becomes ``-``."""
    from suspkin.plotting.figures import format_value

    return "-" if v != v else format_value(fmt, float(v)).strip()


def _echo_table(head: tuple[str, ...], rows: Sequence[tuple[str, ...]], right: set[int]) -> None:
    """Print an aligned text table; columns in ``right`` are right-aligned."""
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(head)]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(
            c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(cells)
        ).rstrip()

    typer.echo(line(head))
    for row in rows:
        typer.echo(line(row))


@app.command()
def check(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML file.")],
    csv: Annotated[
        Path | None, typer.Option("--csv", help="Write the constraint table (SI) as CSV.")
    ] = None,
    plot: Annotated[
        Path | None, typer.Option("--plot", help="Write the 3D study view as an image.")
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 1 if the baseline is not feasible.")
    ] = False,
) -> None:
    """Evaluate the baseline vehicle against all constraints of a study.

    Prints one row per constraint (display units); status is PASS, FAIL, UNSOLVED,
    INFEASIBLE or OFF (disabled). Exit code 2 on a study file or output error, 1 with
    --strict if the baseline is not feasible or lies outside a variable domain.
    """
    from suspkin.optimize import DesignSpace, evaluate
    from suspkin.optimize.design import InfeasibleDesign
    from suspkin.optimize.spec import StudyConfigError, load_study

    try:
        study, vehicle = load_study(study_yaml)
        space = DesignSpace(study, vehicle)
    except (StudyConfigError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    baseline_ok = True
    try:
        space.from_vehicle()
    except InfeasibleDesign as exc:
        baseline_ok = False
        typer.echo(f"warning: baseline is outside a variable domain: {exc}", err=True)
    ev = evaluate(space)
    df = ev.display_frame()
    status = [("OFF" if s == "disabled" else s.upper()) for s in df["status"]]
    records: list[Any] = list(df.itertuples(index=False))
    rows = [
        (
            str(r.name),
            str(r.type),
            _cell(float(r.value)),
            _cell(float(r.min)),
            _cell(float(r.max)),
            str(r.unit) or "-",
            _cell(float(r.g), "{:.3f}"),
            st,
            str(r.detail),
        )
        for r, st in zip(records, status, strict=True)
    ]
    head = ("name", "type", "value", "min", "max", "unit", "g", "status", "detail")
    _echo_table(head, rows, {2, 3, 4, 6})  # right-aligned columns
    enabled = [s for s in status if s != "OFF"]
    n_pass = sum(s == "PASS" for s in enabled)
    typer.echo(
        f"{n_pass}/{len(enabled)} constraints pass, feasible: {'yes' if ev.feasible else 'no'}"
    )
    if ev.objectives:
        odf = ev.objectives_display_frame()
        orecords: list[Any] = list(odf.itertuples(index=False))
        orows = [
            (
                str(r.name),
                str(r.type),
                _cell(float(r.raw)),
                str(r.unit) or "-",
                _cell(float(r.f), "{:.3f}"),
                _cell(float(r.weight), "{:g}"),
                str(r.status).upper(),
                str(r.detail),
            )
            for r in orecords
        ]
        typer.echo("")
        _echo_table(
            ("objective", "type", "raw", "unit", "f", "weight", "status", "detail"),
            orows,
            {2, 4, 5},
        )
        typer.echo(f"f_total (weighted sum): {_cell(ev.f_total, '{:.3f}')}")
        opt = study.optimizer
        if opt is not None:
            typer.echo(f"optimizer: mode {opt.mode}, method {opt.method}")
    try:
        if csv is not None:
            csv.parent.mkdir(parents=True, exist_ok=True)
            ev.frame().to_csv(csv, index=False)
            typer.echo(str(csv))
        if plot is not None:
            from suspkin.plotting import plot_study_3d

            plot.parent.mkdir(parents=True, exist_ok=True)
            plot_study_3d(space).savefig(plot, dpi=200)
            typer.echo(str(plot))
    except Exception as exc:
        typer.echo(f"error: cannot write output: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if strict and not (ev.feasible and baseline_ok):
        raise typer.Exit(code=1)


def _compare_table(
    title: str, head: tuple[str, ...], rows: Sequence[tuple[str, ...]], right: set[int]
) -> None:
    typer.echo("")
    typer.echo(title)
    _echo_table(head, rows, right)


@app.command()
def optimize(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with objectives and optimizer.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Run folder (default results/opt/<study>_<timestamp>)."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue the run in --out from its history.csv.")
    ] = False,
    workers: Annotated[
        str | None,
        typer.Option("--workers", help="Worker processes: an integer or auto (default: study)."),
    ] = None,
    evaluations: Annotated[
        int | None, typer.Option("--evaluations", help="Evaluation budget override.")
    ] = None,
    time: Annotated[
        str | None, typer.Option("--time", help='Wall-time budget override, e.g. "30 min".')
    ] = None,
    method: Annotated[
        str | None,
        typer.Option("--method", help="nsga2 | de | slsqp | trust-constr | de+slsqp."),
    ] = None,
    no_report: Annotated[
        bool, typer.Option("--no-report", help="Skip the before/after report.")
    ] = False,
    surrogate: Annotated[
        Path | None,
        typer.Option(
            "--surrogate",
            help="Surrogate folder (dataset/, models/): active-learning optimizer (experimental).",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="auto | cpu | cuda for the networks (needs --surrogate)."),
    ] = None,
) -> None:
    """Optimise a study's design variables against its objectives and constraints.

    Writes a run folder (history.csv, best.yaml, result.json, figures, report). With
    --surrogate the networks of `surrogate train` search a trust region and the FOM verifies
    the infill points (`--evaluations` is then the FOM budget; al.csv logs the iterations).
    Exit code 2 on a study or argument error, 1 if no feasible design was found, 0 otherwise.
    """
    from suspkin.optimize.drivers import RunError, run_optimization
    from suspkin.optimize.history import HistoryError
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.units import parse_quantity

    if surrogate is None and device is not None:
        typer.echo("error: --device needs --surrogate", err=True)
        raise typer.Exit(code=2)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        typer.echo(f"error: --device must be auto, cpu or cuda, got {device!r}", err=True)
        raise typer.Exit(code=2)
    try:
        seconds = None if time is None else parse_quantity(time, "time")
        n_workers: int | str | None = None
        if workers is not None:
            if workers.strip().lower() == "auto":
                n_workers = "auto"
            else:
                n_workers = int(workers)
                if n_workers < 1:
                    raise ValueError("--workers must be >= 1 or auto")
        if surrogate is None:
            res = run_optimization(
                study_yaml,
                out,
                resume=resume,
                workers=n_workers,
                evaluations=evaluations,
                time=seconds,
                method=method,
                report=not no_report,
                echo=typer.echo,
            )
        else:
            from suspkin.surrogate.active import run_active_learning

            active = run_active_learning(
                study_yaml,
                surrogate,
                out,
                resume=resume,
                workers=n_workers,
                evaluations=evaluations,
                time=seconds,
                method=method,
                device=device,
                report=not no_report,
                echo=typer.echo,
            )
            res = active.run
    except (StudyConfigError, RunError, HistoryError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (RuntimeError, ImportError) as exc:
        if surrogate is None:
            raise
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    base, best = res.baseline, res.best
    typer.echo(
        f"done: {res.n_evals} new evaluations ({res.n_total} in history), "
        f"{res.seconds:.1f} s, {res.workers} workers"
    )
    if surrogate is not None:
        typer.echo(
            f"active learning: {active.new_iterations} new iterations ({active.iterations} in "
            f"al.csv), stopped on {active.stop_reason}, device {active.device}"
        )
    if res.mode == "pareto":
        typer.echo(f"Pareto set: {res.n_pareto} designs; table shows the knee point")
    o0: list[Any] = list(base.objectives_display_frame().itertuples(index=False))
    o1: list[Any] = list(best.objectives_display_frame().itertuples(index=False))
    orows = [
        (
            str(a.name),
            _cell(float(a.raw)),
            _cell(float(b.raw)),
            str(a.unit) or "-",
            _cell(float(a.f), "{:.3f}"),
            _cell(float(b.f), "{:.3f}"),
        )
        for a, b in zip(o0, o1, strict=True)
    ]
    _compare_table(
        "objectives",
        ("objective", "baseline", "best", "unit", "f base", "f best"),
        orows,
        {1, 2, 4, 5},
    )
    typer.echo(f"f_total: baseline {base.f_total:.3f} -> best {best.f_total:.3f}")
    c0: list[Any] = list(base.display_frame().itertuples(index=False))
    c1: list[Any] = list(best.display_frame().itertuples(index=False))
    crows = [
        (
            str(a.name),
            _cell(float(a.value)),
            _cell(float(b.value)),
            str(a.unit) or "-",
            _cell(float(a.g), "{:.3f}"),
            _cell(float(b.g), "{:.3f}"),
            "PASS" if b.status == "pass" else str(b.status).upper(),
        )
        for a, b in zip(c0, c1, strict=True)
        if a.status != "disabled"
    ]
    _compare_table(
        "constraints",
        ("constraint", "baseline", "best", "unit", "g base", "g best", "status"),
        crows,
        {1, 2, 4, 5},
    )
    typer.echo(
        f"feasible: baseline {'yes' if base.feasible else 'no'}, "
        f"best {'yes' if best.feasible else 'no'}"
    )
    typer.echo(str(res.out_dir))
    if not res.feasible:
        typer.echo("error: no feasible design found", err=True)
        raise typer.Exit(code=1)


surrogate_app = typer.Typer(help="Neural-network surrogate (experimental).", no_args_is_help=True)
app.add_typer(surrogate_app, name="surrogate")


@surrogate_app.command("sample")
def surrogate_sample(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[
        Path | None, typer.Option("--root", help="Surrogate folder (dataset/ is written here).")
    ] = None,
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    n: Annotated[
        int | None, typer.Option("--n", min=1, help="LHS samples per axle (default: study).")
    ] = None,
    workers: Annotated[
        str | None, typer.Option("--workers", help="Worker processes: an integer or auto.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue an interrupted sampling in --root.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", hidden=True, help="Deprecated alias of --root.")
    ] = None,
) -> None:
    """Sample the initial Latin-hypercube dataset of the surrogate (experimental).

    Writes ROOT/dataset/<axle>/shard_*.npz and meta.json. Exit code 2 on a study or argument
    error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.data_types import Axle
    from suspkin.surrogate.dataset import DatasetError, sample_axle
    from suspkin.surrogate.sampling import NoAxleVariablesError

    if root is None:
        root = out
    elif out is not None and out != root:
        typer.echo("error: give --root only (--out is a deprecated alias)", err=True)
        raise typer.Exit(code=2)
    if root is None:
        typer.echo("error: missing option --root", err=True)
        raise typer.Exit(code=2)
    axles: list[Axle]
    if axle == "both":
        axles = ["front", "rear"]
    elif axle == "front":
        axles = ["front"]
    elif axle == "rear":
        axles = ["rear"]
    else:
        typer.echo(f"error: --axle must be front, rear or both, got {axle!r}", err=True)
        raise typer.Exit(code=2)
    try:
        n_workers: int | str | None = None
        if workers is not None:
            if workers.strip().lower() == "auto":
                n_workers = "auto"
            else:
                n_workers = int(workers)
                if n_workers < 1:
                    raise ValueError("--workers must be >= 1 or auto")
        done = 0
        for ax in axles:
            try:
                rep = sample_axle(
                    study_yaml, root, ax, n=n, workers=n_workers, resume=resume, echo=typer.echo
                )
            except NoAxleVariablesError as exc:
                if axle != "both":
                    raise
                typer.echo(f"{ax}: skipped ({exc})")
                continue
            done += 1
            typer.echo(
                f"{ax}: {rep.n_samples} samples ({rep.n_new} new), "
                f"feasible {100 * rep.feasible_fraction:.1f} %, {rep.seconds:.1f} s"
            )
            if rep.warning:
                typer.echo(f"warning: {rep.warning}", err=True)
        if done == 0:
            raise StudyConfigError("no axle to sample")
    except (StudyConfigError, DatasetError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _parse_axles(axle: str) -> list[str]:
    if axle == "both":
        return ["front", "rear"]
    if axle in ("front", "rear"):
        return [axle]
    typer.echo(f"error: --axle must be front, rear or both, got {axle!r}", err=True)
    raise typer.Exit(code=2)


def _present_axles(root: Path, axles: list[str], axle: str) -> list[Any]:
    """Axles of ``axles`` that have a dataset (``both`` skips the missing ones)."""
    have = [a for a in axles if (root / "dataset" / a / "meta.json").is_file()]
    if axle != "both":
        missing = [a for a in axles if a not in have]
        if missing:
            typer.echo(f"error: no dataset for {missing[0]} in {root / 'dataset'}", err=True)
            raise typer.Exit(code=2)
    elif not have:
        typer.echo(
            f"error: no dataset in {root / 'dataset'}; run `surrogate sample` first", err=True
        )
        raise typer.Exit(code=2)
    return list(have)


@surrogate_app.command("train")
def surrogate_train(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[Path, typer.Option("--root", help="Surrogate folder (holds dataset/).")],
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    trials: Annotated[
        int | None, typer.Option("--trials", min=1, help="Optuna trials (default: study).")
    ] = None,
    timeout: Annotated[
        str | None,
        typer.Option("--timeout", help="Search time budget, e.g. '2 h' (default: study)."),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | cuda (default: study).")
    ] = None,
) -> None:
    """Search the architecture and train the networks (experimental).

    Writes ROOT/models/<axle>/ (optuna.db, trials.csv, model/). Exit code 2 on a study,
    argument or data error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.dataset import DatasetError
    from suspkin.surrogate.workflow import train_surrogate
    from suspkin.units import parse_quantity

    axles = _parse_axles(axle)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        typer.echo(f"error: --device must be auto, cpu or cuda, got {device!r}", err=True)
        raise typer.Exit(code=2)
    try:
        tmo = None if timeout is None else parse_quantity(timeout, "time")
        if tmo is not None and tmo <= 0:
            raise ValueError("--timeout must be positive")
        use = _present_axles(root, axles, axle)
        train_surrogate(
            study_yaml,
            root,
            axles=use,
            n_trials=trials,
            timeout=tmo,
            device=device,
            echo=typer.echo,
        )
    except (StudyConfigError, DatasetError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@surrogate_app.command("validate")
def surrogate_validate(
    study_yaml: Annotated[Path, typer.Argument(help="Study YAML with a surrogate block.")],
    root: Annotated[Path, typer.Option("--root", help="Surrogate folder (dataset/, models/).")],
    axle: Annotated[str, typer.Option("--axle", help="front | rear | both.")] = "both",
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | cuda (default: study).")
    ] = None,
) -> None:
    """Test-split error per channel against its tolerance (experimental).

    Writes ROOT/validation/<axle>/ (CSV tables and figures). Exit code 1 when a channel is
    over its tolerance, 2 on a study, argument or data error.
    """
    from suspkin.optimize.spec import StudyConfigError
    from suspkin.surrogate.dataset import DatasetError
    from suspkin.surrogate.workflow import validate_surrogate, validation_table

    axles = _parse_axles(axle)
    if device is not None and device not in ("auto", "cpu", "cuda"):
        typer.echo(f"error: --device must be auto, cpu or cuda, got {device!r}", err=True)
        raise typer.Exit(code=2)
    try:
        use = _present_axles(root, axles, axle)
        reports = validate_surrogate(study_yaml, root, axles=use, device=device, echo=typer.echo)
    except (StudyConfigError, DatasetError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    n_over = 0
    for ax, rep in reports.items():
        tab = validation_table(rep)
        n_over += int((~tab["passed"]).sum())
        typer.echo(
            f"{ax}: {rep.n} test samples, score {rep.score:.3f}, validity accuracy "
            f"{100 * rep.validity_accuracy:.1f} %"
        )
        typer.echo(f"  {'channel':<26}{'rmse':>11}{'tol':>11}  {'unit':<5}{'ratio':>8}  status")
        for row in tab.itertuples(index=False):
            typer.echo(
                f"  {row.channel:<26}{row.rmse:>11.4g}{row.tol:>11.4g}  {row.unit:<5}"
                f"{row.ratio:>8.2f}  {'ok' if row.passed else 'OVER'}"
            )
        typer.echo(str(root / "validation" / ax))
    if n_over:
        typer.echo(f"error: {n_over} channel(s) over tolerance", err=True)
        raise typer.Exit(code=1)
