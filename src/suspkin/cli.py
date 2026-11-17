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
