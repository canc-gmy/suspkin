"""Command-line interface for suspkin.

Commands live in group modules; registration order here fixes the ``--help`` order.
"""

import typer

from suspkin import __version__
from suspkin.cli import app as _app_cmds
from suspkin.cli import study, surrogate, vehicle

app = typer.Typer(help="suspkin - suspension kinematics toolkit.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """suspkin command group."""


@app.command()
def version() -> None:
    """Print the installed suspkin version."""
    typer.echo(f"suspkin {__version__}")


app.command()(vehicle.show)
app.command()(vehicle.animate)
app.command()(vehicle.sweep)
app.command()(vehicle.summary)
app.command()(vehicle.vehicle)
app.command()(vehicle.vsweep)
app.command()(vehicle.report)
app.command()(study.check)
app.command()(study.optimize)
app.command("app")(_app_cmds.app_cmd)
app.command("app-export")(_app_cmds.app_export_cmd)

surrogate_app = typer.Typer(help="Neural-network surrogate (experimental).", no_args_is_help=True)
app.add_typer(surrogate_app, name="surrogate")
surrogate_app.command("sample")(surrogate.surrogate_sample)
surrogate_app.command("train")(surrogate.surrogate_train)
surrogate_app.command("validate")(surrogate.surrogate_validate)
