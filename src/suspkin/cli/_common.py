"""Shared helpers for the CLI: error exits and text tables."""

from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import typer

from suspkin.geometry import Vehicle, VehicleConfigError, load_vehicle


def fail(msg: str, code: int = 1) -> NoReturn:
    """Print ``msg`` to stderr and exit with ``code``.

    Args:
        msg: Message for stderr.
        code: Process exit code.
    """
    typer.echo(msg, err=True)
    raise typer.Exit(code=code)


def load_vehicle_or_exit(path: Path) -> Vehicle:
    """Load a vehicle YAML; on a config error print it to stderr and exit with code 1."""
    try:
        return load_vehicle(path)
    except VehicleConfigError as exc:
        fail(str(exc))


def _cell(v: float, fmt: str = "{:.3f}") -> str:
    """Format a table number; ``nan`` (no limit, disabled) becomes ``-``."""
    from suspkin.plotting.units import format_value

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
