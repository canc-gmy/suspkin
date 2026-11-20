"""Shared text-table helpers for CLI output."""

from collections.abc import Sequence

import typer


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
