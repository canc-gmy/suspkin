"""Shared look of the report figures (matplotlib, no global state).

Colour encodes the design (overlay index, Okabe-Ito colour-blind-safe palette), the line
style encodes the side: solid = left / front, dashed = right / rear (dotted and dash-dot
for the third and fourth series of a panel). Use :func:`use_style` as a context manager;
it never mutates the global ``rcParams``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import matplotlib as mpl
from matplotlib.axes import Axes

# Okabe-Ito (colour-blind safe), black last so it is only used for 8+ designs.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

# Line style per series index within a panel: left/front, right/rear, then extras.
SIDE_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")

# Figure widths in inches (A4 portrait text width ~ 6.3 in, landscape ~ 10 in).
_WIDTH_A4 = 6.3
_WIDTH_A4_LANDSCAPE = 10.0

FIGURE_PRESETS: dict[str, tuple[float, float]] = {
    "a4_width": (_WIDTH_A4, 4.2),
    "a4_tall": (_WIDTH_A4, 8.0),
    "a4_landscape": (_WIDTH_A4_LANDSCAPE, 6.5),
    "wide": (_WIDTH_A4_LANDSCAPE, 4.5),
}

RC_PARAMS: dict[str, Any] = {
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "font.size": 8,
}


def design_color(index: int) -> str:
    """Palette colour of design ``index`` (wraps around)."""
    return PALETTE[index % len(PALETTE)]


def side_linestyle(index: int) -> str:
    """Line style of series ``index`` within a panel (0 = left / front, 1 = right / rear)."""
    return SIDE_LINESTYLES[index % len(SIDE_LINESTYLES)]


def figure_size(preset: str = "a4_width", rows: int | None = None) -> tuple[float, float]:
    """Figure size in inches for a named preset; ``rows`` scales the height (2.3 in/row).

    Args:
        preset: Key of ``FIGURE_PRESETS``.
        rows: Number of panel rows; ``None`` keeps the preset height.

    Returns:
        ``(width, height)`` in inches.

    Raises:
        ValueError: Unknown preset.
    """
    if preset not in FIGURE_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(FIGURE_PRESETS)}")
    w, h = FIGURE_PRESETS[preset]
    return (w, 2.3 * rows + 0.6) if rows is not None else (w, h)


def axis_label(name: str, unit: str = "") -> str:
    """``"name [unit]"`` (or just ``name`` for a dimensionless quantity).

    Args:
        name: Quantity name.
        unit: Display unit; empty for dimensionless.

    Returns:
        Axis label text.
    """
    return f"{name} [{unit}]" if unit else name


def zero_lines(ax: Axes, x: bool = True, y: bool = True) -> None:
    """Thin reference lines through the static position (x = 0, y = 0).

    The lines carry ``gid="zero"`` so that tests and callers can tell them from data.

    Args:
        ax: Axes to draw on.
        x: Draw the vertical line at x = 0.
        y: Draw the horizontal line at y = 0.
    """
    kw: dict[str, Any] = {"color": "#8c8c8c", "lw": 0.7, "zorder": 0.5, "gid": "zero"}
    if y:
        ax.axhline(0.0, **kw)
    if x:
        ax.axvline(0.0, **kw)


@contextlib.contextmanager
def use_style() -> Iterator[None]:
    """Apply the report style to everything drawn inside the ``with`` block."""
    with mpl.rc_context(RC_PARAMS):  # type: ignore[arg-type]
        yield
