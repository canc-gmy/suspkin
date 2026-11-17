"""Unit conversions and quantities with units (study YAML).

Internals use SI; mm and deg are for display. :func:`parse_quantity` reads strings such as
``"30 mm"`` into SI floats and checks the dimension.

A dimension is a product of base atoms with integer exponents. Its canonical name is built
by ``_dim_name`` (``"length"``, ``"angle/length"``, ``"1/length"``, ``"force*length/angle"``);
``"ratio"`` is the dimensionless one (``deg/deg`` and ``mm/mm`` are both ``ratio``).
``"percent"`` is a separate atom whose SI value is the percent number itself (50 % -> 50.0),
as for the percent metrics of the package. ``"gload"`` is acceleration in multiples of g
(``deg/g`` for the roll gradient, whose SI value is rad per g); ``"accel"`` is m/s^2.
"""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

ArrayLike = float | npt.NDArray[np.float64]

Dim = str  # canonical dimension name, see the module docstring


def m_to_mm(x: ArrayLike) -> ArrayLike:
    """Convert metres to millimetres."""
    return np.multiply(x, 1000.0)


def mm_to_m(x: ArrayLike) -> ArrayLike:
    """Convert millimetres to metres."""
    return np.multiply(x, 0.001)


def rad_to_deg(x: ArrayLike) -> ArrayLike:
    """Convert radians to degrees."""
    return np.degrees(x)


def deg_to_rad(x: ArrayLike) -> ArrayLike:
    """Convert degrees to radians."""
    return np.radians(x)


_ATOMS: tuple[str, ...] = ("length", "angle", "force", "accel", "gload", "mass", "percent")
_Exps = tuple[int, ...]

_DEG = float(np.pi / 180.0)
# unit symbol -> (SI factor, atom)
_UNITS: dict[str, tuple[float, str]] = {
    "m": (1.0, "length"),
    "cm": (0.01, "length"),
    "mm": (0.001, "length"),
    "rad": (1.0, "angle"),
    "deg": (_DEG, "angle"),
    "N": (1.0, "force"),
    "kN": (1000.0, "force"),
    "mps2": (1.0, "accel"),
    "g": (1.0, "gload"),
    "kg": (1.0, "mass"),
    "%": (1.0, "percent"),
}
# atom -> (display unit, SI -> display factor)
_DISPLAY: dict[str, tuple[str, float]] = {
    "length": ("mm", 1000.0),
    "angle": ("deg", 1.0 / _DEG),
    "force": ("N", 1.0),
    "accel": ("m/s2", 1.0),
    "gload": ("g", 1.0),
    "mass": ("kg", 1.0),
    "percent": ("%", 1.0),
}
_ALIASES: dict[str, str] = {"stiffness": "force/length", "dimensionless": "ratio"}
# composite dimensions whose display unit is not the product of the atom displays
_DISPLAY_OVERRIDE: dict[str, tuple[str, float]] = {
    "force*length/angle": ("N*m/deg", _DEG),
}
_NUM_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def _factors(expr: str, table: dict[str, tuple[float, str]]) -> tuple[float, _Exps]:
    """Factor and exponent vector of a product/quotient such as ``N*m/rad^2``.

    ``table`` maps a symbol to (factor, atom); atom names themselves are also accepted.
    """
    text = re.sub(r"(?<![A-Za-z])m/s\^?2(?![A-Za-z0-9])", "mps2", expr.replace(" ", ""))
    exps = [0] * len(_ATOMS)
    factor = 1.0
    if text in ("", "1", "ratio"):
        return 1.0, tuple(exps)
    head, *tails = text.split("/")
    parts = [(t, 1) for t in head.split("*")]
    parts += [(t, -1) for tail in tails for t in tail.split("*")]
    for tok, sign in parts:
        if tok == "1":
            continue
        name, _, p = tok.partition("^")
        power = sign * (int(p) if p else 1)
        if name in table:
            f, atom = table[name]
        elif name in _ATOMS:
            f, atom = 1.0, name
        else:
            raise ValueError(f"unknown unit or dimension {name!r} in {expr!r}")
        factor *= f**power
        exps[_ATOMS.index(atom)] += power
    return factor, tuple(exps)


def _dim_name(exps: _Exps) -> Dim:
    if not any(exps):
        return "ratio"

    def term(i: int, e: int) -> str:
        return _ATOMS[i] if abs(e) == 1 else f"{_ATOMS[i]}^{abs(e)}"

    num = "*".join(term(i, e) for i, e in enumerate(exps) if e > 0)
    den = "*".join(term(i, e) for i, e in enumerate(exps) if e < 0)
    if not den:
        return num
    return f"{num or '1'}/{den}"


def _dim_exps(dim: Dim) -> _Exps:
    try:
        return _factors(_ALIASES.get(dim, dim), {})[1]
    except ValueError as exc:
        raise ValueError(f"unknown dimension {dim!r}") from exc


def normalize_dim(dim: Dim) -> Dim:
    """Canonical name of a dimension (accepts aliases such as ``"stiffness"``)."""
    return _dim_name(_dim_exps(dim))


def parse_quantity(v: str | float | int, dim: Dim) -> float:
    """Parse a value with optional unit into an SI float of dimension ``dim``.

    A plain number is already SI. Strings look like ``"30 mm"``, ``"-1.5 deg"``,
    ``"0.5 deg/deg"``, ``"1.2 deg/g"``, ``"34 N/mm"``, ``"50 %"``.

    Raises
    ------
    ValueError
        On unparsable text, unknown units, or a unit of the wrong dimension (the
        message names the expected dimension and the unit given).
    """
    if isinstance(v, bool):
        raise ValueError(f"expected a number or a string with unit, got {v!r}")
    if isinstance(v, int | float):
        return float(v)
    if not isinstance(v, str):
        raise ValueError(f"expected a number or a string with unit, got {v!r}")
    m = _NUM_RE.match(v)
    if m is None:
        raise ValueError(f"cannot parse quantity {v!r}")
    num, unit = float(m.group(1)), m.group(2)
    if unit == "":
        return num
    want = _dim_exps(dim)
    label = normalize_dim(dim)
    try:
        factor, exps = _factors(unit, _UNITS)
    except ValueError as exc:
        raise ValueError(f"{exc}; expected a unit of dimension {label!r} in {v!r}") from exc
    if exps != want:
        raise ValueError(
            f"unit {unit!r} in {v!r} has dimension {_dim_name(exps)!r}, "
            f"expected {label!r} (display unit {display_unit(dim) or 'none'!r})"
        )
    return num * factor


def _display(dim: Dim) -> tuple[str, float]:
    name = normalize_dim(dim)
    if name in _DISPLAY_OVERRIDE:
        return _DISPLAY_OVERRIDE[name]
    factor = 1.0
    num: list[str] = []
    den: list[str] = []
    for atom, e in zip(_ATOMS, _dim_exps(name), strict=True):
        if e == 0:
            continue
        sym, f = _DISPLAY[atom]
        factor *= f**e
        (num if e > 0 else den).append(sym if abs(e) == 1 else f"{sym}^{abs(e)}")
    unit = "*".join(num) or ("1" if den else "")
    if den:
        unit += "/" + "*".join(den)
    return unit, factor


def display_unit(dim: Dim) -> str:
    """Display unit of a dimension (``mm``, ``deg``, ``N/mm``, ``%``; ``""`` for a ratio)."""
    return _display(dim)[0]


def display_factor(dim: Dim) -> float:
    """Factor converting an SI value of ``dim`` to its display unit."""
    return _display(dim)[1]


def format_quantity(v_si: float, dim: Dim, digits: int = 4) -> str:
    """Format an SI value in display units, e.g. ``0.03 -> "30 mm"``."""
    unit, factor = _display(dim)
    txt = f"{v_si * factor:.{digits}g}"
    return f"{txt} {unit}" if unit else txt


def slope_dim(metric_dim: Dim, input_dim: Dim) -> Dim:
    """Dimension of ``d(metric)/d(input)``, e.g. ``angle/length`` or ``ratio``."""
    a, b = _dim_exps(metric_dim), _dim_exps(input_dim)
    return _dim_name(tuple(x - y for x, y in zip(a, b, strict=True)))
