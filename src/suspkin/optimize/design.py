"""Design space: map a normalised vector ``x`` in [0, 1]^n to a :class:`Vehicle`.

Each study variable contributes parameters in the unit cube:

* hexahedron / box domain: three parameters ``<label>.u``, ``.v``, ``.w``;
* offset domain: one parameter per given axis, named ``<label>.x`` / ``.y`` / ``.z``. Axes
  that are not given stay fixed at the baseline and are excluded
  from ``n`` and ``names``.

Variable labels: a symmetric or ``side: left`` variable is labelled with its point ref
(``front.uca_front``); an asymmetric ``side: right`` variable gets the suffix ``@right``.

Symmetric variables move the left hardpoint (the right side mirrors it). An asymmetric
left variable moves the left hardpoint and pins the right one (baseline mirrored value
written into ``right_overrides`` unless an override exists); an asymmetric right variable
writes ``right_overrides``. Regions of right variables are in right-side coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import ValidationError
from scipy.optimize import least_squares

from suspkin.geometry.models import Vehicle
from suspkin.optimize.regions import hex_inverse, hex_map
from suspkin.optimize.spec import (
    BoxRegion,
    HexahedronRegion,
    OffsetRegion,
    Study,
    Variable,
    parse_point_ref,
)

F64 = NDArray[np.float64]
_X_TOL = 1e-12
_INV_TOL = 1e-9
_AXES = ("x", "y", "z")


class InfeasibleDesign(ValueError):  # noqa: N818
    """The design vector produces an invalid vehicle, or a point lies outside its domain."""


@dataclass(frozen=True)
class _Slot:
    var: Variable
    axle: Literal["front", "rear"]
    name: str
    mode: Literal["sym", "left", "right"]
    label: str
    axes: tuple[int, ...]  # free coordinate axes (offset) or (0, 1, 2)
    names: tuple[str, ...]
    offset: int  # start index in x

    @property
    def n(self) -> int:
        return len(self.axes)


def _mirror(p: ArrayLike) -> F64:
    a = np.array(p, dtype=np.float64)
    a[1] = -a[1]
    return a


def _get(vehicle: Vehicle, axle: str, name: str) -> F64:
    v = getattr(getattr(vehicle, axle).hardpoints, name)
    if v is None:
        raise ValueError(f"point {axle}.{name} does not exist in the vehicle")
    return np.array(v, dtype=np.float64)


def _point_of(vehicle: Vehicle, slot: _Slot) -> F64:
    """Position of the variable's point in ``vehicle`` (right-side coordinates for ``right``)."""
    left = _get(vehicle, slot.axle, slot.name)
    if slot.mode != "right":
        return left
    ov = getattr(vehicle, slot.axle).right_overrides
    return np.array(ov[slot.name], dtype=np.float64) if slot.name in ov else _mirror(left)


class DesignSpace:
    """Normalised parameterisation of the study variables around a baseline vehicle."""

    def __init__(self, study: Study, vehicle: Vehicle) -> None:
        self.study = study
        self.baseline: Vehicle = vehicle
        slots: list[_Slot] = []
        off = 0
        for var in study.variables:
            axle, name = parse_point_ref(var.point)
            if getattr(getattr(vehicle, axle).hardpoints, name) is None:
                raise ValueError(f"point {var.point!r} does not exist in the vehicle")
            if not var.asymmetric and name in getattr(vehicle, axle).right_overrides:
                raise ValueError(
                    f"symmetric variable on point with right override: {var.point!r} "
                    f"(declare it 'asymmetric: true' or drop the override)"
                )
            mode: Literal["sym", "left", "right"] = (
                "sym" if not var.asymmetric else var.effective_side
            )
            label = f"{var.point}@right" if mode == "right" else var.point
            d = var.domain
            if isinstance(d, OffsetRegion):
                axes = tuple(
                    i
                    for i, rng in enumerate((d.x, d.y, d.z))
                    if rng is not None and rng[1] > rng[0]
                )
                if not axes:
                    raise ValueError(f"variable {var.point!r}: offset domain has no free axis")
                names = tuple(f"{label}.{_AXES[i]}" for i in axes)
            else:
                axes = (0, 1, 2)
                names = tuple(f"{label}.{c}" for c in "uvw")
            slots.append(_Slot(var, axle, name, mode, label, axes, names, off))
            off += len(axes)
        self._slots: tuple[_Slot, ...] = tuple(slots)
        self.n: int = off
        self.names: tuple[str, ...] = tuple(nm for s in slots for nm in s.names)
        self.bounds: F64 = np.tile(np.array([0.0, 1.0]), (self.n, 1))

    # ---- mapping -------------------------------------------------------------------

    def _map(self, slot: _Slot, u: F64) -> F64:
        d = slot.var.domain
        if isinstance(d, HexahedronRegion):
            return np.asarray(hex_map(np.array(d.vertices), u), dtype=np.float64)
        if isinstance(d, BoxRegion):
            lo, hi = np.array(d.lo), np.array(d.hi)
            return np.asarray(lo + u * (hi - lo), dtype=np.float64)
        ranges = (d.x, d.y, d.z)
        p = _point_of(self.baseline, slot).copy()
        for k, ax in enumerate(slot.axes):
            rng = ranges[ax]
            assert rng is not None
            p[ax] += rng[0] + u[k] * (rng[1] - rng[0])
        return p

    def _unmap(self, slot: _Slot, p: F64, clip: bool) -> F64:
        d = slot.var.domain
        outside = f"{slot.label}: point {p.tolist()} lies outside its domain"
        if isinstance(d, HexahedronRegion):
            verts = np.array(d.vertices)
            try:
                return hex_inverse(verts, p)
            except ValueError as exc:
                if not clip:
                    raise InfeasibleDesign(f"{outside} ({exc})") from exc
                sol = least_squares(
                    lambda u: np.asarray(hex_map(verts, u)) - p,
                    x0=np.full(3, 0.5),
                    bounds=(0.0, 1.0),
                    xtol=1e-14,
                    ftol=1e-14,
                    gtol=1e-14,
                )
                clipped: F64 = np.clip(sol.x, 0.0, 1.0)
                return clipped
        if isinstance(d, BoxRegion):
            lo, hi = np.array(d.lo), np.array(d.hi)
            u = (p - lo) / (hi - lo)
        else:
            base = _point_of(self.baseline, slot)
            ranges = (d.x, d.y, d.z)
            u = np.empty(len(slot.axes))
            for k, ax in enumerate(slot.axes):
                rng = ranges[ax]
                assert rng is not None
                u[k] = (p[ax] - base[ax] - rng[0]) / (rng[1] - rng[0])
            if not clip:
                for a in range(3):
                    if a in slot.axes:
                        continue
                    rng = ranges[a] or (0.0, 0.0)
                    if not rng[0] - _INV_TOL <= p[a] - base[a] <= rng[1] + _INV_TOL:
                        raise InfeasibleDesign(f"{outside} (fixed axis {_AXES[a]} moved)")
        if not clip and (np.any(u < -_INV_TOL) or np.any(u > 1.0 + _INV_TOL)):
            raise InfeasibleDesign(f"{outside} (parameters {np.round(u, 6).tolist()})")
        out: F64 = np.clip(u, 0.0, 1.0)
        return out

    def _check_x(self, x: ArrayLike) -> F64:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must be finite")
        if np.any(arr < -_X_TOL) or np.any(arr > 1.0 + _X_TOL):
            raise ValueError("x must lie within [0, 1]")
        out: F64 = np.clip(arr, 0.0, 1.0)
        return out

    # ---- public API ----------------------------------------------------------------

    def points(self, x: ArrayLike) -> dict[str, F64]:
        """Variable label -> SI coordinates of the point for ``x``."""
        xa = self._check_x(x)
        return {s.label: self._map(s, xa[s.offset : s.offset + s.n]) for s in self._slots}

    def to_vehicle(self, x: ArrayLike) -> Vehicle:
        """Build the vehicle for ``x``; raises :class:`InfeasibleDesign` if it is invalid."""
        pts = self.points(x)
        d = self.baseline.model_dump()
        # left-side writes first, then right-side overrides
        for s in sorted(self._slots, key=lambda s: s.mode == "right"):
            p = tuple(float(c) for c in pts[s.label])
            ax = d[s.axle]
            if s.mode == "right":
                ax["right_overrides"][s.name] = p
                continue
            if s.mode == "left" and s.name not in ax["right_overrides"]:
                old = _get(self.baseline, s.axle, s.name)
                ax["right_overrides"][s.name] = tuple(float(c) for c in _mirror(old))
            ax["hardpoints"][s.name] = p
        try:
            return Vehicle.model_validate(d)
        except ValidationError as exc:
            raise InfeasibleDesign(str(exc)) from exc

    def from_vehicle(self, v: Vehicle | None = None, *, clip: bool = False) -> F64:
        """Parameter vector of ``v`` (default: baseline).

        Raises :class:`InfeasibleDesign` if a point lies outside its domain, unless
        ``clip=True`` (then parameters are clipped to [0, 1]).
        """
        veh = self.baseline if v is None else v
        out = np.empty(self.n)
        for s in self._slots:
            out[s.offset : s.offset + s.n] = self._unmap(s, _point_of(veh, s), clip)
        return out
