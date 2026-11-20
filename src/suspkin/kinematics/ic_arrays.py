"""Instant centres and swing angles as arrays over N states (numpy only).

Vectorised twins of ``metrics.corner._ic`` / ``_angle`` / ``_roll_line``; shared by
:mod:`suspkin.metrics.arrays` and :mod:`suspkin.kinematics.batch`.

Front view = ``(y, z)`` plane (``axis=0``), side view = ``(x, z)`` plane (``axis=1``). An
instant centre at infinity (parallel section lines) gives NaN coordinates and a kept common
direction; nothing raises. Sign conventions: ``docs/concepts/conventions.md`` section 5.4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.vectors import dot_rows, norm_rows

Arr = NDArray[np.float64]
_NAN = float("nan")
_PARALLEL_TOL = 1e-9


class Ic:
    """Instant centres of N states in one view (see ``corner._ic``).

    Attributes:
        h: Horizontal IC coordinate in m (``y`` in front view, ``x`` in side view), ``(N,)``;
            NaN at infinity or for degenerate input.
        z: Vertical IC coordinate in m, ``(N,)``; NaN as ``h``.
        finite: Boolean ``(N,)``; True where the IC exists.
        d0: Horizontal component of the unit direction of the IC line (pointing to the IC
            at infinity), ``(N,)``.
        d1: Vertical component of that direction, ``(N,)``.
    """

    def __init__(self, h: Arr, z: Arr, finite: NDArray[np.bool_], d0: Arr, d1: Arr) -> None:
        self.h, self.z, self.finite, self.d0, self.d1 = h, z, finite, d0, d1


# kept separate: float ordering (scalar twin: ``corner._ic``)
def ic_arrays(pts: Callable[[str], Arr], axis: int, cut: Arr) -> Ic:
    """Plane-method instant centre for every state; ``axis`` 0 = front view, 1 = side view.

    Each wishbone plane (two chassis pivots and the outer ball joint) is cut with the plane
    ``coord[axis] == cut`` and the two section lines are intersected.

    Args:
        pts: Callable returning point ``name`` as ``(N, 3)`` m (needs ``{uca,lca}_{front,
            rear,outer}``).
        axis: Index of the cut coordinate: 0 (front view, cut ``x``) or 1 (side view, cut ``y``).
        cut: Cut coordinates in m, ``(N,)`` (the contact-patch x or y).

    Returns:
        :class:`Ic` with coordinates, finiteness flag and line directions.
    """
    keep = [i for i in range(3) if i != axis]
    n = cut.shape[0]
    bad = np.zeros(n, dtype=bool)
    n2s: list[Arr] = []
    offs: list[Arr] = []
    for w in ("uca", "lca"):
        a = pts(f"{w}_front")
        nv = np.cross(pts(f"{w}_rear") - a, pts(f"{w}_outer") - a)
        nn = norm_rows(nv)
        bad |= nn < 1e-12
        nv = nv / nn[:, None]
        n2 = nv[:, keep]
        norm2 = norm_rows(n2)
        bad |= norm2 < 1e-12
        rhs = dot_rows(nv, a) - nv[:, axis] * cut
        n2s.append(n2 / norm2[:, None])
        offs.append(rhs / norm2)
    n1, n2 = n2s
    d0 = n1[:, 1].copy()
    d1 = -n1[:, 0]
    d0[bad] = _NAN
    d1[bad] = _NAN
    det = n1[:, 0] * n2[:, 1] - n1[:, 1] * n2[:, 0]
    parallel = np.abs(det) < _PARALLEL_TOL * (norm_rows(n1) * norm_rows(n2))
    hit = ~bad & ~parallel
    h = np.full(n, _NAN)
    z = np.full(n, _NAN)
    if hit.any():
        mats = np.stack([n1[hit], n2[hit]], axis=1)
        rhs2 = np.stack([offs[0][hit], offs[1][hit]], axis=1)[..., None]
        sol = np.linalg.solve(mats, rhs2)[..., 0]
        h[hit] = sol[:, 0]
        z[hit] = sol[:, 1]
    return Ic(h, z, hit, d0, d1)


def swing_angle(ic: Ic, ref_h: Arr, ref_z: Arr, sign: float) -> Arr:
    """Vectorised ``corner._angle``: signed angle of the line reference point -> IC.

    ``alpha = atan2(z_ic - ref_z, sign * (h_ic - ref_h))``, in ``(-pi, pi]``; positive means
    the IC is above the reference point. For an IC at infinity the line direction is oriented
    to the ``sign`` side.

    Args:
        ic: Instant centres of the N states.
        ref_h: Reference horizontal coordinate in m, ``(N,)``.
        ref_z: Reference height in m, ``(N,)``.
        sign: +1 or -1, selecting the positive horizontal direction.

    Returns:
        Angles in rad, ``(N,)``.
    """
    fin = np.arctan2(ic.z - ref_z, sign * (ic.h - ref_h))
    dh = ic.d0 * sign
    dz = ic.d1
    flip = dh < 0
    dh = np.where(flip, -dh, dh)
    dz = np.where(flip, -dz, dz)
    out: Arr = np.where(ic.finite, fin, np.arctan2(dz, dh))
    return out


def roll_line_arrays(cp_y: Arr, cp_z: Arr, ic: Ic) -> tuple[Arr, Arr]:
    """Direction ``(d_y, d_z)`` of the contact patch -> FVIC line (vectorised ``_roll_line``).

    Args:
        cp_y: Contact-patch y in m, ``(N,)``.
        cp_z: Contact-patch z in m, ``(N,)``.
        ic: Front-view instant centres.

    Returns:
        Unit direction components ``(d_y, d_z)``, each ``(N,)`` (the IC direction for an IC
        at infinity, NaN where undefined).
    """
    dy = ic.h - cp_y
    dz = ic.z - cp_z
    nrm = np.sqrt(dy * dy + dz * dz)
    return (
        np.where(ic.finite, dy / nrm, ic.d0),
        np.where(ic.finite, dz / nrm, ic.d1),
    )
