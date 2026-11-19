"""Optional numba fast path of the corner solver (pure-Python fallback when unavailable).

numba is imported lazily on first use. ``SUSPKIN_NO_NUMBA`` (any value except empty or
``0``) disables the fast path. The kernels live in ``_nb_kernels``; this module packs the
static geometry into flat arrays and wraps the kernel calls.
"""

from __future__ import annotations

import importlib
import os
import warnings
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from suspkin.kinematics.corner import _Ctx

Vec = NDArray[np.float64]
_V3 = tuple[float, float, float]

_mod: ModuleType | None = None
_tried = False
_enabled: bool | None = None  # cached decision of ``available()``
_MOUNT = {"upright": 0.0, "lca": 1.0, "uca": 2.0}


def _warmup(k: ModuleType) -> None:
    """Compile and call both kernels on dummy inputs (they return a failure status)."""
    geom, rk = np.zeros(45), np.zeros(15)
    k.newton(geom, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k.ground(geom, rk, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _disable(reason: str) -> None:
    """Switch the fast path off for this process and say why (once)."""
    global _mod, _enabled
    if _mod is not None or _enabled is not False:
        warnings.warn(
            f"suspkin: numba fast path disabled, using pure Python ({reason})",
            RuntimeWarning,
            stacklevel=3,
        )
    _mod = None
    _enabled = False


def kernels() -> ModuleType | None:
    """Return the kernel module, importing numba on first call (None if unavailable).

    A missing numba is silent; any other failure (import, compile, warm-up) disables the
    fast path for the process with a ``RuntimeWarning``.
    """
    global _mod, _tried
    if not _tried:
        _tried = True
        try:
            k = importlib.import_module("suspkin.kinematics._nb_kernels")
        except ImportError:
            return None
        except Exception as exc:
            _disable(f"import failed: {exc!r}")
            return None
        try:
            _warmup(k)
        except Exception as exc:
            _disable(f"compilation failed: {exc!r}")
            return None
        _mod = k
    return _mod


def reset() -> None:
    """Forget the cached decision so ``SUSPKIN_NO_NUMBA`` is read again (for tests)."""
    global _enabled
    _enabled = None


def available() -> bool:
    """True if numba works and ``SUSPKIN_NO_NUMBA`` does not disable it (cached)."""
    global _enabled
    if _enabled is None:
        off = os.environ.get("SUSPKIN_NO_NUMBA", "") not in ("", "0")
        _enabled = False if off else kernels() is not None
    return _enabled


def pack_geom(c: _Ctx) -> Vec:
    """Flat corner geometry (45 floats): see the index use in ``_nb_kernels``."""
    po0 = np.asarray(c.po0)
    out = np.concatenate(
        (
            [c.sign],
            c.uca_front,
            c.lca_front,
            c.uca_hat,
            c.lca_hat,
            c.r_ubj,
            c.r_lbj,
            c.d0_hat,
            c.r_tie,
            c.r_wc,
            [c.len_ubj, c.len_tie, c.wc0_z],
            c.spin0_t,
            c.r_po,
            po0 - np.asarray(c.lca_front),
            po0 - np.asarray(c.uca_front),
            [_MOUNT[c.mount], c.radius],
        )
    ).astype(np.float64)
    assert out.shape == (45,)
    return out


def pack_rocker(c: _Ctx) -> Vec:
    """Flat rocker data (15 floats): sign, pivot, v_par, w, axis x w, |w|^2, pushrod length."""
    r = c.rocker
    return np.concatenate(([r.sign], r.pivot, r.v_par, r.w, r.axw, [r.ww, r.length])).astype(
        np.float64
    )


def arrays(c: _Ctx) -> tuple[Vec, Vec]:
    """Packed ``(geom, rocker)`` arrays of a corner context (cached on the context)."""
    if c.fast_arrays is None:
        c.fast_arrays = (pack_geom(c), pack_rocker(c))
    return c.fast_arrays


def newton(c: _Ctx, x0: _V3, travel: float, ti: _V3) -> tuple[bool, tuple[_V3, _V3] | None]:
    """Kernel version of ``corner._newton``: ``(handled, result)``.

    ``handled`` is False if the kernel raised (fast path now disabled): run the Python body.
    ``result`` is None if Newton does not converge.
    """
    k = kernels()
    assert k is not None
    try:
        r = k.newton(arrays(c)[0], x0[0], x0[1], x0[2], travel, ti[0], ti[1], ti[2])
    except Exception as exc:
        _disable(f"kernel call failed: {exc!r}")
        return False, None
    if r[0] == 0:
        return True, None
    return True, ((r[1], r[2], r[3]), (r[4], r[5], r[6]))


def ground(
    c: _Ctx,
    n: Vec,
    offset: float,
    z_static: float,
    ti: _V3,
    guess: object,
    slope0: float | None,
) -> tuple[float, _V3, _V3, float, float] | None:
    """Kernel secant of ``vehicle._solve_corner_ground`` (warm-started by ``guess``).

    Returns ``(travel, angles, residuals, phi0, slope)`` or None on any failure status or
    exception, in which case the caller runs the pure-Python path.
    """
    k = kernels()
    assert k is not None
    try:
        geom, rk = arrays(c)
        r = k.ground(
            geom,
            rk,
            float(n[0]),
            float(n[1]),
            float(n[2]),
            offset,
            z_static,
            ti[0],
            ti[1],
            ti[2],
            guess.travel,  # type: ignore[attr-defined]
            guess.theta_uca,  # type: ignore[attr-defined]
            guess.theta_lca,  # type: ignore[attr-defined]
            guess.psi,  # type: ignore[attr-defined]
            guess.rocker_angle,  # type: ignore[attr-defined]
            float("nan") if slope0 is None else slope0,
        )
    except Exception as exc:
        _disable(f"kernel call failed: {exc!r}")
        return None
    if r[0] != 0:
        return None
    return r[1], (r[2], r[3], r[4]), (r[5], r[6], r[7]), r[8], r[10]
