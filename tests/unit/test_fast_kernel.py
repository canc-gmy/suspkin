"""Numba fast path: parity with the pure-Python path, fallback and kernel source coverage."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics import _fast, corner, vehicle
from suspkin.kinematics.corner import _Ctx, solve_corner, sweep_corner
from suspkin.kinematics.rocker import KinematicsError
from suspkin.kinematics.vehicle import _static_contact, solve_vehicle
from suspkin.report.studies import run_standard_studies
from suspkin.sweeps import sweep_vehicle
from tests.unit.helpers import FIXTURE

RTOL, ATOL = 1e-9, 1e-10


@pytest.fixture(scope="module")
def veh() -> Any:
    return load_vehicle(FIXTURE)


@pytest.fixture(autouse=True)
def _fresh_fast_state() -> Iterator[None]:
    _fast.reset()
    yield
    _fast.reset()


@pytest.fixture
def numba_only() -> None:
    pytest.importorskip("numba")
    if not _fast.available():
        pytest.skip("numba path disabled")


def _run_py(monkeypatch: pytest.MonkeyPatch, fn: Callable[[], Any]) -> Any:
    with monkeypatch.context() as m:
        m.setenv("SUSPKIN_NO_NUMBA", "1")
        _fast.reset()
        try:
            return fn()
        finally:
            _fast.reset()


def _cmp_corner(a: Any, b: Any) -> None:
    assert a.travel == pytest.approx(b.travel, rel=RTOL, abs=ATOL)
    for k in ("theta_uca", "theta_lca", "psi", "rocker_angle", "coilover_length"):
        assert getattr(a, k) == pytest.approx(getattr(b, k), rel=RTOL, abs=ATOL)
    for name, p in a.points.items():
        np.testing.assert_allclose(p, b.points[name], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a.spin_axis, b.spin_axis, rtol=RTOL, atol=ATOL)


def _cmp_vehicle(a: list[Any], b: list[Any]) -> None:
    assert len(a) == len(b)
    for va, vb in zip(a, b, strict=True):
        for cid in va.corners:
            _cmp_corner(va.corners[cid], vb.corners[cid])
            np.testing.assert_allclose(
                va.contact_points[cid], vb.contact_points[cid], rtol=RTOL, atol=ATOL
            )


def test_available_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUSPKIN_NO_NUMBA", "1")
    _fast.reset()
    assert not _fast.available()
    monkeypatch.setenv("SUSPKIN_NO_NUMBA", "yes")
    _fast.reset()
    assert not _fast.available()
    for val in ("", "0"):
        monkeypatch.setenv("SUSPKIN_NO_NUMBA", val)
        _fast.reset()
        assert _fast.available() == (_fast.kernels() is not None)
    monkeypatch.delenv("SUSPKIN_NO_NUMBA")
    _fast.reset()
    assert _fast.available() == (_fast.kernels() is not None)


def test_kernel_constants_match_solver() -> None:
    pytest.importorskip("numba")
    k = _fast.kernels()
    assert k is not None
    assert k.RESIDUAL_TOL == corner._RESIDUAL_TOL
    assert k.MAX_ANGLE_STEP == corner._MAX_ANGLE_STEP
    assert k.NEWTON_MAX_ITER == corner._NEWTON_MAX_ITER
    assert k.NEWTON_F_TOL == corner._NEWTON_F_TOL
    assert k.NEWTON_X_TOL == corner._NEWTON_X_TOL
    assert k.NEWTON_STALL_F_TOL == corner._NEWTON_STALL_F_TOL
    assert k.MAX_BACKTRACKS == corner._MAX_BACKTRACKS
    assert k.G_TOL == vehicle._G_TOL
    assert k.MAX_ITER == vehicle._MAX_ITER
    assert k.MAX_TRAVEL == vehicle._MAX_TRAVEL


def test_corner_sweep_parity(numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fl = build_corners(veh)["FL"]
    tr = np.linspace(-0.03, 0.03, 31)
    fast = sweep_corner(fl, tr)
    ref = _run_py(monkeypatch, lambda: sweep_corner(fl, tr))
    for a, b in zip(fast, ref, strict=True):
        _cmp_corner(a, b)


@pytest.mark.parametrize("kind", ["heave", "roll", "rack"])
def test_vehicle_sweep_parity(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    rack_lo, rack_hi = veh.front.steering.rack_travel
    lo, hi = {
        "heave": (-0.025, 0.025),
        "roll": (-np.radians(2.0), np.radians(2.0)),
        "rack": (rack_lo, rack_hi),
    }[kind]
    vals = np.linspace(lo, hi, 25)
    fast = sweep_vehicle(veh, **{kind: vals})
    ref = _run_py(monkeypatch, lambda: sweep_vehicle(veh, **{kind: vals}))
    _cmp_vehicle(fast, ref)


def test_standard_studies_parity(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = run_standard_studies(veh, steps=31).tables()
    ref = _run_py(monkeypatch, lambda: run_standard_studies(veh, steps=31).tables())
    assert fast.keys() == ref.keys()
    for name, df in fast.items():
        num = df.select_dtypes("number").columns
        np.testing.assert_allclose(
            df[num].to_numpy(), ref[name][num].to_numpy(), rtol=1e-6, atol=1e-9, err_msg=name
        )


def test_kernel_failure_falls_back(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    hv = np.linspace(-0.02, 0.02, 9)
    ref = _run_py(monkeypatch, lambda: sweep_vehicle(veh, heave=hv))
    k = _fast.kernels()
    assert k is not None
    calls: list[int] = []

    def failing(*args: Any) -> tuple[float, ...]:
        calls.append(1)
        return (2, *([0.0] * 10))

    monkeypatch.setattr(k, "ground", failing)
    out = sweep_vehicle(veh, heave=hv)
    assert calls
    for a, b in zip(out, ref, strict=True):
        for cid in a.corners:
            # Pure-Python path from the same inputs: identical to the forced-Python run.
            assert a.corners[cid].travel == b.corners[cid].travel
            np.testing.assert_array_equal(a.contact_points[cid], b.contact_points[cid])


def test_kernel_failure_keeps_error_semantics(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    corners = build_corners(veh)
    st = solve_vehicle(veh, corners)
    k = _fast.kernels()
    assert k is not None
    monkeypatch.setattr(k, "ground", lambda *a: (1, *([0.0] * 10)))
    with pytest.raises(KinematicsError, match="cannot reach the ground"):
        solve_vehicle(veh, corners, heave=0.5, guess=st)


def _snapshot_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the module state with monkeypatch so a disabled fast path is restored."""
    for name in ("_mod", "_tried", "_enabled"):
        monkeypatch.setattr(_fast, name, getattr(_fast, name))


@pytest.mark.parametrize("which", ["ground", "newton"])
def test_kernel_exception_disables_and_falls_back(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    hv = np.linspace(-0.02, 0.02, 9)
    ref = _run_py(monkeypatch, lambda: sweep_vehicle(veh, heave=hv))
    k = _fast.kernels()
    assert k is not None
    _snapshot_fast(monkeypatch)

    def boom(*args: Any) -> None:
        raise RuntimeError("simulated LLVM failure")

    monkeypatch.setattr(k, which, boom)
    if which == "newton":
        # The ground kernel would hide the Newton kernel: make it fail by status.
        monkeypatch.setattr(k, "ground", lambda *a: (2, *([0.0] * 10)))
    with pytest.warns(RuntimeWarning, match="fast path disabled"):
        out = sweep_vehicle(veh, heave=hv)
    assert not _fast.available()
    _cmp_vehicle(out, ref)


def test_warmup_failure_disables_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba")
    _snapshot_fast(monkeypatch)

    def bad_warmup(k: Any) -> None:
        raise RuntimeError("cache dir not writable")

    monkeypatch.setattr(_fast, "_warmup", bad_warmup)
    monkeypatch.setattr(_fast, "_mod", None)
    monkeypatch.setattr(_fast, "_tried", False)
    with pytest.warns(RuntimeWarning, match="cache dir not writable"):
        assert _fast.kernels() is None
    assert not _fast.available()


def test_missing_numba_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    _snapshot_fast(monkeypatch)
    monkeypatch.setitem(sys.modules, "suspkin.kinematics._nb_kernels", None)
    monkeypatch.setattr(_fast, "_mod", None)
    monkeypatch.setattr(_fast, "_tried", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _fast.kernels() is None
        assert not _fast.available()


def _with_mount(v: Any, mount: str) -> Any:
    return v.model_copy(
        update={
            "front": v.front.model_copy(update={"pushrod_mount": mount}),
            "rear": v.rear.model_copy(update={"pushrod_mount": mount}),
        }
    )


@pytest.mark.parametrize("mount", ["upright", "lca", "uca"])
def test_pushrod_mount_parity(
    numba_only: None, veh: Any, monkeypatch: pytest.MonkeyPatch, mount: str
) -> None:
    v = _with_mount(veh, mount)
    hv = np.linspace(-0.02, 0.02, 21)
    tr = np.linspace(-0.02, 0.02, 21)
    fast = sweep_vehicle(v, heave=hv)
    ref = _run_py(monkeypatch, lambda: sweep_vehicle(v, heave=hv))
    _cmp_vehicle(fast, ref)
    fl = build_corners(v)["FL"]
    cfast = sweep_corner(fl, tr)
    cref = _run_py(monkeypatch, lambda: sweep_corner(fl, tr))
    for a, b in zip(cfast, cref, strict=True):
        _cmp_corner(a, b)


# --- kernel source coverage: run the kernels as plain Python (JIT disabled) -------------


@pytest.fixture(scope="module")
def pure() -> Iterator[ModuleType]:
    numba = pytest.importorskip("numba")
    k = _fast.kernels()
    assert k is not None
    spec = importlib.util.spec_from_file_location("_nb_kernels_pure", k.__file__)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old = numba.config.DISABLE_JIT
    numba.config.DISABLE_JIT = True
    try:
        spec.loader.exec_module(mod)
    finally:
        numba.config.DISABLE_JIT = old
    yield mod


def _ground_args(veh: Any, cid: str = "FL", heave: float = 0.004) -> list[Any]:
    c = build_corners(veh)[cid]
    ctx = _Ctx(c)
    guess = solve_corner(c, 0.001)
    z_static = float(_static_contact(c, ctx.spin0)[2])
    ti = corner._tup(corner._tie_inner(ctx, 0.0))
    geom, rk = _fast.arrays(ctx)
    return [
        geom, rk, 0.0, 0.0, 1.0, heave, z_static, *ti,
        guess.travel, guess.theta_uca, guess.theta_lca, guess.psi, guess.rocker_angle,
        float("nan"),
    ]  # fmt: skip


@pytest.mark.parametrize("cid", ["FL", "RR"])
def test_pure_kernels_match_compiled(
    numba_only: None, pure: ModuleType, veh: Any, cid: str
) -> None:
    k = _fast.kernels()
    assert k is not None
    args = _ground_args(veh, cid)
    a = pure.ground(*args)
    b = k.ground(*args)
    assert a[0] == b[0] == 0
    np.testing.assert_allclose(a[1:], b[1:], rtol=1e-12, atol=1e-14)
    # Warm-started slope and the Newton kernel on its own.
    warm = [*args[:-1], 0.98]
    np.testing.assert_allclose(pure.ground(*warm)[1:], k.ground(*warm)[1:], rtol=1e-12, atol=1e-14)
    n_args = (args[0], *args[11:14], args[10], *args[7:10])
    np.testing.assert_allclose(pure.newton(*n_args), k.newton(*n_args), rtol=1e-12, atol=1e-14)


def test_pure_kernel_failure_statuses(pure: ModuleType, veh: Any) -> None:
    args = _ground_args(veh)
    assert pure.ground(*args)[0] == 0
    far = list(args)
    far[10] = 0.5  # travel guess out of range
    assert pure.ground(*far)[0] == 1
    jump = list(args)
    jump[11] += 0.5  # starting angles far from the closing solution
    assert pure.ground(*jump)[0] in (2, 3, 4)
    high = list(args)
    high[5] = 0.3  # ground 0.3 m above: wheel cannot reach it
    assert pure.ground(*high)[0] != 0
    # Rocker unreachable: shrink the pushrod length in the packed rocker data.
    rk = np.array(args[1])
    rk[14] = 1e-6
    bad = list(args)
    bad[1] = rk
    assert pure.ground(*bad)[0] == 6
    # Wild start: Newton either fails or lands on some root, but never raises.
    nr = pure.newton(args[0], 3.0, 3.0, 3.0, 0.0, *args[7:10])
    assert nr[0] in (0, 1)


@pytest.mark.parametrize("mount", ["lca", "uca"])
def test_pure_kernel_wishbone_mounts(
    numba_only: None, pure: ModuleType, veh: Any, mount: str
) -> None:
    k = _fast.kernels()
    assert k is not None
    args = _ground_args(_with_mount(veh, mount))
    a, b = pure.ground(*args), k.ground(*args)
    assert a[0] == b[0] == 0  # the kernel solves it, no silent fallback
    np.testing.assert_allclose(a[1:], b[1:], rtol=1e-12, atol=1e-14)


# --- wrapper logic with a fake kernel module (independent of numba and of the env var) --


class _FakeKernels:
    def __init__(self, newton: Any, ground: Any) -> None:
        self.newton = newton
        self.ground = ground


def _use_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeKernels) -> None:
    _snapshot_fast(monkeypatch)
    monkeypatch.setattr(_fast, "kernels", lambda: fake)


def _ctx_and_guess(veh: Any) -> tuple[_Ctx, Any]:
    c = build_corners(veh)["FL"]
    return _Ctx(c), solve_corner(c, 0.001)


def test_wrapper_newton_status_and_exception(veh: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = _ctx_and_guess(veh)
    ok = (1, 0.1, 0.2, 0.3, 1e-13, 2e-13, 3e-13)
    _use_fake(monkeypatch, _FakeKernels(lambda *a: ok, None))
    assert _fast.newton(ctx, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)) == (
        True,
        ((0.1, 0.2, 0.3), (1e-13, 2e-13, 3e-13)),
    )
    _use_fake(monkeypatch, _FakeKernels(lambda *a: (0, *([0.0] * 6)), None))
    assert _fast.newton(ctx, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)) == (True, None)

    def boom(*a: Any) -> None:
        raise RuntimeError("kernel exploded")

    _use_fake(monkeypatch, _FakeKernels(boom, None))
    with pytest.warns(RuntimeWarning, match="kernel exploded"):
        assert _fast.newton(ctx, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)) == (False, None)


def test_wrapper_ground_status_and_exception(veh: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, guess = _ctx_and_guess(veh)
    n = np.array([0.0, 0.0, 1.0])
    ti = (0.0, 0.0, 0.0)
    seen: list[Any] = []

    def good(*a: Any) -> tuple[float, ...]:
        seen.append(a)
        return (0, 0.002, 0.1, 0.2, 0.3, 1e-13, 2e-13, 3e-13, 0.4, 5e-12, 0.97)

    _use_fake(monkeypatch, _FakeKernels(None, good))
    assert _fast.ground(ctx, n, 0.0, 0.0, ti, guess, None) == (
        0.002,
        (0.1, 0.2, 0.3),
        (1e-13, 2e-13, 3e-13),
        0.4,
        0.97,
    )
    assert np.isnan(seen[0][-1])  # no warm-start slope -> NaN
    _fast.ground(ctx, n, 0.0, 0.0, ti, guess, 0.9)
    assert seen[1][-1] == 0.9

    _use_fake(monkeypatch, _FakeKernels(None, lambda *a: (4, *([0.0] * 10))))
    assert _fast.ground(ctx, n, 0.0, 0.0, ti, guess, None) is None

    def boom(*a: Any) -> None:
        raise RuntimeError("kernel exploded")

    _use_fake(monkeypatch, _FakeKernels(None, boom))
    with pytest.warns(RuntimeWarning, match="kernel exploded"):
        assert _fast.ground(ctx, n, 0.0, 0.0, ti, guess, None) is None


def test_kernels_import_and_warmup_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    _snapshot_fast(monkeypatch)
    fake = ModuleType("fake_kernels")
    monkeypatch.setitem(sys.modules, "suspkin.kinematics._nb_kernels", fake)
    calls: list[ModuleType] = []
    monkeypatch.setattr(_fast, "_warmup", calls.append)
    monkeypatch.setattr(_fast, "_mod", None)
    monkeypatch.setattr(_fast, "_tried", False)
    assert _fast.kernels() is fake and calls == [fake]
    assert _fast.kernels() is fake and len(calls) == 1  # cached, no second warm-up
    # Non-ImportError during import: disabled with a warning.
    monkeypatch.setattr(_fast, "_mod", None)
    monkeypatch.setattr(_fast, "_tried", False)

    def bad_import(name: str) -> ModuleType:
        raise RuntimeError("broken install")

    monkeypatch.setattr(_fast.importlib, "import_module", bad_import)
    with pytest.warns(RuntimeWarning, match="broken install"):
        assert _fast.kernels() is None


def test_pack_arrays_shapes_and_cache(veh: Any) -> None:
    ctx, _ = _ctx_and_guess(veh)
    geom, rk = _fast.arrays(ctx)
    assert geom.shape == (45,) and rk.shape == (15,)
    assert _fast.arrays(ctx)[0] is geom  # cached on the context
    assert geom[43] == 0.0  # upright mount code
