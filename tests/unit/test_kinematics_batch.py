"""Batched corner kinematics (``suspkin.kinematics.batch``) against the scalar solver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from suspkin.geometry import build_corners, load_vehicle
from suspkin.kinematics.batch import corner_columns, make_static, rebuild
from suspkin.kinematics.corner import solve_corner

VEHICLE = Path(__file__).resolve().parents[2] / "configs" / "vehicles" / "fs_reference.yaml"


def test_batch_matches_scalar() -> None:
    """Rebuilt points of 20 random rack-0 heave states equal ``solve_corner`` (rtol 1e-9)."""
    fl = build_corners(load_vehicle(VEHICLE))["FL"]
    rng = np.random.default_rng(7)
    travel = np.sort(rng.uniform(-0.03, 0.03, 20))
    states = [solve_corner(fl, float(t), 0.0) for t in travel]
    st = make_static([fl])

    def col(name: str) -> np.ndarray:
        return np.array([[getattr(s, name) for s in states]])

    rb = rebuild(st, col("theta_uca"), col("theta_lca"), col("psi"), col("rocker_angle"))
    for name, p in rb.moved.items():
        ref = np.array([s.points[name] for s in states])
        np.testing.assert_allclose(p[0], ref, rtol=1e-9, atol=1e-12, err_msg=name)
    np.testing.assert_allclose(rb.spin[0], [s.spin_axis for s in states], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rb.coil[0], col("coilover_length")[0], rtol=1e-9)
    cols = corner_columns(rb, travel)
    assert cols["camber"].shape == (1, 20)
    assert np.all(np.isfinite(cols["camber"]))


def test_import_without_numba() -> None:
    """With ``import numba`` failing, the package imports and a corner solves (pure Python)."""
    code = (
        "import sys; sys.modules['numba'] = None\n"
        "from suspkin.kinematics import _numba, solve_corner\n"
        "from suspkin.geometry import build_corners, load_vehicle\n"
        f"fl = build_corners(load_vehicle(r'{VEHICLE}'))['FL']\n"
        "s = solve_corner(fl, 0.01, 0.0)\n"
        "assert not _numba.available()\n"
        "assert abs(s.travel - 0.01) < 1e-12\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
