"""Analytic Jacobian of the corner closing equations vs central finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from suspkin.kinematics import solve_corner
from suspkin.kinematics.corner import _Ctx, _jacobian, _residuals
from tests.unit.helpers import fixture_corners

_H = 1e-6


def _fd_jacobian(c: _Ctx, x: np.ndarray, travel: float, tie_inner: np.ndarray) -> np.ndarray:
    cols = []
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = _H
        cols.append(
            (_residuals(c, x + dx, travel, tie_inner) - _residuals(c, x - dx, travel, tie_inner))
            / (2 * _H)
        )
    return np.array(cols).T


def _check(cid: str, travel: float, rack: float, perturb: tuple[float, float, float]) -> None:
    corner = fixture_corners()[cid]  # type: ignore[index]
    st = solve_corner(corner, travel, rack)
    c = _Ctx(corner)
    x = np.array([st.theta_uca, st.theta_lca, st.psi]) + np.array(perturb)
    tie_inner = np.asarray(st.points["tie_inner"])
    jac = _jacobian(c, x, travel, tie_inner)
    fd = _fd_jacobian(c, x, travel, tie_inner)
    err = np.max(np.abs(jac - fd)) / max(np.max(np.abs(fd)), 1e-12)
    assert err < 1e-6


@pytest.mark.parametrize("cid", ["FL", "FR", "RL", "RR"])
@pytest.mark.parametrize("travel", [0.0, 0.03, -0.03])
def test_jacobian_matches_fd_travel(cid: str, travel: float) -> None:
    _check(cid, travel, 0.0, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("cid", ["FL", "FR"])
@pytest.mark.parametrize("rack", [0.02, -0.02])
def test_jacobian_matches_fd_rack(cid: str, rack: float) -> None:
    _check(cid, 0.01, rack, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("cid", ["FL", "FR", "RL", "RR"])
@pytest.mark.parametrize(
    "perturb", [(0.02, -0.01, 0.03), (-0.03, 0.02, -0.05), (0.0, 0.0, 0.2), (0.01, 0.01, -0.3)]
)
def test_jacobian_matches_fd_off_solution(cid: str, perturb: tuple[float, float, float]) -> None:
    """Valid at arbitrary angles, not only at solved poses (Newton iterates)."""
    _check(cid, 0.0, 0.0, perturb)


def test_hybr_fallback_gives_same_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    from suspkin.kinematics import corner as corner_mod

    corner = fixture_corners()["FL"]
    ref = solve_corner(corner, 0.003)
    monkeypatch.setattr(corner_mod, "_newton", lambda *a, **k: None)
    alt = solve_corner(corner, 0.003)
    np.testing.assert_allclose(
        [alt.theta_uca, alt.theta_lca, alt.psi], [ref.theta_uca, ref.theta_lca, ref.psi], atol=1e-9
    )
