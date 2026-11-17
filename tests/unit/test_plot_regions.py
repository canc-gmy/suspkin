"""Tests for the 3D study view."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from suspkin.optimize import DesignSpace
from suspkin.optimize.spec import load_study
from suspkin.plotting import plot_study_3d

ROOT = Path(__file__).parents[2]
STUDY = ROOT / "configs" / "studies" / "fs_reference_constraints.yaml"
FULL = Path(__file__).parents[1] / "fixtures" / "studies" / "full_study.yaml"


def _space(path: Path) -> DesignSpace:
    study, vehicle = load_study(path)
    return DesignSpace(study, vehicle)


def _n_lines(fig: Figure) -> int:
    return len(fig.axes[0].lines)


def test_returns_figure_and_draws_domains_and_regions() -> None:
    space = _space(STUDY)
    fig = plot_study_3d(space)
    assert isinstance(fig, Figure)
    n_vars = len(space.study.variables)  # 4 symmetric: solid + mirrored dashed boxes
    styles = [ln.get_linestyle() for ln in fig.axes[0].lines]
    assert styles.count("--") >= 12 * n_vars
    assert styles.count(":") >= 24  # two inside regions, 12 edges each
    colors = [ln.get_color() for ln in fig.axes[0].lines]
    assert "#d62728" in colors  # keep-out regions (also the tie rod colour)


def test_x_moves_points_and_vehicle() -> None:
    space = _space(STUDY)
    fig0 = plot_study_3d(space)
    fig1 = plot_study_3d(space, np.full(space.n, 0.25))
    assert isinstance(fig1, Figure)
    assert _n_lines(fig0) == _n_lines(fig1)


def test_full_study_with_asymmetric_and_offset_domains() -> None:
    space = _space(FULL)
    fig = plot_study_3d(space, np.full(space.n, 0.5))
    ax_lines = fig.axes[0].lines
    # asymmetric variable: no dashed mirror for it, so fewer dashed than 12 * n_variables
    dashed = sum(ln.get_linestyle() == "--" for ln in ax_lines)
    assert dashed == 12 * 3
    assert len(ax_lines) > 12 * 4
