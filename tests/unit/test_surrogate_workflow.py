"""Surrogate training / validation workflow and plots (tiny CPU problem)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suspkin.optimize.spec import StudyConfigError
from suspkin.plotting.surrogate import plot_parity, plot_rmse_ratio, plot_worst_curves
from suspkin.surrogate.dataset import DatasetError, sample_axle
from suspkin.surrogate.workflow import (
    display_unit,
    train_surrogate,
    validate_surrogate,
    validation_table,
)
from tests.unit.test_surrogate_sampling import REAR_VAR, STUDY

TRAIN_BLOCKS = """  training: {device: cpu, max_epochs: 4, patience: 2}
  hpo: {n_trials: 2, timeout: 5 min}
"""


def study_text(n: int = 40) -> str:
    text = STUDY.replace(REAR_VAR, "").replace("n_initial: 12", f"n_initial: {n}")
    return text.rstrip("\n") + "\n" + TRAIN_BLOCKS


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("wf")
    study = base / "study.yaml"
    study.write_text(study_text(), encoding="utf-8")
    root = base / "root"
    sample_axle(study, root, "front")
    res = train_surrogate(study, root, axles=["front"])
    assert res["front"].model_dir == root / "models" / "front" / "model"
    return study, root


def test_units_of_channels() -> None:
    assert display_unit("camber")[0] == "deg"
    assert display_unit("coilover_length") == ("mm", 1e3)
    assert display_unit("fv_swing_angle")[0] == "deg"


def test_validate_writes_tables_and_figures(trained: tuple[Path, Path]) -> None:
    study, root = trained
    reports = validate_surrogate(study, root, axles=["front"], device="cpu")
    rep = reports["front"]
    vdir = root / "validation" / "front"
    tab = pd.read_csv(vdir / "validation.csv")
    assert list(tab["channel"]) == list(rep.channels)
    assert np.allclose(tab["ratio"], rep.per_channel_ratio)
    assert np.allclose(tab["rmse"] / tab["tol"], tab["ratio"])
    assert list(tab["passed"]) == list(tab["ratio"] <= 1.0)
    assert validation_table(rep).shape == tab.shape
    assert (vdir / "worst.csv").is_file() and (vdir / "summary.json").is_file()
    parity = pd.read_csv(vdir / "parity.csv")
    assert {"camber", "toe", "coilover_length", "fv_swing_angle"} <= set(parity["channel"])
    curves = pd.read_csv(vdir / "worst_curves.csv")
    assert 1 <= curves["rank"].nunique() <= 3
    for name in ("rmse_ratio", "parity", "worst_curves"):
        assert (vdir / "figures" / f"{name}.png").stat().st_size > 0


def test_mismatched_study_rejected(trained: tuple[Path, Path], tmp_path: Path) -> None:
    _, root = trained
    other = tmp_path / "other.yaml"
    other.write_text(study_text().replace("points: 7", "points: 9"), encoding="utf-8")
    with pytest.raises(DatasetError, match="does not match the study"):
        train_surrogate(other, root, axles=["front"])
    with pytest.raises(DatasetError, match="does not match the study"):
        validate_surrogate(other, root, axles=["front"])


def test_validate_without_model(trained: tuple[Path, Path], tmp_path: Path) -> None:
    study, root = trained
    import shutil

    cp = tmp_path / "cp"
    shutil.copytree(root / "dataset", cp / "dataset")
    with pytest.raises(DatasetError, match="no trained model"):
        validate_surrogate(study, cp, axles=["front"], device="cpu")


def test_study_without_surrogate_block(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(STUDY.split("surrogate:")[0], encoding="utf-8")
    with pytest.raises(StudyConfigError):
        train_surrogate(p, tmp_path, axles=["front"])


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    names = ["camber", "toe", "coilover_length"]
    val = pd.DataFrame(
        {"channel": names, "ratio": [0.4, 1.7, 80.0], "passed": [True, False, False]}
    )
    fom = rng.normal(size=60)
    par = pd.DataFrame(
        {
            "channel": np.repeat(names, 20),
            "unit": "deg",
            "fom": fom,
            "pred": fom + 0.01 * rng.normal(size=60),
        }
    )
    t = np.linspace(-20, 20, 7)
    cur = pd.concat(
        [
            pd.DataFrame(
                {
                    "rank": k,
                    "sample": k + 3,
                    "channel": "toe",
                    "unit": "deg",
                    "travel_mm": t,
                    "fom": t * 0.01,
                    "pred": t * 0.01 + 0.002 * k,
                }
            )
            for k in range(3)
        ]
    )
    return val, par, cur


def test_figures_from_synthetic_tables() -> None:
    val, par, cur = _tables()
    f1 = plot_rmse_ratio(val)
    assert len(f1.axes[0].patches) == 3
    f2 = plot_parity(par)
    assert sum(ax.get_visible() for ax in f2.axes) == 3
    f3 = plot_worst_curves(cur)
    assert len(f3.axes[0].lines) == 6
    with pytest.raises(ValueError):
        plot_rmse_ratio(val.drop(columns="ratio"))
