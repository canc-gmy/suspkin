"""History CSV writer/reader and the resume cache."""

from pathlib import Path

import numpy as np
import pytest

from suspkin.optimize import DesignSpace, EvalCache, HistoryWriter, evaluate, read_history
from suspkin.optimize.history import HistoryError, cache_key, history_columns
from suspkin.optimize.spec import Study, load_study

FIX = Path(__file__).parents[1] / "fixtures" / "studies" / "objectives_study.yaml"


@pytest.fixture(scope="module")
def setup() -> tuple[Study, DesignSpace]:
    study, vehicle = load_study(FIX)
    return study, DesignSpace(study, vehicle)


def test_column_order(setup: tuple[Study, DesignSpace], tmp_path: Path) -> None:
    study, space = setup
    path = tmp_path / "history.csv"
    with HistoryWriter(path, space, study):
        pass
    cols = list(read_history(path).columns)
    assert cols == history_columns(space, study)
    exp = ["eval_id", "x_front.tie_inner.u", "x_front.tie_inner.v", "x_front.tie_inner.w"]
    exp += [f"p_front.tie_inner_{a}" for a in "xyz"]
    exp += [f"f_{o.name}" for o in study.objectives]
    exp += ["f_total", "g_mr_static", "g_camber_band"]
    exp += ["max_g", "feasible", "status", "seconds", "source"]
    assert cols == exp


def test_row_roundtrip_and_cache(setup: tuple[Study, DesignSpace], tmp_path: Path) -> None:
    study, space = setup
    path = tmp_path / "h.csv"
    x = np.array([0.3, 0.5, 0.7])
    ev = evaluate(space, x)
    with HistoryWriter(path, space, study) as w:
        w.append(4, x, ev, 0.25, "de")
    df = read_history(path)
    assert len(df) == 1
    assert df.loc[0, "status"] == "ok"
    assert df.loc[0, "source"] == "de"
    assert df.loc[0, "f_total"] == pytest.approx(ev.f_total)
    np.testing.assert_allclose(
        df[["p_front.tie_inner_x", "p_front.tie_inner_y", "p_front.tie_inner_z"]].to_numpy()[0],
        space.points(x)["front.tie_inner"],
    )
    cache = EvalCache.from_history(df, space, study)
    ent = cache[cache_key(x)]
    np.testing.assert_allclose(ent.f_vector, ev.f_vector())
    np.testing.assert_allclose(ent.g_vector, ev.g_vector())
    assert ent.feasible == ev.feasible


def test_from_history_rejects_other_study(setup: tuple[Study, DesignSpace], tmp_path: Path) -> None:
    study, space = setup
    path = tmp_path / "h.csv"
    with HistoryWriter(path, space, study):
        pass
    df = read_history(path).drop(columns=["g_mr_static"])
    with pytest.raises(HistoryError, match="g_mr_static"):
        EvalCache.from_history(df, space, study)
