"""Training, fine-tuning, persistence and prediction tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from suspkin.surrogate.data import SPLIT_VAL

torch = pytest.importorskip("torch")

from suspkin.surrogate.model import MLPConfig  # noqa: E402
from suspkin.surrogate.train import (  # noqa: E402
    SurrogatePredictor,
    TrainConfig,
    TrainedModel,
    evaluate,
    train,
)
from tests.unit.test_surrogate_model import make_ds  # noqa: E402

MCFG = MLPConfig(n_layers=3, width=64, activation="silu", residual=True)
TCFG = TrainConfig(
    lr=3e-3, batch_size=64, max_epochs=60, patience=60, warmup_epochs=3, device="cpu"
)


def test_training_reduces_val_score() -> None:
    ds = make_ds()
    untrained = train(ds, MCFG, replace(TCFG, max_epochs=1, warmup_epochs=1, lr=1e-9))
    model = train(ds, MCFG, TCFG)
    assert model.val_score < 0.25 * untrained.val_score
    rep = evaluate(model, ds, SPLIT_VAL)
    assert rep.score == pytest.approx(model.val_score, rel=1e-3)
    assert rep.validity_accuracy > 0.8
    assert len(rep.to_frame()) == 3
    assert rep.worst


def test_model_stores_per_channel_ratios() -> None:
    ds = make_ds()
    model = train(ds, MCFG, replace(TCFG, max_epochs=5, patience=5))
    ratios = model.meta["val_channel_ratio"]
    assert list(ratios) == list(ds.channels)
    assert np.mean(list(ratios.values())) == pytest.approx(model.val_score, rel=1e-4)
    rep = evaluate(model, ds, SPLIT_VAL)
    np.testing.assert_allclose(list(ratios.values()), rep.per_channel_ratio, rtol=1e-3)


def test_pod_training_runs() -> None:
    ds = make_ds()
    model = train(ds, replace(MCFG, output="pod", pod_k=12), replace(TCFG, max_epochs=20))
    assert model.pod is not None
    assert model.pod.k == 12
    assert np.isfinite(model.val_score)


def test_finetune_keeps_scaler_and_improves_on_new_points() -> None:
    ds = make_ds(n=300)
    model = train(ds, MCFG, replace(TCFG, max_epochs=30))
    new = make_ds(n=300, seed=7)
    rng = np.random.default_rng(1)
    new = replace(new, X=0.2 * new.X + 0.8, weight=rng.random(new.n) + 0.5)
    before = evaluate(model, new, SPLIT_VAL).score
    tuned = train(new, MCFG, replace(TCFG, max_epochs=40, lr=1e-3), init=model)
    assert tuned.scaler is model.scaler
    assert evaluate(tuned, new, SPLIT_VAL).score < before


def test_save_load_and_predictor(tmp_path: Path) -> None:
    ds = make_ds()
    model = train(ds, replace(MCFG, output="pod", pod_k=10), replace(TCFG, max_epochs=5))
    model.save(tmp_path / "m")
    for f in ("model.pt", "model.json", "norm.npz"):
        assert (tmp_path / "m" / f).exists()
    loaded = TrainedModel.load(tmp_path / "m", device="cpu")
    assert {k: v for k, v in loaded.meta.items() if k != "val_channel_ratio"} == ds.meta
    assert set(loaded.meta["val_channel_ratio"]) == set(ds.channels)
    p1 = SurrogatePredictor(model, device="cpu")
    p2 = SurrogatePredictor.load(tmp_path / "m", device="cpu")
    y1, v1 = p1.predict(ds.X[:20])
    y2, v2 = p2.predict(ds.X[:20])
    assert y1.shape == (20, 3, 2, 7)
    assert y1.dtype == np.float64
    assert v1.shape == (20, 2, 7)
    assert np.array_equal(y1, y2)
    assert np.array_equal(v1, v2)
    with pytest.raises(ValueError, match="shape"):
        p2.predict(ds.X[:, :3])
    with pytest.raises(ValueError, match="param_names"):
        p2.predict(ds.X, param_names=("a", "b", "c", "d"))


def test_load_uses_map_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = make_ds()
    model = train(ds, MCFG, replace(TCFG, max_epochs=2))
    model.save(tmp_path / "m")
    seen: dict[str, object] = {}
    real = torch.load

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "load", spy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = TrainedModel.load(tmp_path / "m")  # device "auto" -> cpu
    assert seen["map_location"] == torch.device("cpu")
    assert all(v.device.type == "cpu" for v in loaded.state_dict.values())


def test_same_seed_same_score() -> None:
    ds = make_ds()
    t = replace(TCFG, max_epochs=8)
    assert train(ds, MCFG, t).val_score == train(ds, MCFG, t).val_score


def test_finetune_never_worse_than_init() -> None:
    ds = make_ds()
    model = train(ds, MCFG, replace(TCFG, max_epochs=40))
    # absurd lr: every epoch is worse than init, so the init weights must come back
    bad = replace(TCFG, max_epochs=3, lr=1e3, finetune_lr_factor=1.0, patience=3)
    tuned = train(ds, MCFG, bad, init=model)
    assert tuned.val_score <= model.val_score + 1e-12
    assert tuned.history[0]["epoch"] == -1.0
    assert tuned.meta.get("kept_init_weights") is True
    for k, v in model.state_dict.items():
        assert torch.equal(v, tuned.state_dict[k])


def test_extra_val_enters_the_early_stopping_score() -> None:
    ds = make_ds(n=300)
    model = train(ds, MCFG, replace(TCFG, max_epochs=10))
    new = make_ds(n=60, seed=7)
    new = replace(new, X=0.2 * new.X + 0.8)
    plain = train(ds, MCFG, replace(TCFG, max_epochs=3), init=model)
    mixed = train(ds, MCFG, replace(TCFG, max_epochs=3), init=model, extra_val=new)
    s_val = evaluate(model, ds, SPLIT_VAL).score
    s_new = evaluate(model, new, None).score
    assert plain.history[0]["val"] == pytest.approx(s_val, rel=1e-3)  # default unchanged
    assert mixed.history[0]["val"] == pytest.approx(0.5 * (s_val + s_new), rel=1e-3)
