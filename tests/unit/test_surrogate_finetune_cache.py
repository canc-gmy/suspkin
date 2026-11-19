"""FinetuneCache: cached device tensors equal rebuilt ones; fine-tune patience (CPU)."""

from dataclasses import replace

import numpy as np
import pytest

from suspkin.surrogate.data_types import SPLIT_TRAIN, SPLIT_VAL, AxleDataset

torch = pytest.importorskip("torch")

from suspkin.surrogate.model import MLPConfig  # noqa: E402
from suspkin.surrogate.train import FinetuneCache, TrainConfig, TrainedModel, _Split, train  # noqa: E402
from tests.unit.test_surrogate_model import make_ds  # noqa: E402

MCFG = MLPConfig(n_layers=3, width=64, activation="silu", residual=True)
TCFG = TrainConfig(
    lr=3e-3, batch_size=64, max_epochs=60, patience=60, warmup_epochs=3, device="cpu"
)


def _concat(a: AxleDataset, b: AxleDataset) -> AxleDataset:
    return replace(
        a,
        X=np.concatenate([a.X, b.X]),
        Y=np.concatenate([a.Y, b.Y]),
        mask=np.concatenate([a.mask, b.mask]),
        feasible=np.concatenate([a.feasible, b.feasible]),
        split=np.concatenate([a.split, b.split]),
        source=np.concatenate([a.source, b.source]),
        weight=np.concatenate([a.weight, b.weight]),
    )


def _rows(ds: AxleDataset, sel: slice) -> AxleDataset:
    m = np.zeros(ds.n, dtype=bool)
    m[sel] = True
    return ds.subset(m)


def _case() -> tuple[TrainedModel, AxleDataset, AxleDataset]:
    """Model, static part (val + 40 replay rows) and 12 new train rows."""
    ds = make_ds(n=300)
    model = train(ds, MCFG, replace(TCFG, max_epochs=10))
    new = make_ds(n=12, seed=7)
    new = replace(
        new, X=0.2 * new.X + 0.8, split=np.zeros(new.n, dtype=np.uint8), weight=np.full(new.n, 5.0)
    )
    rng = np.random.default_rng(3)
    keep = ds.split == SPLIT_VAL
    keep[rng.choice(np.flatnonzero(ds.split == SPLIT_TRAIN), size=40, replace=False)] = True
    return model, ds.subset(keep), new


def test_cache_tensors_equal_rebuilt_ones() -> None:
    model, static, new = _case()
    cache = FinetuneCache(model, static, "cpu")
    cache.add_rows(_rows(new, slice(0, 5)))
    cache.add_rows(_rows(new, slice(5, None)))  # appended in two steps
    assert cache.n_al == new.n
    tr_s, va_s, ex_s = cache.splits()
    assert ex_s is not None
    full = _concat(static, new)
    cpu = torch.device("cpu")
    want = (
        _Split(full.subset(SPLIT_TRAIN), model.scaler, cpu, True),
        _Split(full.subset(SPLIT_VAL), model.scaler, cpu, True),
        _Split(new, model.scaler, cpu, True),
    )
    for got, ref in zip((tr_s, va_s, ex_s), want, strict=True):
        assert got.n == ref.n
        for a, b in zip(got.tensors, ref.tensors, strict=True):
            assert torch.equal(a, b)


def test_finetune_with_cache_matches_without() -> None:
    model, static, new = _case()
    tcfg = replace(TCFG, max_epochs=6, patience=6, seed=5)
    plain = train(_concat(static, new), MCFG, tcfg, init=model, extra_val=new)
    cache = FinetuneCache(model, static, "cpu")
    cache.add_rows(new)
    cached = train(cache.template, MCFG, tcfg, init=model, cache=cache)
    assert cached.val_score == pytest.approx(plain.val_score, rel=1e-5)
    assert [h["val"] for h in cached.history] == pytest.approx(
        [h["val"] for h in plain.history], rel=1e-5
    )
    with pytest.raises(ValueError, match="needs init"):
        train(cache.template, MCFG, tcfg, cache=cache)
    other = replace(model, scaler=replace(model.scaler, std=model.scaler.std * 2.0))
    with pytest.raises(ValueError, match="scaler"):
        train(cache.template, MCFG, tcfg, init=other, cache=cache)


def test_finetune_patience_counts_from_the_run_best() -> None:
    ds = make_ds(n=300)
    model = train(ds, MCFG, replace(TCFG, max_epochs=20))
    # a hot learning rate: the epochs are worse than init, but the patience counts from the
    # best epoch of the run, so it stops exactly `patience` epochs after that one
    hot = replace(TCFG, max_epochs=60, lr=5e-2, finetune_lr_factor=1.0, patience=4)
    tuned = train(ds, MCFG, hot, init=model)
    scores = [h["val"] for h in tuned.history if h["epoch"] >= 0]
    best_at = int(np.argmin(scores))
    assert len(scores) == best_at + 1 + 4 or len(scores) == 60
    assert len(scores) < 60
    assert tuned.val_score <= model.val_score + 1e-12  # never worse than init
