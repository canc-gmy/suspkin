"""Normalisation, loss and network tests."""

import numpy as np
import pytest

from suspkin.surrogate.data import SPLIT_TEST, SPLIT_VAL, AxleDataset
from suspkin.surrogate.normalise import ChannelScaler, PodBasis

torch = pytest.importorskip("torch")

from suspkin.surrogate.model import MLPConfig, SurrogateMLP  # noqa: E402
from suspkin.surrogate.train import masked_loss, resolve_device  # noqa: E402


def make_ds(n: int = 300, seed: int = 0, d: int = 4) -> AxleDataset:
    rng = np.random.default_rng(seed)
    r, t = 2, 7
    x = rng.random((n, d))
    travel = np.linspace(-0.03, 0.03, t)
    rack = np.array([-0.01, 0.01])
    tt = travel[None, :] / 0.03
    rr = rack[:, None] / 0.01
    y = np.zeros((n, 3, r, t))
    y[:, 0] = 0.02 * (x[:, 0, None, None] * tt + 0.5 * x[:, 1, None, None] * rr)
    y[:, 1] = 0.01 * (np.sin(2 * x[:, 2, None, None]) * tt**2 + x[:, 3, None, None] * rr * tt)
    y[:, 2] = 0.03 * (x[:, 0, None, None] + x[:, 3, None, None] ** 2) * (1 + 0.3 * tt + 0 * rr)
    mask = np.ones((n, r, t), dtype=bool)
    bad = rng.random(n) < 0.2
    mask[bad, :, -2:] = False
    y[~np.broadcast_to(mask[:, None], y.shape)] = np.nan
    split = np.zeros(n, dtype=np.uint8)
    split[int(0.7 * n) : int(0.85 * n)] = SPLIT_VAL
    split[int(0.85 * n) :] = SPLIT_TEST
    return AxleDataset(
        axle="front",
        X=x,
        Y=y.astype(np.float32),
        mask=mask,
        feasible=mask.all(axis=(1, 2)),
        split=split,
        source=np.array(["lhs"] * n),
        weight=np.ones(n),
        channels=("camber", "toe", "caster"),
        param_names=tuple(f"p{i}" for i in range(d)),
        travel=travel,
        rack=rack,
        meta={"hash": "abc"},
    )


def test_scaler_roundtrip_and_stats() -> None:
    ds = make_ds()
    sc = ChannelScaler.fit(ds.Y, ds.mask)
    z = sc.transform(ds.Y, ds.mask)
    assert np.isfinite(z).all()
    m = np.broadcast_to(ds.mask[:, None], z.shape)
    for c in range(3):
        assert abs(z[:, c][m[:, c]].mean()) < 1e-5
        assert abs(z[:, c][m[:, c]].std() - 1) < 1e-4
    back = sc.inverse(z)
    assert np.allclose(back[m], ds.Y[m], atol=1e-6)
    sc2 = ChannelScaler.from_dict(sc.to_dict())
    assert np.array_equal(sc2.mean, sc.mean)


def test_pod_roundtrip_full_rank() -> None:
    ds = make_ds(n=60)
    sc = ChannelScaler.fit(ds.Y, ds.mask)
    z = sc.transform(ds.Y, ds.mask)
    pod = PodBasis.fit(z, k=10_000)
    assert pod.k == 42  # rank bound: C*R*T
    assert pod.explained_total == pytest.approx(1.0)
    rec = pod.decode(pod.encode(z)).reshape(z.shape)
    assert np.allclose(rec, z, atol=1e-4)
    small = PodBasis.fit(z, k=5)
    assert small.explained_total < 1.0
    pod2 = PodBasis.from_dict(small.to_dict())
    assert np.array_equal(pod2.components, small.components)


@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh"])
def test_mlp_shapes(act: str) -> None:
    cfg = MLPConfig(
        n_layers=3,
        width=16,
        activation=act,  # type: ignore[arg-type]
        residual=True,
        layernorm=True,
        dropout=0.1,
    )
    net = SurrogateMLP(4, 42, 14, cfg)
    reg, logit = net(torch.rand(5, 4))
    assert reg.shape == (5, 42)
    assert logit.shape == (5, 14)


def _loss_inputs() -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    b, c, rt = 6, 3, 14
    reg = torch.randn(b, c * rt, generator=g, requires_grad=True)
    logits = torch.randn(b, rt, generator=g, requires_grad=True)
    z = torch.randn(b, c * rt, generator=g)
    mask = torch.rand(b, rt, generator=g) > 0.3
    return reg, logits, z, mask


def test_masked_points_do_not_affect_loss_or_gradients() -> None:
    reg, logits, z, mask = _loss_inputs()
    w = torch.ones(6)
    grads = []
    for z_in in (z, torch.where(mask.repeat(1, 3), z, torch.full_like(z, 1e6))):
        reg.grad = None
        loss = masked_loss(reg, logits, z_in, mask, w, 0.1, None, 3)
        loss.backward()
        assert reg.grad is not None
        grads.append(reg.grad.clone())
    assert torch.allclose(grads[0], grads[1])


def test_sample_weights_change_loss() -> None:
    reg, logits, z, mask = _loss_inputs()
    ones = torch.ones(6)
    base = masked_loss(reg, logits, z, mask, ones, 0.1, None, 3)
    scaled = masked_loss(reg, logits, z, mask, 5 * ones, 0.1, None, 3)
    assert torch.isclose(base, scaled)  # uniform weights cancel
    w = ones.clone()
    w[0] = 10.0
    weighted = masked_loss(reg, logits, z, mask, w, 0.1, None, 3)
    per = [
        masked_loss(
            reg[i : i + 1], logits[i : i + 1], z[i : i + 1], mask[i : i + 1], ones[:1], 0.1, None, 3
        )
        for i in range(6)
    ]
    expect = (10 * per[0] + sum(per[1:])) / 15
    assert torch.isclose(weighted, expect)


def test_resolve_device_cpu_and_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="auto"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        resolve_device("tpu")


@pytest.mark.parametrize("fill", [123.0, np.nan])
def test_scaler_ignores_masked_entries_and_stays_finite(fill: float) -> None:
    ds = make_ds()
    sc = ChannelScaler.fit(ds.Y, ds.mask)
    y2 = ds.Y.copy()
    y2[~np.broadcast_to(ds.mask[:, None], y2.shape)] = fill
    sc2 = ChannelScaler.fit(y2, ds.mask)
    assert np.isfinite(sc2.mean).all()
    assert np.array_equal(sc.mean, sc2.mean)
    assert np.array_equal(sc.std, sc2.std)
    z = sc2.transform(y2, ds.mask)
    assert np.isfinite(z).all()
    pod = PodBasis.fit(z, k=8)
    assert np.isfinite(pod.components).all()
    assert np.isfinite(pod.encode(z)).all()
    b = len(z)
    reg = torch.zeros(b, z[0].size, requires_grad=True)
    logits = torch.zeros(b, ds.mask.shape[1] * ds.mask.shape[2])
    loss = masked_loss(
        reg,
        logits,
        torch.from_numpy(z.reshape(b, -1)),
        torch.from_numpy(ds.mask.reshape(b, -1)),
        torch.ones(b),
        0.1,
        None,
        3,
    )
    assert torch.isfinite(loss)
