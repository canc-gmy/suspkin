"""Training, evaluation, persistence and batched inference of the per-axle surrogate.

Experimental, needs torch.

Install note: the ``surrogate`` extra resolves to the CUDA 12.4 torch wheels on Windows.
They also run on machines without an NVIDIA GPU (CUDA is then simply unavailable and
``device="auto"`` falls back to the CPU), so one extra is enough everywhere.
"""

# ruff: noqa: N803  (X, Y, Z follow the dataset contract)
from __future__ import annotations

import os

# Must be set before any CUDA context exists (deterministic cuBLAS).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import copy
import json
import logging
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.nn import functional as F  # noqa: N812  # noqa: E402

from suspkin.surrogate.channels import CHANNELS
from suspkin.surrogate.data import SPLIT_TRAIN, SPLIT_VAL, AxleDataset
from suspkin.surrogate.model import ChannelScaler, MLPConfig, PodBasis, SurrogateMLP

_MAX_RESIDENT_FRACTION = 0.35
_LOG = logging.getLogger(__name__)
_LOGGED: set[str] = set()


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser, schedule and loss settings."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 30
    warmup_epochs: int = 5
    lam_feas: float = 0.1
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    finetune_lr_factor: float = 0.1  # lr multiplier when fine-tuning from `init` (no warm-up)
    num_threads: int | None = None  # CPU only; None leaves the torch default


def _log_once(msg: str) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        _LOG.info(msg)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``"auto"`` / ``"cpu"`` / ``"cuda"`` to a torch device.

    ``"auto"`` picks CUDA when available, else the CPU; the choice is logged once.
    An unavailable ``"cuda"`` raises ``RuntimeError``.
    """
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device 'cuda' requested but no CUDA device is available: use device "
                "'auto' or 'cpu', or install the CUDA wheels of torch"
            )
        return torch.device("cuda")
    if device != "auto":
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {device!r}")
    if torch.cuda.is_available():
        _log_once(f"using CUDA device {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    _log_once("no CUDA device found, training on CPU")
    return torch.device("cpu")


def channel_tolerances(channels: tuple[str, ...]) -> NDArray[np.float64]:
    """Registry tolerances for channel names (raises on unknown names)."""
    tol = {c.name: c.tol for c in CHANNELS}
    missing = [c for c in channels if c not in tol]
    if missing:
        raise ValueError(f"channels not in registry: {missing}")
    return np.array([tol[c] for c in channels], dtype=np.float64)


# --------------------------------------------------------------------------- model container


@dataclass
class TrainedModel:
    """A trained axle network with everything needed to predict and to fine-tune."""

    axle: str
    mcfg: MLPConfig
    tcfg: TrainConfig
    scaler: ChannelScaler
    pod: PodBasis | None
    state_dict: dict[str, torch.Tensor]
    channels: tuple[str, ...]
    param_names: tuple[str, ...]
    travel: NDArray[np.float64]
    rack: NDArray[np.float64]
    history: list[dict[str, float]] = field(default_factory=list)
    val_score: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def curve_shape(self) -> tuple[int, int, int]:
        """``(C, R, T)``."""
        return len(self.channels), len(self.rack), len(self.travel)

    def build_network(self, device: torch.device | str = "cpu") -> SurrogateMLP:
        """Instantiate the network with the stored weights, in eval mode."""
        c, r, t = self.curve_shape
        n_out = self.pod.k if self.pod is not None else c * r * t
        net = SurrogateMLP(len(self.param_names), n_out, r * t, self.mcfg)
        net.load_state_dict(self.state_dict)
        return net.to(device).eval()

    def save(self, directory: str | Path) -> None:
        """Write ``model.pt``, ``model.json`` and ``norm.npz`` into ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, out / "model.pt")
        info = {
            "axle": self.axle,
            "mcfg": asdict(self.mcfg),
            "tcfg": asdict(self.tcfg),
            "channels": list(self.channels),
            "param_names": list(self.param_names),
            "history": self.history,
            "val_score": self.val_score,
            "meta": self.meta,
            "has_pod": self.pod is not None,
        }
        (out / "model.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        arrays: dict[str, NDArray[Any]] = {
            **self.scaler.to_dict(),
            "travel": self.travel,
            "rack": self.rack,
        }
        if self.pod is not None:
            arrays.update(self.pod.to_dict())
        np.savez(out / "norm.npz", **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, directory: str | Path, device: str = "auto") -> TrainedModel:
        """Inverse of :meth:`save`; weights saved on any device load on ``device``."""
        src = Path(directory)
        info = json.loads((src / "model.json").read_text(encoding="utf-8"))
        with np.load(src / "norm.npz") as z:
            arrays = {k: z[k] for k in z.files}
        dev = resolve_device(device)
        state = torch.load(src / "model.pt", map_location=dev, weights_only=True)
        state = {k: v.cpu() for k, v in state.items()}
        return cls(
            axle=info["axle"],
            mcfg=MLPConfig(**info["mcfg"]),
            tcfg=TrainConfig(**info["tcfg"]),
            scaler=ChannelScaler.from_dict(arrays),
            pod=PodBasis.from_dict(arrays) if info["has_pod"] else None,
            state_dict=state,
            channels=tuple(info["channels"]),
            param_names=tuple(info["param_names"]),
            travel=arrays["travel"],
            rack=arrays["rack"],
            history=info["history"],
            val_score=float(info["val_score"]),
            meta=info["meta"],
        )


# --------------------------------------------------------------------------- loss


def masked_loss(
    reg: torch.Tensor,
    logits: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    lam_feas: float,
    pod_components: torch.Tensor | None,
    n_channels: int,
) -> torch.Tensor:
    """Weighted masked MSE on standardised curves plus ``lam_feas`` * validity BCE.

    ``z`` is ``(B, C*R*T)``, ``mask`` ``(B, R*T)`` bool, ``w`` ``(B,)``. Masked entries of
    ``z`` never influence the value or the gradients. With ``pod_components`` the
    regression output holds POD coefficients that are decoded first.
    """
    pred = reg @ pod_components if pod_components is not None else reg
    b = z.shape[0]
    se = (pred.reshape(b, n_channels, -1) - z.reshape(b, n_channels, -1)) ** 2
    m = mask[:, None, :]
    se = torch.where(m, se, torch.zeros_like(se))
    cnt = (mask.sum(dim=1) * n_channels).clamp(min=1).to(se.dtype)
    mse = se.sum(dim=(1, 2)) / cnt
    bce = F.binary_cross_entropy_with_logits(logits, mask.to(logits.dtype), reduction="none").mean(
        dim=1
    )
    wsum = w.sum().clamp(min=1e-12)
    return (w * (mse + lam_feas * bce)).sum() / wsum


# --------------------------------------------------------------------------- data


class _Split:
    """Tensors of one split, resident on the device or in (pinned) host memory."""

    def __init__(
        self,
        ds: AxleDataset,
        scaler: ChannelScaler,
        device: torch.device,
        resident: bool,
    ) -> None:
        n = ds.n
        z = scaler.transform(ds.Y, ds.mask).reshape(n, -1)
        tensors = (
            torch.from_numpy(ds.X.astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(z)),
            torch.from_numpy(ds.mask.reshape(n, -1)),
            torch.from_numpy(ds.weight.astype(np.float32)),
        )
        self.device = device
        self.n = n
        if resident:
            self.tensors = tuple(t.to(device) for t in tensors)
        elif device.type == "cuda":
            self.tensors = tuple(t.pin_memory() for t in tensors)
        else:
            self.tensors = tensors
        self.resident = resident

    def batches(
        self, batch_size: int, gen: torch.Generator | None
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        idx = torch.randperm(self.n, generator=gen) if gen is not None else torch.arange(self.n)
        if self.resident:
            idx = idx.to(self.device)
        for i in range(0, self.n, batch_size):
            sel = idx[i : i + batch_size]
            yield tuple(
                t[sel].to(self.device, non_blocking=self.device.type == "cuda")
                for t in self.tensors
            )

    @classmethod
    def concat(cls, a: _Split, b: _Split) -> _Split:
        """Rows of ``a`` followed by rows of ``b`` (same device and residency)."""
        out = cls.__new__(cls)
        out.device, out.resident, out.n = a.device, a.resident, a.n + b.n
        cat = tuple(torch.cat([x, y]) for x, y in zip(a.tensors, b.tensors, strict=True))
        if not a.resident and a.device.type == "cuda":
            cat = tuple(t.pin_memory() for t in cat)
        out.tensors = cat
        return out


class FinetuneCache:
    """Tensors reused by successive fine-tunes of one axle (active learning).

    Holds the validation split and a fixed set of replayed train rows, normalised with the
    scaler of ``init`` and moved to the device once, plus the growing set of active-learning
    rows (:meth:`add_rows`). :func:`train` with ``cache=`` then skips the per-call data
    assembly; the resulting splits equal the ones built from the same rows directly.
    """

    def __init__(
        self,
        init: TrainedModel,
        static: AxleDataset,
        device: torch.device | str = "auto",
    ) -> None:
        """``static``: the validation rows (split val) and the replay rows (split train)."""
        if init.channels != static.channels or init.param_names != static.param_names:
            raise ValueError("init model channels / param_names do not match the dataset")
        va, rp = static.subset(SPLIT_VAL), static.subset(SPLIT_TRAIN)
        if va.n == 0:
            raise ValueError("fine-tune cache needs a non-empty val split")
        self.device = resolve_device(device) if not isinstance(device, torch.device) else device
        self.scaler = init.scaler
        self.resident = _fits_on_device([va, rp], self.device)
        self.val = _Split(va, self.scaler, self.device, self.resident)
        self.replay = _Split(rp, self.scaler, self.device, self.resident) if rp.n else None
        self.template = va.subset(np.arange(va.n) == 0)  # one row: shapes, names, meta
        self.al: _Split | None = None

    @property
    def n_al(self) -> int:
        """Number of cached active-learning rows."""
        return self.al.n if self.al is not None else 0

    def add_rows(self, rows: AxleDataset) -> None:
        """Append active-learning rows (they are trained on and scored as ``extra_val``)."""
        if rows.n == 0:
            return
        new = _Split(rows, self.scaler, self.device, self.resident)
        self.al = new if self.al is None else _Split.concat(self.al, new)

    def splits(self) -> tuple[_Split, _Split, _Split | None]:
        """``(train, val, extra_val)``: train is the replay rows followed by the AL rows."""
        if self.replay is None and self.al is None:
            raise ValueError("fine-tune cache has no train rows")
        if self.replay is None:
            assert self.al is not None
            return self.al, self.val, self.al
        if self.al is None:
            return self.replay, self.val, None
        return _Split.concat(self.replay, self.al), self.val, self.al


def _fits_on_device(arrays: list[AxleDataset], device: torch.device) -> bool:
    if device.type != "cuda":
        return True
    need = sum(a.Y.nbytes + a.X.nbytes for a in arrays)
    free, _ = torch.cuda.mem_get_info(device)
    return need < _MAX_RESIDENT_FRACTION * free


def _channel_ratios(
    net: SurrogateMLP,
    split: _Split,
    pod_t: torch.Tensor | None,
    n_channels: int,
    std: NDArray[np.float64],
    tol: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per channel physical masked RMSE / tol, computed in standardised space."""
    net.eval()
    se = torch.zeros(n_channels, dtype=torch.float64, device=split.device)
    cnt = torch.zeros((), dtype=torch.float64, device=split.device)
    with torch.no_grad():
        for xb, z, m, _w in split.batches(4096, None):
            reg, _logits = net(xb)
            pred = reg @ pod_t if pod_t is not None else reg
            b = z.shape[0]
            d = (pred.reshape(b, n_channels, -1) - z.reshape(b, n_channels, -1)) ** 2
            d = torch.where(m[:, None, :], d, torch.zeros_like(d))
            se += d.sum(dim=(0, 2)).double()
            cnt += m.sum().double()
    rmse = torch.sqrt(se / cnt.clamp(min=1.0)).cpu().numpy() * std
    out: NDArray[np.float64] = rmse / tol
    return out


def _score_split(
    net: SurrogateMLP,
    split: _Split,
    pod_t: torch.Tensor | None,
    n_channels: int,
    std: NDArray[np.float64],
    tol: NDArray[np.float64],
) -> float:
    """Score: mean over channels of :func:`_channel_ratios` (the search objective)."""
    return float(np.mean(_channel_ratios(net, split, pod_t, n_channels, std, tol)))


# --------------------------------------------------------------------------- train


def _lr_at(epoch: int, tcfg: TrainConfig, finetune: bool = False) -> float:
    lr = tcfg.lr * tcfg.finetune_lr_factor if finetune else tcfg.lr
    warm = 0 if finetune else tcfg.warmup_epochs
    if epoch < warm:
        return lr * (epoch + 1) / warm
    span = max(1, tcfg.max_epochs - warm)
    frac = min(1.0, (epoch - warm) / span)
    return lr * max(0.5 * (1.0 + math.cos(math.pi * frac)), 0.01)


def train(
    ds: AxleDataset,
    mcfg: MLPConfig,
    tcfg: TrainConfig,
    *,
    init: TrainedModel | None = None,
    report: Callable[[int, float], None] | None = None,
    extra_val: AxleDataset | None = None,
    cache: FinetuneCache | None = None,
) -> TrainedModel:
    """Fit on the train split, early-stop on the validation score, restore the best weights.

    With ``init`` the weights, architecture, scaler and POD of that model are reused (not
    refitted): this is the active-learning update. In that case the learning rate is
    ``lr * finetune_lr_factor`` without warm-up, and the returned model is never worse (on
    the val split) than ``init``: if no epoch beats it, the init weights are returned and
    ``history`` starts with an ``epoch = -1`` entry. ``report(epoch, val_score)`` is called
    after every epoch (it may raise to abort, e.g. for pruning).

    ``extra_val`` (default ``None``: no change) is a second scoring set, all rows used whatever
    their split: the early-stopping score becomes ``mean(val score, extra_val score)``. The
    active-learning loop passes its new points, so the update cannot silently ignore them.

    ``cache`` (needs ``init``; ``ds`` then only supplies shapes and metadata, e.g.
    ``cache.template``) takes the train / val / extra splits from a :class:`FinetuneCache`
    instead of building them. With ``init`` the patience counts epochs without a new best
    score of this run (the score of ``init`` is not the reference), so a slow start still
    gets its ``patience`` epochs.
    """
    torch.manual_seed(tcfg.seed)
    # Process-wide switch (not restored): deterministic kernels where available, warn otherwise.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(tcfg.device)
    if device.type == "cpu" and tcfg.num_threads:
        torch.set_num_threads(tcfg.num_threads)

    tol = channel_tolerances(ds.channels)
    n_c = len(ds.channels)
    tr: AxleDataset | None = None
    if cache is not None:
        if init is None:
            raise ValueError("cache needs init")
        a, b = cache.scaler.to_dict(), init.scaler.to_dict()
        if cache.device != device or a.keys() != b.keys():
            raise ValueError("fine-tune cache was built for another device or model")
        if not all(np.array_equal(a[k], b[k]) for k in a):
            raise ValueError("fine-tune cache was built for another scaler")
    else:
        tr, va = ds.subset(SPLIT_TRAIN), ds.subset(SPLIT_VAL)
        if tr.n == 0 or va.n == 0:
            raise ValueError("dataset needs non-empty train and val splits")

    if init is not None:
        if init.channels != ds.channels or init.param_names != ds.param_names:
            raise ValueError("init model channels / param_names do not match the dataset")
        mcfg, scaler, pod = init.mcfg, init.scaler, init.pod
    else:
        assert tr is not None
        scaler = ChannelScaler.fit(tr.Y, tr.mask)
        pod = None
        if mcfg.output == "pod":
            pod = PodBasis.fit(scaler.transform(tr.Y, tr.mask), mcfg.pod_k, seed=tcfg.seed)

    pod_t = (
        torch.from_numpy(pod.components.astype(np.float32)).to(device) if pod is not None else None
    )
    if cache is not None:
        tr_s, va_s, ex_s = cache.splits()
    else:
        assert tr is not None
        resident = _fits_on_device([tr, va], device)
        tr_s, va_s = _Split(tr, scaler, device, resident), _Split(va, scaler, device, resident)
        ex_s = (
            _Split(extra_val, scaler, device, resident)
            if extra_val is not None and extra_val.n > 0
            else None
        )

    def val_score(net_: SurrogateMLP) -> float:
        v = _score_split(net_, va_s, pod_t, n_c, scaler.std, tol)
        if ex_s is None:
            return v
        return 0.5 * (v + _score_split(net_, ex_s, pod_t, n_c, scaler.std, tol))

    n_out = pod.k if pod is not None else n_c * len(ds.rack) * len(ds.travel)
    net = SurrogateMLP(ds.X.shape[1], n_out, len(ds.rack) * len(ds.travel), mcfg)
    if init is not None:
        net.load_state_dict(init.state_dict)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    gen = torch.Generator().manual_seed(tcfg.seed)

    best = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    history: list[dict[str, float]] = []
    if init is not None:
        best = val_score(net)
        history.append({"epoch": -1.0, "train": float("nan"), "val": best})
    init_score = best
    run_best = float("inf")  # best epoch score of this run: the patience reference
    stale = 0
    for epoch in range(tcfg.max_epochs):
        for g in opt.param_groups:
            g["lr"] = _lr_at(epoch, tcfg, init is not None)
        net.train()
        tot, nb = 0.0, 0
        for xb, zb, mb, wb in tr_s.batches(tcfg.batch_size, gen):
            opt.zero_grad(set_to_none=True)
            reg, logits = net(xb)
            loss = masked_loss(reg, logits, zb, mb, wb, tcfg.lam_feas, pod_t, n_c)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            tot += float(loss.detach())
            if not math.isfinite(tot):
                raise FloatingPointError("non-finite training loss")
            nb += 1
        score = val_score(net)
        history.append({"epoch": float(epoch), "train": tot / max(nb, 1), "val": score})
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if score < run_best:
            run_best, stale = score, 0
        else:
            stale += 1
        if report is not None:
            report(epoch, score)
        if stale >= tcfg.patience:
            break

    # Per-channel ratios of the restored best weights (the score is only their mean).
    net.load_state_dict(best_state)
    ratios = _channel_ratios(net, va_s, pod_t, n_c, scaler.std, tol)
    return TrainedModel(
        axle=ds.axle,
        mcfg=mcfg,
        tcfg=tcfg,
        scaler=scaler,
        pod=pod,
        state_dict=best_state,
        channels=ds.channels,
        param_names=ds.param_names,
        travel=ds.travel,
        rack=ds.rack,
        history=history,
        val_score=best,
        meta={
            **copy.deepcopy(ds.meta),
            "val_channel_ratio": {c: float(r) for c, r in zip(ds.channels, ratios, strict=True)},
            **({"kept_init_weights": True} if init is not None and best == init_score else {}),
        },
    )


# --------------------------------------------------------------------------- prediction / eval


def predict_physical(
    model: TrainedModel,
    X: NDArray[np.float64],
    *,
    net: SurrogateMLP | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 4096,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Curves ``(B, C, R, T)`` in physical units and validity probabilities ``(B, R, T)``."""
    dev = torch.device(device)
    net = net if net is not None else model.build_network(dev)
    c, r, t = model.curve_shape
    pod_t = (
        torch.from_numpy(model.pod.components.astype(np.float32)).to(dev)
        if model.pod is not None
        else None
    )
    ys: list[NDArray[np.float32]] = []
    ps: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size]).astype(np.float32))
            reg, logits = net(xb.to(dev))
            pred = reg @ pod_t if pod_t is not None else reg
            ys.append(pred.reshape(-1, c, r, t).cpu().numpy())
            ps.append(torch.sigmoid(logits).reshape(-1, r, t).cpu().numpy())
    if not ys:
        return np.zeros((0, c, r, t)), np.zeros((0, r, t))
    y = model.scaler.inverse(np.concatenate(ys))
    return y, np.concatenate(ps).astype(np.float64)


@dataclass
class EvalReport:
    """Validation report in engineering units."""

    channels: tuple[str, ...]
    tol: NDArray[np.float64]
    per_channel_rmse: NDArray[np.float64]
    per_channel_ratio: NDArray[np.float64]
    score: float
    worst: list[tuple[int, float]]
    validity_accuracy: float
    validity_auc: float
    n: int

    def to_frame(self) -> pd.DataFrame:
        """One row per channel: ``channel, rmse, tol, ratio``."""
        return pd.DataFrame(
            {
                "channel": list(self.channels),
                "rmse": self.per_channel_rmse,
                "tol": self.tol,
                "ratio": self.per_channel_ratio,
            }
        )


def _auc(p: NDArray[np.float64], y: NDArray[np.bool_]) -> float:
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(
    model: TrainedModel,
    ds: AxleDataset,
    split: int | None = SPLIT_VAL,
    *,
    device: torch.device | str = "cpu",
    n_worst: int = 10,
) -> EvalReport:
    """Masked RMSE per channel (physical units) and ``mean(RMSE/tol)`` on a split.

    ``split=None`` uses the whole dataset. Worst-sample indices refer to the subset.
    """
    sub = ds if split is None else ds.subset(int(split))
    if sub.channels != model.channels:
        raise ValueError("dataset channels do not match the model")
    tol = channel_tolerances(model.channels)
    y, p = predict_physical(model, sub.X, device=device)
    m = sub.mask[:, None, :, :]
    err = np.where(m, y - np.nan_to_num(sub.Y.astype(np.float64)), 0.0) ** 2
    cnt = max(int(sub.mask.sum()), 1)
    rmse = np.sqrt(err.sum(axis=(0, 2, 3)) / cnt)
    ratio = rmse / tol
    per_n = sub.mask.sum(axis=(1, 2))
    per_sample = np.sqrt(err.sum(axis=(2, 3)) / np.maximum(per_n, 1)[:, None]) / tol
    ps = np.where(per_n > 0, per_sample.mean(axis=1), -np.inf)
    order = np.argsort(-ps)[:n_worst]
    worst = [(int(i), float(ps[i])) for i in order if np.isfinite(ps[i])]
    acc = float(((p > 0.5) == sub.mask).mean()) if sub.n else float("nan")
    auc = _auc(p.ravel(), sub.mask.ravel()) if sub.n else float("nan")
    return EvalReport(
        channels=model.channels,
        tol=tol,
        per_channel_rmse=rmse,
        per_channel_ratio=ratio,
        score=float(ratio.mean()),
        worst=worst,
        validity_accuracy=acc,
        validity_auc=auc,
        n=sub.n,
    )


# --------------------------------------------------------------------------------------
# batched inference
# --------------------------------------------------------------------------------------


class SurrogatePredictor:
    """Evaluate a trained axle network on design vectors in the unit cube."""

    def __init__(self, model: TrainedModel, device: str = "auto", batch_size: int = 8192) -> None:
        self.model = model
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._net = model.build_network(self.device)

    @classmethod
    def load(
        cls, model_dir: str | Path, device: str = "auto", batch_size: int = 8192
    ) -> SurrogatePredictor:
        """Load ``model.pt`` / ``model.json`` / ``norm.npz`` from ``model_dir``."""
        return cls(
            TrainedModel.load(model_dir, device=device), device=device, batch_size=batch_size
        )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Expected design parameter order."""
        return self.model.param_names

    def predict(
        self, X: NDArray[np.float64], *, param_names: Sequence[str] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(Y (B, C, R, T) physical units float64, p_valid (B, R, T))``.

        Raises ``ValueError`` if ``X`` is not ``(B, d)`` or ``param_names`` differ.
        """
        x = np.asarray(X, dtype=np.float64)
        d = len(self.model.param_names)
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError(f"X must have shape (B, {d}), got {x.shape}")
        if param_names is not None and tuple(param_names) != self.model.param_names:
            raise ValueError("param_names do not match the model")
        with torch.no_grad():
            return predict_physical(
                self.model, x, net=self._net, device=self.device, batch_size=self.batch_size
            )
