"""Sharded on-disk dataset of one axle: sampling with resume, append, and loading.

Experimental (see ``docs/guides/stability.md``). Layout: ``DIR/dataset/<axle>/shard_00000.npz``
plus ``meta.json``. Shards are written atomically and never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from suspkin import __version__
from suspkin.kinematics.rocker import KinematicsError
from suspkin.optimize.design import DesignSpace, InfeasibleDesign
from suspkin.optimize.outputs import git_hash
from suspkin.optimize.pool import make_pool, resolve_workers
from suspkin.optimize.spec import StudyConfigError, load_study
from suspkin.surrogate.channels import CornerSample, channels_for, extract_corner
from suspkin.surrogate.data import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Axle,
    AxleDataset,
    AxleGrid,
    AxleSpace,
    NoAxleVariablesError,
    axle_grids,
    lhs,
    resolve_lhs_optimization,
)

F64 = NDArray[np.float64]
_FEASIBLE_WARN = 0.3
_META_KEYS = (
    "axle",
    "channels",
    "param_names",
    "travel",
    "rack",
    "grid_hash",
    "design_space_hash",
    "vehicle_hash",
    "seed",
    "n_initial",
    "shard_size",
    "lhs_optimization",
)

__all__ = [
    "DatasetError",
    "SampleReport",
    "append_rows",
    "assign_split",
    "bytes_per_sample",
    "check_dataset_matches",
    "check_model_matches",
    "load_dataset",
    "sample_axle",
]


class DatasetError(ValueError):
    """The dataset directory is inconsistent, missing, or does not match the study."""


# --------------------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------------------


def assign_split(seed: int, rows: ArrayLike) -> NDArray[np.uint8]:
    """Deterministic 80 / 10 / 10 train / val / test split code per LHS row index.

    A splitmix64 hash of ``(seed, row)`` gives a uniform number per row; the result does not
    depend on the other rows or on the dataset size.

    Args:
        seed: Dataset seed.
        rows: LHS row indices ``(n,)``.

    Returns:
        ``uint8`` codes ``(n,)``: train, validation or test.
    """
    r = np.asarray(rows, dtype=np.int64).astype(np.uint64)
    golden = 0x9E3779B97F4A7C15
    mask64 = (1 << 64) - 1
    with np.errstate(over="ignore"):
        z = r + np.uint64(((int(seed) + 1) * golden) & mask64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    out = np.full(u.shape, SPLIT_TRAIN, dtype=np.uint8)
    out[u >= 0.8] = SPLIT_VAL
    out[u >= 0.9] = SPLIT_TEST
    return out


# --------------------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------------------


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _design_space_hash(space: DesignSpace, aspace: AxleSpace) -> str:
    axle = getattr(space.baseline, aspace.axle)
    slots = [s for s in space._slots if s.axle == aspace.axle]
    return _digest(
        {
            "variables": [s.var.model_dump(mode="json", exclude={"resolution"}) for s in slots],
            "names": list(aspace.names),
            "hardpoints": axle.hardpoints.model_dump(mode="json"),
            "right_overrides": {k: list(v) for k, v in axle.right_overrides.items()},
        }
    )


def bytes_per_sample(d: int, channels: int, grid: AxleGrid) -> int:
    """Uncompressed bytes of one sample in a shard (``Y`` float32, the rest as stored).

    Args:
        d: Number of design variables.
        channels: Number of channels.
        grid: Travel and rack grid.

    Returns:
        Size in bytes.
    """
    n_r, n_t = grid.shape
    # Y + mask + X + row + weight + feasible + split + source ('<U3' = 12 bytes)
    return channels * n_r * n_t * 4 + n_r * n_t + d * 8 + 8 + 8 + 1 + 1 + 12


def _meta_dir(out_dir: str | Path, axle: str) -> Path:
    return Path(out_dir) / "dataset" / axle


def _read_meta(adir: Path) -> dict[str, Any]:
    p = adir / "meta.json"
    if not p.is_file():
        raise DatasetError(f"{adir}: no meta.json (not a dataset directory)")
    try:
        meta: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{p}: cannot read ({exc})") from exc
    return meta


def _shard_files(adir: Path) -> list[Path]:
    return sorted(adir.glob("shard_*.npz"))


def _write_shard(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# row evaluation (worker side is top level and picklable)
# --------------------------------------------------------------------------------------

_RowResult = tuple[NDArray[np.float32], NDArray[np.bool_], bool]
_WORKER: tuple[DesignSpace, AxleSpace, AxleGrid] | None = None


def _context(study_path: str, axle: Axle) -> tuple[DesignSpace, AxleSpace, AxleGrid]:
    study, vehicle = load_study(study_path)
    if study.surrogate is None:
        raise StudyConfigError(f"{study_path}: study has no 'surrogate' block")
    space = DesignSpace(study, vehicle)
    return space, AxleSpace(space, axle), axle_grids(study.surrogate.grid)[axle]


def _init_worker(study_path: str, axle: Axle) -> None:
    global _WORKER
    _WORKER = _context(study_path, axle)


def _eval_row(space: DesignSpace, aspace: AxleSpace, grid: AxleGrid, x: F64) -> _RowResult:
    n_c = len(channels_for(aspace.axle))
    n_r, n_t = grid.shape
    try:
        vehicle = space.to_vehicle(aspace.to_full(x), quantise=False)  # training data: continuous
        s = extract_corner(vehicle, aspace.axle, grid)
    except (InfeasibleDesign, KinematicsError):
        return (
            np.full((n_c, n_r, n_t), np.nan, dtype=np.float32),
            np.zeros((n_r, n_t), dtype=bool),
            False,
        )
    return s.Y.astype(np.float32), s.mask, bool(s.feasible)


def _eval_worker(x: F64) -> _RowResult:
    if _WORKER is None:
        raise RuntimeError("worker not initialised")
    return _eval_row(*_WORKER, x)


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleReport:
    """Outcome of :func:`sample_axle`.

    ``n_samples`` and ``feasible_fraction`` cover the whole dataset directory (resumed rows
    included); ``n_new`` counts rows evaluated by this call.
    """

    axle: Axle
    n_samples: int
    n_new: int
    feasible_fraction: float
    seconds: float
    workers: int
    expected_bytes: int
    warning: str | None


def _tally(adir: Path) -> tuple[int, int]:
    n = n_ok = 0
    for p in _shard_files(adir):
        with np.load(p) as z:
            f = z["feasible"]
        n += int(f.size)
        n_ok += int(f.sum())
    return n, n_ok


def _check_meta(old: dict[str, Any], new: dict[str, Any], adir: Path) -> None:
    for k in _META_KEYS:
        if old.get(k) != new[k]:
            raise DatasetError(
                f"{adir}: dataset does not match the study ({k} differs); use a new --out directory"
            )


def _current_hashes(study_path: str | Path, axle: Axle) -> dict[str, str]:
    """Grid, design-space and vehicle hashes of the study for ``axle``."""
    sp = str(Path(study_path))
    study, vehicle = load_study(sp)
    if study.surrogate is None:
        raise StudyConfigError(f"{sp}: study has no 'surrogate' block")
    space = DesignSpace(study, vehicle)
    return {
        "grid_hash": axle_grids(study.surrogate.grid)[axle].grid_hash(),
        "design_space_hash": _design_space_hash(space, AxleSpace(space, axle)),
        "vehicle_hash": _digest(vehicle.model_dump(mode="json")),
    }


def _check_channels(where: Path, channels: object, axle: Axle) -> None:
    now = [c.name for c in channels_for(axle)]
    if list(channels) != now:  # type: ignore[call-overload]
        have = [] if channels is None else list(channels)  # type: ignore[call-overload]
        miss = sorted(set(now) - set(have))
        raise DatasetError(
            f"{where}: channels differ from the current registry for the {axle} axle "
            f"(missing {miss}, extra {sorted(set(have) - set(now))}); "
            "sample again (and retrain) with this version"
        )


def check_dataset_matches(study_path: str | Path, out_dir: str | Path, axle: Axle) -> None:
    """Raise :class:`DatasetError` if the dataset of ``axle`` was sampled from another study.

    Compares the grid, design-space and vehicle hashes of ``out_dir/dataset/<axle>/meta.json``
    with the study, and its channel list with the current registry; the message names the
    first differing item.

    Args:
        study_path: Study YAML.
        out_dir: Surrogate output folder containing ``dataset/``.
        axle: Axle to check.

    Raises:
        DatasetError: The dataset was built for another study (message names the item).
    """
    now = _current_hashes(study_path, axle)
    adir = _meta_dir(out_dir, axle)
    meta = _read_meta(adir)
    for key, val in now.items():
        if meta.get(key) != val:
            raise DatasetError(
                f"{adir}: dataset does not match the study ({key} differs); "
                "sample again with this study"
            )
    _check_channels(adir, meta.get("channels"), axle)


def check_model_matches(study_path: str | Path, model_dir: str | Path, axle: Axle) -> None:
    """Raise :class:`DatasetError` if the model in ``model_dir`` was trained for another study.

    Compares the hashes stored in the model meta (copied from its dataset) and the model
    channels with the current study and registry; the message names the differing item.

    Args:
        study_path: Study YAML.
        model_dir: Folder of the trained model.
        axle: Axle to check.

    Raises:
        DatasetError: The model was trained for another study (message names the item).
    """
    mdir = Path(model_dir)
    info = json.loads((mdir / "model.json").read_text(encoding="utf-8"))
    meta = info.get("meta", {})
    for key, val in _current_hashes(study_path, axle).items():
        if meta.get(key) != val:
            raise DatasetError(
                f"{mdir}: model does not match the study ({key} differs); "
                "sample and train again with this study"
            )
    _check_channels(mdir, info.get("channels"), axle)


def _lhs_matrix(adir: Path, aspace: AxleSpace, n: int, seed: int, opt: str) -> F64:
    """The LHS matrix (row 0 = baseline): loaded from ``lhs.npy`` or computed once and stored."""
    path = adir / "lhs.npy"
    base = aspace.baseline()
    if path.is_file():
        x: F64 = np.load(path)
        if x.shape != (n, aspace.n) or not np.allclose(x[0], base, atol=1e-12):
            raise DatasetError(f"{path}: stored LHS matrix does not match the study")
        return x
    x = lhs(n, aspace.n, seed, opt)
    x[0] = base
    tmp = path.with_name("lhs.tmp.npy")
    np.save(tmp, x)
    os.replace(tmp, path)
    return x


def sample_axle(
    study_path: str | Path,
    out_dir: str | Path,
    axle: Axle,
    *,
    n: int | None = None,
    workers: int | str | None = None,
    resume: bool = False,
    echo: Callable[[str], None] | None = None,
) -> SampleReport:
    """Sample the initial LHS dataset of one axle into ``out_dir/dataset/<axle>``.

    Row 0 is the baseline. Rows are solved in order, ``shard_size`` rows per shard (a shard
    is written only when complete). With ``resume`` the existing shards are kept and the
    rest is computed; the result equals an uninterrupted run. Without ``resume`` a non-empty
    dataset directory is refused.

    Args:
        n: Number of LHS rows (default: ``surrogate.dataset.n_initial``).
        workers: ``None`` or 1: in process; an integer or ``"auto"``: spawn process pool.
        echo: Receives progress lines (expected size, one line per shard).

    Raises:
        StudyConfigError: Study without ``surrogate`` block, asymmetric variable, or no variable on
            the axle
            (:class:`NoAxleVariablesError`).
        DatasetError: Existing directory refused, or resume with a changed study.
    """
    t0 = time.perf_counter()
    say = echo or (lambda _msg: None)
    sp = str(Path(study_path))
    study, vehicle = load_study(sp)
    if study.surrogate is None:
        raise StudyConfigError(f"{sp}: study has no 'surrogate' block")
    spec = study.surrogate.dataset
    n_total = spec.n_initial if n is None else int(n)
    if n_total < 1:
        raise ValueError("n must be >= 1")
    space = DesignSpace(study, vehicle)
    aspace = AxleSpace(space, axle)
    if aspace.n == 0:
        raise NoAxleVariablesError(f"{sp}: the study has no design variable on the {axle} axle")
    grid = axle_grids(study.surrogate.grid)[axle]
    chans = tuple(c.name for c in channels_for(axle))
    n_workers = 1 if workers is None else resolve_workers(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")

    meta: dict[str, Any] = {
        "axle": axle,
        "channels": list(chans),
        "param_names": list(aspace.names),
        "travel": [float(v) for v in grid.travel],
        "rack": [float(v) for v in grid.rack],
        "grid_hash": grid.grid_hash(),
        "design_space_hash": _design_space_hash(space, aspace),
        "vehicle_hash": _digest(vehicle.model_dump(mode="json")),
        "seed": spec.seed,
        "n_initial": n_total,
        "shard_size": spec.shard_size,
        "lhs_optimization": resolve_lhs_optimization(spec.lhs_optimization, n_total),
    }
    adir = _meta_dir(out_dir, axle)
    if resume and adir.is_dir() and any(adir.iterdir()):
        check_dataset_matches(sp, out_dir, axle)
        _check_meta(_read_meta(adir), meta, adir)
    else:
        if adir.is_dir() and any(adir.iterdir()):
            raise DatasetError(f"{adir} is not empty; use --resume or a new --out directory")
        adir.mkdir(parents=True, exist_ok=True)
        full_meta = {
            **meta,
            "package_version": __version__,
            "git_hash": git_hash(Path(__file__).resolve().parent),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (adir / "meta.json").write_text(json.dumps(full_meta, indent=2), encoding="utf-8")

    per = bytes_per_sample(aspace.n, len(chans), grid)
    say(f"{axle}: {n_total} samples, about {per * n_total} bytes uncompressed ({per} per sample)")

    x_all = _lhs_matrix(adir, aspace, n_total, spec.seed, meta["lhs_optimization"])
    size = spec.shard_size
    n_shards = math.ceil(n_total / size)
    todo = [k for k in range(n_shards) if not (adir / f"shard_{k:05d}.npz").exists()]
    n_new = 0
    pool: Executor | None = None
    try:
        if n_workers > 1 and todo:
            pool = make_pool(n_workers, _init_worker, (sp, axle))
        for k in todo:
            lo, hi = k * size, min((k + 1) * size, n_total)
            xs = [x_all[i] for i in range(lo, hi)]
            if pool is None:
                res = [_eval_row(space, aspace, grid, x) for x in xs]
            else:
                res = list(pool.map(_eval_worker, xs, chunksize=max(1, len(xs) // (4 * n_workers))))
            rows = np.arange(lo, hi, dtype=np.int64)
            feas = np.array([r[2] for r in res], dtype=bool)
            _write_shard(
                adir / f"shard_{k:05d}.npz",
                row=rows,
                X=x_all[lo:hi].astype(np.float64),
                Y=np.stack([r[0] for r in res]).astype(np.float32),
                mask=np.stack([r[1] for r in res]),
                feasible=feas,
                split=assign_split(spec.seed, rows),
                source=np.array(["lhs"] * (hi - lo)),
                weight=np.ones(hi - lo, dtype=np.float64),
            )
            n_new += hi - lo
            say(f"{axle}: rows {hi}/{n_total}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    total, ok = _tally(adir)
    frac = ok / total if total else 0.0
    warning = None
    if frac < _FEASIBLE_WARN:
        warning = (
            f"only {100 * frac:.1f} % of the {axle} samples are feasible (< 30 %): "
            f"the variable domains are probably too large"
        )
    return SampleReport(
        axle=axle,
        n_samples=total,
        n_new=n_new,
        feasible_fraction=frac,
        seconds=time.perf_counter() - t0,
        workers=n_workers,
        expected_bytes=per * n_total,
        warning=warning,
    )


# --------------------------------------------------------------------------------------
# append and load
# --------------------------------------------------------------------------------------


def append_rows(
    out_dir: str | Path,
    axle: Axle,
    X: ArrayLike,  # noqa: N803
    samples: Sequence[CornerSample],
    source: str,
    weight: float | ArrayLike = 1.0,
) -> Path:
    """Append extra (for example active-learning) rows as a new shard; returns its path.

    Rows get ``row = -1`` and split train. ``X`` is ``(m, d)`` in the axle unit cube.

    Args:
        out_dir: Surrogate output folder containing ``dataset/``.
        axle: Axle of the rows.
        X: Designs ``(m, d)`` in the axle unit cube.
        samples: ``m`` FOM corner samples matching ``X``.
        source: Label stored per row (for example ``al_3``).
        weight: Scalar or ``(m,)`` sample weights.

    Returns:
        Path of the new shard.
    """
    adir = _meta_dir(out_dir, axle)
    meta = _read_meta(adir)
    xm = np.atleast_2d(np.asarray(X, dtype=np.float64))
    m = len(samples)
    if xm.shape != (m, len(meta["param_names"])):
        raise DatasetError(f"X must have shape ({m}, {len(meta['param_names'])}), got {xm.shape}")
    if m == 0:
        raise DatasetError("no rows to append")
    if source == "lhs":
        raise DatasetError("appended rows need a source other than 'lhs'")
    chans = tuple(meta["channels"])
    if any(s.channels != chans for s in samples):
        raise DatasetError("sample channels differ from the dataset channels")
    w = np.broadcast_to(np.asarray(weight, dtype=np.float64), (m,)).copy()
    files = _shard_files(adir)
    nxt = max((int(p.stem.split("_")[1]) for p in files), default=-1) + 1
    path = adir / f"shard_{nxt:05d}.npz"
    _write_shard(
        path,
        row=np.full(m, -1, dtype=np.int64),
        X=xm,
        Y=np.stack([s.Y for s in samples]).astype(np.float32),
        mask=np.stack([s.mask for s in samples]),
        feasible=np.array([s.feasible for s in samples], dtype=bool),
        split=np.full(m, SPLIT_TRAIN, dtype=np.uint8),
        source=np.array([source] * m),
        weight=w,
    )
    return path


def load_dataset(out_dir: str | Path, axle: Axle) -> AxleDataset:
    """Concatenate all shards of an axle (in shard order) into an :class:`AxleDataset`.

    Args:
        out_dir: Surrogate output folder containing ``dataset/``.
        axle: Axle to load.

    Raises:
        DatasetError: Missing directory, no shards, or shards that do not match ``meta.json``.
    """
    adir = _meta_dir(out_dir, axle)
    meta = _read_meta(adir)
    if meta.get("axle") != axle:
        raise DatasetError(f"{adir}: meta.json is for axle {meta.get('axle')!r}")
    files = _shard_files(adir)
    if not files:
        raise DatasetError(f"{adir}: no shards")
    d = len(meta["param_names"])
    shape = (len(meta["channels"]), len(meta["rack"]), len(meta["travel"]))
    parts: dict[str, list[NDArray[Any]]] = {
        k: [] for k in ("row", "X", "Y", "mask", "feasible", "split", "source", "weight")
    }
    for p in files:
        with np.load(p) as z:
            for k in parts:
                parts[k].append(z[k])
        n = parts["X"][-1].shape[0]
        if parts["X"][-1].shape != (n, d) or parts["Y"][-1].shape != (n, *shape):
            raise DatasetError(f"{p}: shapes do not match meta.json")
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    lhs_rows = cat["row"][cat["row"] >= 0]
    if np.unique(lhs_rows).size != lhs_rows.size:
        raise DatasetError(f"{adir}: duplicate LHS rows")
    return AxleDataset(
        axle=axle,
        X=cat["X"].astype(np.float64),
        Y=cat["Y"].astype(np.float32),
        mask=cat["mask"].astype(bool),
        feasible=cat["feasible"].astype(bool),
        split=cat["split"].astype(np.uint8),
        source=cat["source"],
        weight=cat["weight"].astype(np.float64),
        channels=tuple(meta["channels"]),
        param_names=tuple(meta["param_names"]),
        travel=np.array(meta["travel"], dtype=np.float64),
        rack=np.array(meta["rack"], dtype=np.float64),
        meta=meta,
    )
