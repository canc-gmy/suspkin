"""Golden-snapshot helpers: write (``--update-golden``) or compare CSV / JSON snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from suspkin.metrics.table import GRADIENTS

GOLDEN = Path(__file__).resolve().parent / "golden"
HINT = "run `python -m uv run pytest tests/regression --update-golden` to create it"
RTOL = 1e-6
ATOL = 1e-9
# Gradient columns are finite differences of the sweep: near-zero values carry round-off
# noise that differs a little between platforms/BLAS builds, so they get a looser atol.
ATOL_GRADIENT = 1e-8


# Finite-difference gradient columns added by ``add_gradients`` (plus the actuation
# derivatives) and the polynomial-slope keys of the vehicle summary.
_TABLE_GRADIENTS = {name for cols in GRADIENTS.values() for name in cols.values()} | {
    "mr_spring",
    "mr_progression",
    "wheel_load_spring",
    "wheel_rate_linear",
    "wheel_rate",
    "ride_rate",
}
_SUMMARY_GRADIENTS = {
    "roll_camber",
    "roll_steer",
    "rc_migration",
    "rc_height_curvature",
}
_GRADIENT_NAMES = _TABLE_GRADIENTS | _SUMMARY_GRADIENTS


def _atol(col: str) -> float:
    """Loose atol for gradient columns (also ``FL_``-prefixed / dotted summary keys)."""
    base = col.rsplit(".", 1)[-1]
    if base in _GRADIENT_NAMES:
        return ATOL_GRADIENT
    return ATOL_GRADIENT if any(base.endswith(f"_{g}") for g in _GRADIENT_NAMES) else ATOL


def refuse_in_ci() -> None:
    """``--update-golden`` must never rewrite snapshots on CI."""
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(
            "--update-golden is refused when CI / GITHUB_ACTIONS is set: "
            "regenerate snapshots locally and commit them",
            pytrace=False,
        )


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def write_frame(name: str, df: pd.DataFrame) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLDEN / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")


def check_frames(frames: dict[str, pd.DataFrame], update: bool) -> None:
    """Write all snapshots (and skip) when ``update``, else compare each one."""
    if update:
        refuse_in_ci()
        for name, df in frames.items():
            write_frame(name, df)
        written = ", ".join(f"{GOLDEN.name}/{n}.csv" for n in frames)
        print(f"golden snapshots written: {written}")
        pytest.skip(f"golden snapshots written: {written}")
    for name, df in frames.items():
        _compare_frame(name, df)


def _compare_frame(name: str, df: pd.DataFrame) -> None:
    path = GOLDEN / f"{name}.csv"
    if not path.exists():
        pytest.fail(f"missing golden file {path.name}: {HINT}")
    gold = pd.read_csv(path)
    added = [c for c in df.columns if c not in gold.columns]
    removed = [c for c in gold.columns if c not in df.columns]
    assert not (added or removed), f"{name}: columns added={added} removed={removed}"
    assert len(df) == len(gold), f"{name}: row count {len(df)} != golden {len(gold)}"
    for col in gold.columns:
        new, old = df[col], gold[col]
        if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
            np.testing.assert_allclose(
                new.to_numpy(dtype=float),
                old.to_numpy(dtype=float),
                rtol=RTOL,
                atol=_atol(col),
                equal_nan=True,
                err_msg=f"{name}.{col}",
            )
        else:
            assert new.astype(str).tolist() == old.astype(str).tolist(), f"{name}.{col}"


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to ``a.b.0`` keys."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = obj
    return out


def check_json(name: str, data: dict[str, Any], update: bool) -> None:
    path = GOLDEN / f"{name}.json"
    clean = {k: _jsonable(v) for k, v in data.items()}
    if update:
        refuse_in_ci()
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(clean, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"golden snapshot written: {GOLDEN.name}/{path.name}")
        pytest.skip(f"golden snapshot written: {GOLDEN.name}/{path.name}")
    if not path.exists():
        pytest.fail(f"missing golden file {path.name}: {HINT}")
    gold = json.loads(path.read_text(encoding="utf-8"))
    added, removed = sorted(set(clean) - set(gold)), sorted(set(gold) - set(clean))
    assert not (added or removed), f"{name}: keys added={added} removed={removed}"
    for k, old in gold.items():
        new = clean[k]
        if old is None or new is None:
            assert old is None and new is None, f"{name}.{k}: {new} != {old}"
        elif isinstance(old, (int, float)) and not isinstance(old, bool):
            np.testing.assert_allclose(new, old, rtol=RTOL, atol=_atol(k), err_msg=f"{name}.{k}")
        else:
            assert new == old, f"{name}.{k}"
