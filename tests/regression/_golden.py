"""Golden-snapshot helpers: write (``--update-golden``) or compare CSV / JSON snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).resolve().parent / "golden"
HINT = "run `python -m uv run pytest tests/regression --update-golden` to create it"
RTOL = 1e-6
ATOL = 1e-9
# Gradient columns are finite differences of the sweep: near-zero values carry round-off
# noise that differs a little between platforms/BLAS builds, so they get a looser atol.
ATOL_GRADIENT = 1e-8


def _atol(col: str) -> float:
    return ATOL_GRADIENT if col.startswith("d_") or "gradient" in col else ATOL


def write_frame(name: str, df: pd.DataFrame) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLDEN / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")


def check_frames(frames: dict[str, pd.DataFrame], update: bool) -> None:
    """Write all snapshots (and skip) when ``update``, else compare each one."""
    if update:
        for name, df in frames.items():
            write_frame(name, df)
        pytest.skip(f"golden snapshots written: {', '.join(frames)}")
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
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in data.items()}
    if update:
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(clean, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        pytest.skip(f"golden snapshot written: {path.name}")
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
