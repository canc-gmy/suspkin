"""Append-only run history (CSV), its reader, and the evaluation cache used for resume."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from suspkin.optimize.design import DesignSpace
from suspkin.optimize.evaluate import Evaluation
from suspkin.optimize.spec import Study

F64 = NDArray[np.float64]
_KEY_DECIMALS = 12


class HistoryError(ValueError):
    """The history file does not match the study (columns differ)."""


@dataclass(frozen=True)
class CacheEntry:
    """Cached result of one design vector."""

    f_vector: F64
    g_vector: F64
    f_total: float
    max_g: float
    feasible: bool


class EvalCache(dict[tuple[float, ...], CacheEntry]):
    """Cache keyed by ``tuple(np.round(x, 12))``."""

    @classmethod
    def from_history(cls, df: pd.DataFrame, space: DesignSpace, study: Study) -> EvalCache:
        """Rebuild the cache from a history table (for resume).

        Raises
        ------
        HistoryError
            If the history columns differ from the study's expected columns.
        """
        out = cls()
        out.update(cache_from_history(df, space, study))
        return out


def cache_key(x: ArrayLike) -> tuple[float, ...]:
    """Cache key: ``x`` rounded to 12 decimals."""
    return tuple(np.round(np.asarray(x, dtype=np.float64), _KEY_DECIMALS).tolist())


def cache_keys(xs: ArrayLike) -> list[tuple[float, ...]]:
    """:func:`cache_key` of every row of ``xs`` ``(m, n)`` in one vectorised pass."""
    rows = np.round(np.atleast_2d(np.asarray(xs, dtype=np.float64)), _KEY_DECIMALS).tolist()
    return [tuple(r) for r in rows]


def objective_names(study: Study) -> list[str]:
    """Objective names in YAML order."""
    return [o.name for o in study.objectives]


def constraint_names(study: Study) -> list[str]:
    """Names of the enabled constraints in YAML order (matches ``Evaluation.g_vector``)."""
    return [c.name for c in study.constraints if c.enabled]


def history_columns(space: DesignSpace, study: Study) -> list[str]:
    """Column order of ``history.csv``."""
    cols = ["eval_id"]
    cols += [f"x_{n}" for n in space.names]
    labels = list(space.points(space.from_vehicle()))
    cols += [f"p_{lab}_{a}" for lab in labels for a in "xyz"]
    cols += [f"f_{n}" for n in objective_names(study)]
    cols.append("f_total")
    cols += [f"g_{n}" for n in constraint_names(study)]
    cols += ["max_g", "feasible", "status", "seconds", "source"]
    return cols


def evaluation_status(ev: Evaluation) -> str:
    """``infeasible`` if any result is infeasible, else ``unsolved`` if any is, else ``ok``."""
    st = [r.status for r in ev.results] + [o.status for o in ev.objectives]
    if "infeasible" in st:
        return "infeasible"
    if "unsolved" in st:
        return "unsolved"
    return "ok"


class HistoryWriter:
    """Row-by-row CSV writer; the header is written on create, each row is flushed."""

    def __init__(
        self, path: str | Path, space: DesignSpace, study: Study, *, resume: bool = False
    ) -> None:
        self.path = Path(path)
        self.space = space
        self.study = study
        self.columns = history_columns(space, study)
        self._n_pts = 3 * len(list(space.points(space.from_vehicle())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        append = resume and self.path.exists() and self.path.stat().st_size > 0
        self._fh: IO[str] = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        if not append:
            self._csv.writerow(self.columns)
            self._fh.flush()

    def append(
        self,
        eval_id: int,
        x: ArrayLike,
        evaluation: Evaluation,
        seconds: float,
        source: str,
        points: ArrayLike | None = None,
        *,
        flush: bool = True,
    ) -> None:
        """Write one row and flush.

        ``points`` optionally gives the already decoded point coordinates of ``x`` (flat,
        ``3 * n_variables`` values in variable order) so that they need not be recomputed.
        """
        xa = np.asarray(x, dtype=np.float64)
        status = evaluation_status(evaluation)
        if status == "infeasible":
            pts = [float("nan")] * self._n_pts
        elif points is not None:
            pts = [float(c) for c in np.ravel(points)]
        else:
            pts = [float(c) for p in self.space.points(xa).values() for c in p]
        row: list[Any] = [eval_id, *map(float, xa), *pts]
        row += [float(v) for v in evaluation.f_vector()]
        row.append(float(evaluation.f_total))
        row += [float(v) for v in evaluation.g_vector()]
        row += [float(evaluation.max_g), bool(evaluation.feasible), status]
        row += [float(seconds), source]
        self._csv.writerow(row)
        if flush:
            self._fh.flush()

    def flush(self) -> None:
        """Flush the file (after ``append(..., flush=False)`` rows)."""
        self._fh.flush()

    def close(self) -> None:
        """Close the file (idempotent)."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> HistoryWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_history(path: str | Path) -> pd.DataFrame:
    """Read ``history.csv`` into a DataFrame."""
    return pd.read_csv(Path(path))


def cache_from_history(
    df: pd.DataFrame, space: DesignSpace, study: Study
) -> dict[tuple[float, ...], CacheEntry]:
    """Rebuild the evaluation cache from a history table (for resume).

    Raises
    ------
    HistoryError
        If the history columns differ from the study's expected columns.
    """
    expected = history_columns(space, study)
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise HistoryError(
            f"history does not match the study: missing columns {missing}, "
            f"unexpected columns {extra}"
        )
    xs = df[[f"x_{n}" for n in space.names]].to_numpy(dtype=np.float64)
    fs = df[[f"f_{n}" for n in objective_names(study)]].to_numpy(dtype=np.float64)
    gs = df[[f"g_{n}" for n in constraint_names(study)]].to_numpy(dtype=np.float64)
    ft = df["f_total"].to_numpy(dtype=np.float64)
    mg = df["max_g"].to_numpy(dtype=np.float64)
    fe = df["feasible"].to_numpy(dtype=bool)
    cache: dict[tuple[float, ...], CacheEntry] = {}
    for i in range(len(df)):
        cache[cache_key(xs[i])] = CacheEntry(
            fs[i].copy(), gs[i].copy(), float(ft[i]), float(mg[i]), bool(fe[i])
        )
    return cache
