"""Markdown report of the Optuna search per axle (reads ``optuna.db`` read-only, no torch).

Safe to call while another process still writes the study: the database is opened with
SQLite's read-only mode and only plain SELECTs are issued.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["SCORE_DEFINITION", "TRIAL_RATIO_ATTR", "write_hpo_report"]

#: What the validation score of the search and of the final model means (see ``train.py``).
SCORE_DEFINITION = (
    "Score = mean over channels of (masked RMSE in physical units / channel tolerance) on the "
    "complete validation split; lower is better, 1.0 means every channel is at its tolerance. "
    "The mean can hide a single bad channel: see the median / max channel ratio and the worst "
    "channels of the final model. Search-trial scores (subsampled train rows, trial epoch cap) "
    "are not directly comparable with the final-model score (full train split, full schedule)."
)

#: Trial user attribute holding ``{channel: ratio}`` (optional, written by the search).
TRIAL_RATIO_ATTR = "channel_ratio"
_N_WORST = 5

_MAIN_PARAMS = (
    "n_layers",
    "width",
    "activation",
    "residual",
    "layernorm",
    "dropout",
    "lr",
    "batch_size",
    "pod_k",
)
_TOP_N = 10


def _connect_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True, timeout=30.0)


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _param_value(raw: float, dist_json: str) -> Any:
    """Decode a stored parameter (categorical values are stored as choice indices)."""
    dist = json.loads(dist_json)
    attrs = dist.get("attributes", {})
    if dist.get("name") == "CategoricalDistribution":
        return attrs["choices"][int(raw)]
    if dist.get("name") == "IntDistribution":
        return int(raw)
    return float(raw)


def _read_study(db: Path) -> dict[str, Any]:
    """Trials (state, times, value, params, intermediate steps) and user attrs of the study."""
    con = _connect_ro(db)
    try:
        trials: dict[int, dict[str, Any]] = {}
        for tid, num, state, t0, t1 in con.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials "
            "ORDER BY number"
        ):
            trials[tid] = {
                "number": int(num),
                "state": str(state),
                "start": _parse_dt(t0),
                "end": _parse_dt(t1),
                "value": None,
                "params": {},
                "last_step": None,
                "ratios": None,
            }
        for tid, value in con.execute("SELECT trial_id, value FROM trial_values WHERE objective=0"):
            if tid in trials:
                trials[tid]["value"] = None if value is None else float(value)
        for tid, name, raw, dist in con.execute(
            "SELECT trial_id, param_name, param_value, distribution_json FROM trial_params"
        ):
            if tid in trials:
                trials[tid]["params"][name] = _param_value(raw, dist)
        for tid, last in con.execute(
            "SELECT trial_id, MAX(step) FROM trial_intermediate_values GROUP BY trial_id"
        ):
            if tid in trials:
                trials[tid]["last_step"] = int(last)
        try:
            for tid, key, val in con.execute(
                "SELECT trial_id, key, value_json FROM trial_user_attributes"
            ):
                if tid in trials and key == TRIAL_RATIO_ATTR:
                    ratios = json.loads(val)
                    trials[tid]["ratios"] = ratios if isinstance(ratios, dict) else None
        except sqlite3.Error:
            pass  # older database layout: no per-trial ratios
        attrs: dict[str, Any] = {}
        for key, val in con.execute("SELECT key, value_json FROM study_user_attributes"):
            attrs[str(key)] = json.loads(val)
    finally:
        con.close()
    return {"trials": sorted(trials.values(), key=lambda t: t["number"]), "attrs": attrs}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _duration(t: dict[str, Any]) -> float | None:
    if t["start"] is None or t["end"] is None:
        return None
    return float((t["end"] - t["start"]).total_seconds())


def _epochs(t: dict[str, Any]) -> str:
    return "n/a" if t["last_step"] is None else str(t["last_step"] + 1)


def _worst_channel(ratios: dict[str, Any] | None) -> str:
    """``name (ratio)`` of the largest channel ratio, ``n/a`` without data."""
    if not ratios:
        return "n/a"
    name, val = max(ratios.items(), key=lambda kv: float(kv[1]))
    return f"{name} ({float(val):.3g})"


def _ratio_lines(ratios: dict[str, Any]) -> list[str]:
    """Median, max and worst-channel lines of a ``{channel: ratio}`` mapping."""
    items = sorted(((str(k), float(v)) for k, v in ratios.items()), key=lambda kv: -kv[1])
    vals = [v for _, v in items]
    lines = [
        f"- Channel ratios (RMSE / tolerance): mean {statistics.fmean(vals):.4g}, "
        f"median {statistics.median(vals):.4g}, max {max(vals):.4g}.",
        "",
        f"Worst {min(_N_WORST, len(items))} channels:",
        "",
        "| channel | ratio |",
        "|---|---|",
        *[f"| {k} | {v:.4g} |" for k, v in items[:_N_WORST]],
    ]
    return lines


def _axle_section(axle: str, adir: Path) -> list[str]:
    lines = [f"## {axle.capitalize()} axle", ""]
    db = adir / "optuna.db"
    if not db.exists():
        return [*lines, f"No `optuna.db` found in `{adir}`.", ""]
    try:
        study = _read_study(db)
    except sqlite3.Error as exc:
        return [*lines, f"Could not read `{db}`: {exc}", ""]
    trials: list[dict[str, Any]] = study["trials"]
    budget: dict[str, Any] = study["attrs"].get("trial_budget") or {}
    res = _read_json(adir / "hpo_result.json") or {}
    model_info = _read_json(adir / "model" / "model.json")
    counts = Counter(t["state"] for t in trials)
    running = counts.get("RUNNING", 0)
    has_final = model_info is not None or "final_score" in res
    if running and not has_final:
        lines += ["**Status: search in progress.**", ""]

    # Dataset and budget.
    lines += ["### Dataset and budget", ""]
    n_tr, n_va, n_te = res.get("n_train"), res.get("n_val"), res.get("n_test")
    sub = budget.get("subsample", res.get("subsample"))
    rows = [
        ("Rows train / val / test", f"{n_tr} / {n_va} / {n_te}" if n_tr is not None else "n/a"),
        ("Rows per search trial (train)", "all" if sub is None else f"{sub}"),
        ("Device", res.get("device", "n/a")),
        ("Trials requested", res.get("n_trials_requested", "n/a")),
        ("Timeout", _fmt_s(res.get("timeout_s")) if res.get("timeout_s") else "none / n/a"),
        ("Trial epoch cap", budget.get("max_epochs", "n/a")),
        ("Trial patience", budget.get("patience", "n/a")),
    ]
    lines += ["| item | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows], ""]
    lines += [SCORE_DEFINITION, ""]

    # Trial counts and wall time.
    states = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no trials"
    starts = [t["start"] for t in trials if t["start"] is not None]
    ends = [t["end"] for t in trials if t["end"] is not None]
    span = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    lines += [
        "### Trials",
        "",
        f"- Trial counts by state: {states} (total {len(trials)}).",
        f"- Search wall time (first start to last finish in the database): {_fmt_s(span)}.",
    ]
    if res.get("retrain_only"):
        lines.append("- Final model: **retrain only** (best trial retrained, no new trials run).")
    elif res.get("search_seconds") is not None:
        lines.append(f"- Last `search` call took {_fmt_s(res['search_seconds'])}.")
    lines.append("")

    done = sorted(
        (t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None),
        key=lambda t: t["value"],
    )
    if done:
        b = done[0]
        with_ratios = any(t["ratios"] for t in done[:_TOP_N])
        lines += [
            "### Best trial",
            "",
            f"Trial {b['number']}, score {b['value']:.4f}.",
            "",
            "| param | value |",
            "|---|---|",
            *[f"| {k} | {_fmt_num(v)} |" for k, v in sorted(b["params"].items())],
            "",
            f"### Top {min(_TOP_N, len(done))} completed trials",
            "",
            "| trial | score | epochs | duration | "
            + ("worst channel | " if with_ratios else "")
            + " | ".join(_MAIN_PARAMS)
            + " |",
            "|" + "---|" * (4 + int(with_ratios) + len(_MAIN_PARAMS)),
        ]
        for t in done[:_TOP_N]:
            cells = [_fmt_num(t["params"].get(p, "")) for p in _MAIN_PARAMS]
            worst = f"{_worst_channel(t['ratios'])} | " if with_ratios else ""
            lines.append(
                f"| {t['number']} | {t['value']:.4f} | {_epochs(t)} | "
                f"{_fmt_s(_duration(t))} | {worst}" + " | ".join(cells) + " |"
            )
        lines.append("")
    else:
        lines += ["### Best trial", "", "No completed trial yet.", ""]

    pruned = [t for t in trials if t["state"] == "PRUNED"]
    lines += ["### Pruning", ""]
    if pruned:
        at = ", ".join(f"#{t['number']}@{_epochs(t)}" for t in pruned)
        lines.append(f"{len(pruned)} pruned trial(s) (number@epochs reached): {at}.")
    else:
        lines.append("No pruned trial.")
    lines.append("")

    lines += ["### Final model", ""]
    if model_info is not None or "final_score" in res:
        final_score = res.get("final_score", (model_info or {}).get("val_score"))
        hist = (model_info or {}).get("history") or []
        epochs = res.get("final_epochs", len(hist) if hist else None)
        lines += [
            f"- Validation score: {_fmt_num(final_score)}.",
            f"- Epochs trained: {epochs if epochs is not None else 'n/a'}.",
            f"- Training time: {_fmt_s(res.get('final_seconds'))}.",
            "- Trained on the full train split with the full epoch cap and patience.",
        ]
        ratios = ((model_info or {}).get("meta") or {}).get("val_channel_ratio")
        if isinstance(ratios, dict) and ratios:
            lines += _ratio_lines(ratios)
    else:
        lines.append("Not available yet (search not finished).")
    lines.append("")
    return lines


def write_hpo_report(
    models_dir: str | Path,
    axles: Sequence[str] = ("front", "rear"),
    *,
    out: str | Path | None = None,
) -> Path:
    """Write the search report (one section per axle) and return its path.

    Reads ``models_dir/<axle>/optuna.db`` read-only (safe while the search still runs) and,
    when present, ``hpo_result.json`` and ``model/model.json`` of the axle. The default
    output is ``models_dir/hpo_report.md``; axles without a database are noted, not fatal.
    """
    mdir = Path(models_dir)
    target = Path(out) if out is not None else mdir / "hpo_report.md"
    lines = ["# Surrogate architecture search report", ""]
    for ax in axles:
        lines += _axle_section(ax, mdir / ax)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
