"""Profile one full-order (FOM) design evaluation and one surrogate dataset sample.

Measures, single process, on ``configs/studies/fs_reference_optimize.yaml``:

1. total time of one ``_eval_with`` call (the optimizer entry point), median of repeats;
2. a stage breakdown (exclusive time per stage via wrapped functions) plus a cProfile
   top list (cumulative);
3. the number of corner states solved per evaluation;
4. the cost of one dataset sample (see docs/concepts/surrogate.md): left front corner on a
   travel x rack grid and left rear corner on a travel grid, solve only and solve + metric tables;
5. the per-state cost of solving and of building the metric tables.

Run: ``python -m uv run python scripts/profile_fom.py [--study PATH] [--repeats N]``.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from suspkin.geometry.corners import build_corners
from suspkin.kinematics import corner as kcorner
from suspkin.kinematics.corner import CornerState, solve_corner, sweep_corner
from suspkin.metrics import joint_angles_frame, metrics_frame
from suspkin.optimize.design import DesignSpace
from suspkin.optimize.problem import _eval_with
from suspkin.optimize.spec import load_study

DEFAULT_STUDY = (
    Path(__file__).resolve().parents[1] / "configs" / "studies" / ("fs_reference_optimize.yaml")
)
N_TRAVEL = 41
N_RACK = 21
TRAVEL_HALF = 0.030
RACK_HALF = 0.025

# (module, attribute) -> stage label; wrapped for exclusive timing during the breakdown.
_STAGES: dict[str, tuple[str, ...]] = {
    "vehicle build": ("DesignSpace.to_vehicle", "build_corners"),
    "corner solve": ("sweep_corner", "solve_corner", "sweep_rack"),
    "vehicle sweep": ("sweep_vehicle",),
    "metric tables": (
        "states_to_frame",
        "metrics_frame",
        "joint_angles_frame",
        "add_gradients",
        "vehicle_frame",
    ),
    "vehicle summary": ("vehicle_summary",),
    "joint angles": ("body_rotations", "joint_angle"),
}


class _Timer:
    """Exclusive (self) time per stage; nested stage calls are subtracted from the parent."""

    def __init__(self) -> None:
        self.excl: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._stack: list[list[Any]] = []

    def wrap(self, fn: Callable[..., Any], stage: str) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            frame: list[Any] = [stage, 0.0]  # label, time spent in children
            self._stack.append(frame)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self._stack.pop()
                self.excl[stage] += dt - frame[1]
                self.calls[stage] += 1
                if self._stack:
                    self._stack[-1][1] += dt

        return inner


class _Counter:
    """Counts calls of the raw per-state solve and the states returned by the sweeps."""

    def __init__(self) -> None:
        self.raw_solves = 0
        self.states_returned = 0

    def wrap_raw(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            self.raw_solves += 1
            return fn(*args, **kwargs)

        return inner

    def wrap_states(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            out = fn(*args, **kwargs)
            self.states_returned += len(out) if isinstance(out, list) else 1
            return out

        return inner


def _suspkin_modules() -> list[Any]:
    return [m for n, m in list(sys.modules.items()) if n.startswith("suspkin") and m is not None]


@contextmanager
def _patched(targets: dict[tuple[Any, str], Callable[..., Any]]) -> Iterator[None]:
    saved = {key: getattr(key[0], key[1]) for key in targets}
    try:
        for (obj, name), fn in targets.items():
            setattr(obj, name, fn)
        yield
    finally:
        for (obj, name), fn in saved.items():
            setattr(obj, name, fn)


def _resolve(name: str) -> tuple[Callable[..., Any], list[tuple[Any, str]]]:
    """Original function and every (namespace, attribute) holding it."""
    if name.startswith("DesignSpace."):
        attr = name.split(".", 1)[1]
        return getattr(DesignSpace, attr), [(DesignSpace, attr)]
    holders = [(m, name) for m in _suspkin_modules() if callable(getattr(m, name, None))]
    orig = getattr(holders[0][0], name)
    return orig, [h for h in holders if getattr(h[0], h[1]) is orig]


def _instrumented(timer: _Timer, counter: _Counter) -> dict[tuple[Any, str], Callable[..., Any]]:
    targets: dict[tuple[Any, str], Callable[..., Any]] = {}
    for stage, names in _STAGES.items():
        for name in names:
            orig, holders = _resolve(name)
            wrapped = timer.wrap(orig, stage)
            if name in ("sweep_corner", "sweep_rack", "solve_corner"):
                wrapped = counter.wrap_states(wrapped)
            for h in holders:
                targets[h] = wrapped
    targets[(kcorner, "_solve_raw")] = counter.wrap_raw(kcorner._solve_raw)
    return targets


def _median(fn: Callable[[], Any], repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _rack_range(space: DesignSpace) -> tuple[float, float]:
    st = space.baseline.front.steering
    if st is None:
        return (-RACK_HALF, RACK_HALF)
    lo, hi = float(st.rack_travel[0]), float(st.rack_travel[1])
    return (max(lo, -RACK_HALF), min(hi, RACK_HALF))


def _front_states(corner: Any, travels: Any, racks: Any) -> list[CornerState]:
    states: list[CornerState] = []
    for r in racks:
        states += sweep_corner(corner, travels, rack=float(r))
    return states


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    ap.add_argument("--repeats", type=int, default=10, help="evaluation repeats (default 10)")
    ap.add_argument("--sample-repeats", type=int, default=5, help="dataset repeats (default 5)")
    ap.add_argument("--top", type=int, default=12, help="cProfile rows to print")
    args = ap.parse_args(argv)

    study, vehicle = load_study(args.study)
    space = DesignSpace(study, vehicle)
    x0 = space.from_vehicle()

    # 1. total evaluation, as the optimizer does it
    t_eval = _median(lambda: _eval_with(space, x0), args.repeats)
    ev, _ = _eval_with(space, x0)
    n_ok = sum(1 for c in ev.results if c.status == "pass")
    print(f"study: {args.study.name}  (constraints pass {n_ok}/{len(ev.results)})")
    print(f"\n[1] one evaluation (median of {args.repeats}): {t_eval:.4f} s")

    # 2./3. stage breakdown and state counts (one instrumented run, after warm-up)
    timer, counter = _Timer(), _Counter()
    with _patched(_instrumented(timer, counter)):
        t0 = time.perf_counter()
        _eval_with(space, x0)
        t_inst = time.perf_counter() - t0
    staged = sum(timer.excl.values())
    rest = max(t_inst - staged, 0.0)
    print(f"\n[2] stage breakdown (exclusive; instrumented run {t_inst:.4f} s)")
    print(f"    {'stage':<20}{'calls':>7}{'seconds':>10}{'%':>7}")
    rows = sorted(timer.excl.items(), key=lambda kv: -kv[1])
    rows.append(("constraints/objectives+other", rest))
    for stage, sec in rows:
        calls = timer.calls.get(stage, 0)
        print(f"    {stage:<20}{calls:>7}{sec:>10.4f}{100 * sec / t_inst:>7.1f}")
    print("\n[3] corner states per evaluation")
    print(f"    raw per-state solves (_solve_raw): {counter.raw_solves}")
    print(f"    states returned by sweep/solve calls: {counter.states_returned}")

    prof = cProfile.Profile()
    prof.enable()
    _eval_with(space, x0)
    prof.disable()
    print(f"\n    cProfile, top {args.top} by cumulative time (one evaluation)")
    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative").print_stats(args.top)
    print(f"    cProfile, top {args.top} by own time (one evaluation)")
    stats.sort_stats("tottime").print_stats(args.top)

    # 4. dataset sample, 5. per-state cost
    corners = dict(build_corners(space.baseline))
    fl, rl = corners["FL"], corners["RL"]
    travels = np.linspace(-TRAVEL_HALF, TRAVEL_HALF, N_TRAVEL)
    rlo, rhi = _rack_range(space)
    racks = np.linspace(rlo, rhi, N_RACK)
    nf, nr = N_TRAVEL * N_RACK, N_TRAVEL

    def solve_front() -> list[CornerState]:
        return _front_states(fl, travels, racks)

    def solve_rear() -> list[CornerState]:
        return sweep_corner(rl, travels)

    def tables(states: list[CornerState], corner: Any) -> None:
        metrics_frame(states, corner)
        joint_angles_frame(states, corner)

    fstates, rstates = solve_front(), solve_rear()
    reps = args.sample_repeats
    res = {
        "front solve": _median(solve_front, reps),
        "front solve+tables": _median(lambda: tables(solve_front(), fl), reps),
        "rear solve": _median(solve_rear, reps),
        "rear solve+tables": _median(lambda: tables(solve_rear(), rl), reps),
    }
    print(
        f"\n[4] dataset sample (median of {reps}); front {N_TRAVEL}x{N_RACK}={nf} states, "
        f"rack [{rlo * 1e3:.0f}, {rhi * 1e3:.0f}] mm; rear {nr} states"
    )
    for k, v in res.items():
        print(f"    {k:<22}{v:>9.4f} s")

    t_fm = _median(lambda: metrics_frame(fstates, fl), reps)
    t_ja = _median(lambda: joint_angles_frame(fstates, fl), reps)
    t_s1 = _median(lambda: [solve_corner(fl, float(t)) for t in travels], reps)
    print("\n[5] per-state cost (front grid)")
    print(f"    solve (continuation)   {res['front solve'] / nf * 1e3:>8.3f} ms/state")
    print(f"    solve (cold, rack 0)   {t_s1 / N_TRAVEL * 1e3:>8.3f} ms/state")
    print(f"    metrics_frame          {t_fm / nf * 1e3:>8.3f} ms/state")
    print(f"    joint_angles_frame     {t_ja / nf * 1e3:>8.3f} ms/state")
    print(f"    rear solve             {res['rear solve'] / nr * 1e3:>8.3f} ms/state")
    total = res["front solve+tables"] + res["rear solve+tables"]
    print(f"    sample total (F+R, solve+tables) {total:.4f} s")
    del rstates


if __name__ == "__main__":
    main()
