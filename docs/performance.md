# Kinematics performance

The kinematics solver feeds the standard studies, the optimizer, the surrogate training data
and the web app, so its speed matters for all of them. `scripts/bench_kinematics.py` measures it
on `configs/vehicles/fs_reference.yaml`:

```bash
python -m uv run python scripts/bench_kinematics.py            # timings
python -m uv run python scripts/bench_kinematics.py --profile  # + cProfile top list
```

Each case is the median of repeated runs in a single process, after one warm-up run.

## Results (Nov 2026)

Measured on an AMD Ryzen desktop (Windows 11, CPython 3.12.16, numpy 2.5.5). Times are in ms.

| Case | Points | Before | After | Speed-up |
|---|---:|---:|---:|---:|
| Corner solve, 1 pose (FL, 10 mm bump, 5 mm rack) | 1 | 0.38 | 0.28 | 1.4x |
| Corner sweep, FL travel | 31 | 1.8 | 1.3 | 1.4x |
| | 61 | 3.4 | 2.3 | 1.5x |
| | 121 | 6.6 | 4.5 | 1.5x |
| Vehicle sweep, heave | 31 | 35.5 | 14.3 | 2.5x |
| | 61 | 66.7 | 26.4 | 2.5x |
| | 121 | 120.1 | 52.2 | 2.3x |
| Vehicle sweep, roll | 31 | 35.4 | 14.0 | 2.5x |
| | 61 | 67.6 | 26.1 | 2.6x |
| | 121 | 119.1 | 52.3 | 2.3x |
| Vehicle sweep, rack | 31 | 22.2 | 11.3 | 2.0x |
| | 61 | 43.7 | 21.4 | 2.0x |
| | 121 | 84.3 | 41.0 | 2.1x |
| `run_standard_studies` (end to end) | 31 | 214.6 | 111.5 | 1.9x |
| | 61 | 335.9 | 161.8 | 2.1x |
| | 121 | 547.9 | 271.7 | 2.0x |

## What changed

The profile of `run_standard_studies` showed that almost 90 % of the time was spent in the vehicle
sweeps. Each pose solves every corner several times: a secant iteration on wheel travel makes the
contact point reach the ground. Each of these corner solves built a full `CornerState` (about 20
point arrays), but only the converged one was kept.

- **The point table is built once per pose.** `_solve_pose` solves the closing angles and the
  rocker and keeps the result as plain tuples (`_Pose`). The secant loop works on these poses.
  Only the converged pose becomes a `CornerState` (`_finish`). The rocker is still solved on every
  iterate, so failure behaviour is unchanged. `solve_corner` uses the same path.
- **Static points are shared, not copied.** Static points in `CornerState.points` are now the
  corner's read-only arrays. Only the moved points get new arrays.
- **Hot 3-vector helpers are inlined.** This covers `_rodrigues`, `_min_rot` and the Jacobian
  columns of `_eval`, with the same operation order. The rocker returns tuples internally
  (`RockerCtx.solve_tuple`).
- **The secant iteration is warm-started.** Two changes:
  - The first secant step uses the converged slope of the neighbouring pose. It is kept in the
    private `VehicleState._slopes` and used only within 0.5 to 2 times `n_z`.
  - The trivial re-solve of the guess at its own travel is skipped when the guess has already
    converged.

  Together these take a heave or roll pose from about 4 to about 2 corner solves per corner.
- **The ARB drop-link solve uses scalar math.** It no longer calls `np.cross`, `axis_angle_matrix`
  or `np.linalg.norm` on 3-vectors. Drop-link lengths are computed once per solve.

## Accuracy

Everything except the secant warm start reproduces the previous results bit for bit. The warm
start changes the converged pose only within the ground-contact tolerance, because the secant
stops at another iterate below `_G_TOL = 1e-11` m. The comparison covers all `run_standard_studies`
tables and the summary at 31, 61 and 121 points, against the previous code:

- Geometric columns differ by at most 8e-10 m (an instant-centre coordinate far from the car).
  Wheel travels and hardpoints differ by about 1e-11 m.
- The largest absolute difference is 4.4e-7 N in a spring force.
- The worst column uses 28 % of the golden tolerance (`rtol = 1e-6`, `atol = 1e-9`, no
  loosened `atol` for the gradient columns).

The golden snapshot tests pass unchanged, without `--update-golden`.

## Optional numba fast path

Install the `fast` extra (`uv sync --extra fast`, or `pip install suspkin[fast]`) to compile the
solver core with numba:

- `newton` compiles the corner Newton solve (`_eval`, the analytic Jacobian, `_solve3`).
- `ground` compiles the whole ground-contact secant of one corner, including the rocker, the wheel
  centre and the contact height.

These kernels live in `kinematics/_nb_kernels.py`; `kinematics/_fast.py` loads them lazily.
Without numba, or with `SUSPKIN_NO_NUMBA=1`, the pure-Python path runs and the results above
are reproduced bit for bit. Fallbacks:

- A kernel that reports a failure (no convergence, branch jump, unreachable rocker, and so on)
  makes that corner re-run in pure Python, so errors and continuation fallbacks are unchanged.
- A numba failure at import, compile or call time disables the fast path for the process, with
  one `RuntimeWarning`.

Measured on the same machine, in ms:

| Case | Points | Pure Python | numba | Speed-up |
|---|---:|---:|---:|---:|
| Corner solve, 1 pose | 1 | 0.28 | 0.15 | 1.9x |
| Corner sweep, FL travel | 61 | 2.3 | 1.4 | 1.6x |
| Vehicle sweep, heave | 61 | 26.8 | 12.1 | 2.2x |
| Vehicle sweep, roll | 61 | 26.8 | 11.9 | 2.3x |
| Vehicle sweep, rack | 61 | 21.6 | 11.8 | 1.8x |
| `run_standard_studies` | 31 | 110.3 | 65.4 | 1.7x |
| | 61 | 162.9 | 92.1 | 1.8x |
| | 121 | 317.1 | 144.7 | 2.2x |

Against the code before this work, `run_standard_studies` at 61 points goes from 336 ms to 92 ms,
3.6 times faster.

The compile cost is paid once per process:

- The first compile takes about 2 s. After that, numba's on-disk cache (`cache=True`) brings the
  one-off cost to about 0.8 s for the numba import and the cache load.
- That cost pays off in long-lived processes and long runs: the web app, an optimizer run, or
  surrogate dataset generation.
- For a single one-shot `run_standard_studies` it is more than the time saved. Set
  `SUSPKIN_NO_NUMBA=1` there if startup latency matters.

The numba results match the pure-Python path to rtol 1e-9; both are checked in
`tests/unit/test_fast_kernel.py`. Against the old code, the comparison above gives the same
worst-case numbers in both modes.

After the kernels, the remaining time is Python: `_pose` and `_finish`, the ARB solve, building
the `CornerState` arrays, and the metric tables. Going further would mean moving those into
kernels too.
