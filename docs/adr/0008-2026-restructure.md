# ADR 0008: The November 2026 restructure

Status: accepted. Date: 2026-11-19 to 2026-11-20.

## Context

By 1.1.0 the repository had grown by feature: a 1095-line `cli.py`, several copies of the
same vector helpers, Kabsch fit, Hermite interpolation, rotation matrices and worker-pool
setup, a surrogate that reached into optimiser modules and back, circular imports hidden by
lazy-import workarounds, three places that defined display units, and some dead code. The
owner found it hard to understand, and a documentation rewrite (which followed) needed a
structure that can be explained part by part. Behaviour had to stay the same.

## Decision

**Behaviour lock first.** Before any move, `tests/regression/behaviour/` recorded the current
behaviour and is never regenerated during a refactor (`--update-lock` is refused in CI):

* the CLI: stdout, exit code and every text or data file written by each command, on the
  reference vehicle and the study configs; images are checked for existence and size only
  (their data is locked through the figure-spec JSON);
* the web app: one JSON snapshot per endpoint through `TestClient`, plus equality of the
  generated `openapi.json` with the committed one (ignoring `info.version`, which was stale);
* the surrogate: evaluator outputs with the oracle predictor on fixed designs, and file layout
  and shapes of a tiny CPU training run (no value checks, GPU training is non-deterministic);
* volatile fields (timestamps, git hash, absolute paths, durations, versions) are scrubbed by
  a small explicit helper, and the lock is recorded with the numpy path (no numba).

**What must stay identical (the rulings).**

* CLI commands, flags, defaults, exit codes and printed text; vehicle and study YAML formats
  and validation errors; the 16 `/api/v1` endpoints and their JSON; optional extras stay
  optional (importing `suspkin` must not require numba, torch, optuna, plotly, pymoo or
  fastapi).
* **Numeric tolerance `rtol = 1e-12`** (with a matching tiny `atol` near zero), or the
  existing golden tolerance where looser. Human-readable text must match exactly. This
  deliberately replaced "bit for bit" so that duplicated math could be merged into one
  implementation.
* **Python import paths are free to change**, together with module, function and class names,
  private helpers, lazy exports and test layout. `suspkin.__all__` and everything the
  examples and docs use stayed importable.

**New layout** (boundaries, not just names): `geometry/{vectors,model,corners,io}`;
`kinematics/{corner,rocker,arb,vehicle,batch,ic_arrays,_numba}`; `metrics/{corner,arrays,arb,
table,vehicle}`; a single `sweeps.py`; `optimize/spec/{variables,criteria,study,surrogate}`
plus shared `pool` and `outputs`; fewer `surrogate` modules; `plotting` with one units module;
`cli/` split by command group.

**Further rulings during the work.**

* **`results/` stays tracked.** The design assumed the benchmark surrogate results were
  git-ignored. They are deliberately whitelisted (trained networks via Git LFS) so users need
  not retrain, so untracking them would remove a shipped asset.
* **Surrogate settings live in `optimize/spec/surrogate.py`**, not a `surrogate/spec.py`: the
  `surrogate:` YAML section is part of the study schema owned by `optimize`. This removed a
  new top-level helper module, a lazy `surrogate/__init__` and an import cycle.
* **Two display-unit tables are kept**, co-located in `plotting/units.py`
  (`SUMMARY_ROWS` and `VEHICLE_CLI_UNITS`): the `suspkin vehicle` CLI output uses different
  unit spellings and extra fields, and its output is locked.
* **`surrogate.report` and `surrogate.normalise` stay free of torch and optuna**, so the
  stdlib/numpy-only scripts and the report path work without the extra. An earlier merge of
  these modules into the torch modules was reverted for that reason.
* **`kinematics/_numba.py` stays a deliberate twin** of the pure-Python solver (numba cannot
  call the numpy code); a test asserts agreement. Scalar and batch math that cannot be merged
  within the tolerance stay side by side with a comment saying why.
* **Scalar metric functions unused by the package** (the per-state alignment and centre
  functions) were moved to `tests/unit/_scalar_oracles.py`: they now serve as independent
  oracles for the vectorised `metrics.arrays`.
* The `optimize` to `plotting` import loop was broken by importing `optimize.spec` inside
  the plotting functions that need it.

## Consequences

* Two extra simplification passes (-61 and -396 source lines) followed the planned tasks. The package went from 85 files and 23,662 lines to 83 files and 23,185 lines; the size
  target was missed (dedup gains were offset by split-file headers), the structural target
  was met. Kernel benchmarks stayed within about 5 % of the baseline.
* Old module paths are gone (`geom_math`, `models`, `metrics.batch`, `surrogate.spec`, ...).
  Any external code importing them must change; a minor version bump (1.2.0) is the
  owner's call at merge time. The CLI, the YAML formats and `suspkin.__all__` did not change.
* The lock is a permanent safety net, but it is also coupled to text output: a deliberate
  output change needs `--update-lock` and a stated reason, like the golden snapshots
  (ADR 0006).
* Documentation was rewritten against the new structure (see the concept pages).
