# Refactor design: restructure and prune `suspkin`

Date: 2026-11-19. Branch: `claude/superpowers-refactor-docs-d114ea` (isolated worktree; merged to `main` only via a PR the owner approves).

This document covers **project 1: the code refactor**. Project 2 (rewriting all documentation with long, detailed guides) follows after this lands and gets its own design round, because the docs must describe the final code.

## 1. Intent

The owner finds the repo chaotic and hard to understand. The goal is a smaller, cleaner codebase whose structure is easy to explain, so the project-2 docs can walk through every part precisely. Behaviour must not change.

## 2. What "behaviour" means (decided: option A)

Must stay identical:

- CLI: every command, subcommand, flag, default, exit code, and the content of what it prints and writes (CSV, JSON, YAML, Markdown, HTML report text, `best.yaml`, `history.csv`, `result.json`, bundles).
- YAML formats: vehicle and study files load exactly as today, same validation errors.
- Numeric results: within `rtol=1e-12` of today's values, or the existing golden tolerance where looser (see §11).
- Web app: all 16 `/api/v1` endpoints, request/response JSON, `openapi.json`, and the frontend's behaviour.
- Optional extras still optional: numba (`fast`), torch/optuna (`surrogate`), plotly (`viz`), pymoo (`opt`), fastapi (`app`). Importing `suspkin` must not require them.
- Performance: no meaningful regression (the numba fast path stays; `scripts/bench_kinematics.py` within ~10%).

Free to change: Python import paths, module/function/class names, private helpers, `suspkin/__init__.py` lazy exports, deprecated Python shims, test layout.

## 3. Goals, non-goals, anti-goals

Goals
- Fewer, clearer modules with one obvious home per concept.
- One implementation per piece of math (no copy-pasted vector helpers, Kabsch, Hermite, rotation, worker pools, study-artifact helpers).
- Remove dead code and committed artifacts.
- `cli.py` (1095 lines) split by command group.
- A behaviour lock that proves "unchanged" mechanically.

Non-goals
- New features, new CLI commands, new metrics.
- Changing numerical methods, solver tolerances or optimiser algorithms.
- Frontend redesign. The frontend only changes if a backend move forces it (it should not: the API is frozen).
- Documentation rewrite (project 2). Docstrings are touched only where code moves.

Anti-goals (failure even if tests pass)
- Any golden snapshot regenerated to make a slice pass.
- A "simplification" that silently changes a printed number or file.
- Merging scalar and batch math in a way that makes the code harder to follow than two clear versions.

## 4. Approaches considered

1. Prune in place: delete dead code and duplicates, keep the layout. Low risk, ~15% smaller, structure stays muddy.
2. **Rebuild the layout, then prune (chosen by the owner).** Redraw package boundaries where muddy, merge tiny modules, split giant ones, de-duplicate math, delete dead code. More churn; made safe by the behaviour lock.
3. Rewrite from scratch beside the old code. Rejected: highest risk of silent behaviour change.

Honest size estimate after the survey: most of the gain is structural. Expect roughly 10-20% fewer `src` lines (de-duplication, dead code, cli boilerplate) and a large drop in file count and cross-module coupling, not 30-40% fewer lines.

## 5. Behaviour lock (slice 0, before any refactor)

New `tests/regression/behaviour/`, recorded from the current `main` code and never regenerated during the refactor:

- CLI snapshots on `configs/vehicles/fs_reference.yaml` and the three `configs/studies/*.yaml`: for each command in §2 (`version`, `show`, `animate`, `sweep` incl. `--rack-sweep` and `--metrics`, `summary [--json]`, `vehicle [--json]`, `vsweep` each axis, `report` incl. `--compare`, `check` incl. `--csv --strict`, `optimize` with a small seeded budget, `app-export`, `surrogate sample` tiny run) record stdout, exit code, and every text/data file written. Images (PNG/GIF/MP4) are checked for existence and non-zero size only; their underlying data is locked via the figure spec JSON.
- App: one JSON snapshot per endpoint via `TestClient` on a fixed workspace, plus `openapi.json` equality with the committed `frontend/src/api/openapi.json`.
- Surrogate: evaluator outputs with the oracle predictor on a fixed design set (no network training, which is non-deterministic on GPU); a tiny CPU training run checked for file layout and shapes only.
- Volatile fields (timestamps, git hash, absolute paths, durations, versions) are normalised by a small, explicit scrubber in the test helper.
- Comparison: human-readable text exact; data files and floats at `rtol=1e-12`, golden tolerances where they already exist (§11).
- Numba (builder-check call): the lock is recorded and run with the pure-numpy path (`SUSPKIN_NO_NUMBA`-style switch or by monkeypatching the loader); the numba path is held to the numpy path by the existing `test_fast_kernel.py` at its current tolerance. If CI shows ubuntu/windows differences at `rtol=1e-12`, slice 0 widens only the affected fields, documented in the test.
- Optimiser (builder-check call): `optimize` runs with a fixed seed and `--workers 1`. If a driver turns out not to be reproducible across platforms, lock the file layout, column names and row count exactly and leave values to the existing `study_*` goldens.

The lock plus the existing golden, unit and property tests must pass after every slice.

## 6. Target layout

The builder may adjust names; the boundaries are the decision.

```
suspkin/
  units.py
  geometry/
    vectors.py      # was geom_math/* + all copied _dot/_cross/_unit/_rodrigues helpers + rotation helpers (body_rotation, _rot_batch)
    model.py        # pydantic Vehicle/Axle/Corner (was models.py); uses vectors.py, keeps its own epsilon if it differs
    corners.py, io.py
  kinematics/
    rocker.py, corner.py, arb.py, vehicle.py
    batch.py        # batched corner kinematics moved out of surrogate/evaluate.py
    _numba.py       # was _fast.py + _nb_kernels.py (still optional, lazy)
  metrics/
    corner.py       # alignment + actuation + centres + joints (scalar)
    arrays.py       # batch versions (was batch.py); shares Kabsch/IC/angle helpers with corner.py
    arb.py, table.py
    vehicle.py      # vehicle + vehicle_summary
  sweeps.py         # corner_sweeps + vehicle_sweeps
  optimize/
    spec/           # spec.py split: variables/regions, constraints, objectives, study (surrogate block moves to surrogate/spec.py)
    design.py       # + quantise
    regions.py, evaluate.py, problem.py, drivers.py, history.py
    registry.py     # metric_registry minus 3 dead functions
    outputs.py      # + shared artifacts: study_hash, copy_inputs, meta (used by drivers, active, dataset)
    pool.py         # shared worker-pool setup (problem.py + surrogate/dataset.py)
  surrogate/
    spec.py, data.py (dataset + data_types + sampling + grid + channels + extract),
    model.py (model + normalise + train + predict), hpo.py (+ report),
    evaluate.py (predictors + evaluator), vehicle_layer.py (shares Hermite with evaluate),
    active.py (driver; pure trust/stop/infill/finetune helpers may stay a sibling module), workflow.py, problem.py
  plotting/          # one source of truth for display units/summary rows (today in cli.py, figures.py, spec.py)
  report/
  app/
  cli/               # package: __init__ (Typer app), vehicle.py, study.py, app.py, surrogate.py, _format.py
```

Notes:
- The `optimize -> plotting -> optimize` package-level loop is broken by making `plotting/regions.py` import `optimize.spec` inside its functions and having `plotting/__init__.py` stop importing `regions` eagerly. No new top-level module (builder-check call). This belongs to slice 5.
- `examples/*.py`, `scripts/*.py`, and `mkdocs build --strict` must keep working after every slice: when a module moves, update their imports and the `:::` paths in `docs/reference/*.md` in the same slice (only paths; the prose rewrite is project 2).
- Scalar and batch math that is genuinely the same is shared; where merging would move results beyond `rtol=1e-12`, both stay, side by side in one module, with a comment saying why.
- `kinematics/_numba.py` remains a deliberate twin of `corner.py` (numba cannot call the numpy code). A test asserts the two agree; this already exists in `test_fast_kernel.py` and stays.

## 7. Deletions

- `_deprecation.py` and its test (no production users).
- Dead functions: `metric_registry.{corner,vehicle,summary}_metric_names`; definition-only or test-only helpers (`units.mm_to_m`, `deg_to_rad`, `format_quantity`, `intersect_line_plane`, `rotate_about_line`, `rotate_rocker_points`, `bundle_to_results`) and the tests that only exercise them. Keep any that the builder finds used by docs examples.
- Committed artifacts: untrack `results/benchmark_surrogate/**` (already git-ignored; files stay on disk).
- `scripts/`: kept as they are, only imports updated (builder-check call: merging the two profilers is out of scope).
- Kept on purpose: the hidden CLI alias `surrogate sample --out` (it is CLI surface, so §2 keeps it).

## 8. Constraints

- `main` stays green; Conventional Commits; each slice is one commit (`refactor(<area>): ...`).
- Tooling unchanged: uv (`python -m uv` on this PC), ruff, mypy, pytest + hypothesis; CI on ubuntu + windows.
- Orchestration per `CLAUDE.md`: Opus specs and commits; `coder` (sonnet) implements; `verifier` gates; `reviewer` checks math-heavy slices (geometry, kinematics, metrics, surrogate kernels).
- Git mechanics: each slice is committed locally on this branch; this design doc is committed with slice 0. Pushing the branch and opening a draft PR happens once at the end, after the owner says yes.
- Release: version bump to `1.2.0` is the owner's call at merge time (public Python paths changed, CLI did not). CHANGELOG gets one `Changed` entry.

## 9. Order of work

0. Behaviour lock (§5). Record on unchanged code, commit.
1. Untrack `results/`, delete `_deprecation.py` and dead code (§7).
2. geometry + vectors consolidation.
3. kinematics (incl. `batch.py`, `_numba.py`).
4. metrics + sweeps.
5. optimize (spec split, outputs/artifacts, pool, registry).
6. surrogate.
7. plotting + report (single display-units source, break import loop).
8. cli package split.
9. app touch-ups for moved imports; `openapi.json` must be byte-identical.
10. Final: full gate, benchmark, reviewer pass over the whole diff, line/file count before vs after.

Slices 2-4 are sequential (each depends on the last). 5-6 are sequential. 7-9 may run in parallel once 6 lands if they touch disjoint files.

## 10. Left to the builder

- Exact module and function names inside the layout above.
- Whether a merge is worth it when the result would exceed ~800 lines (prefer splitting then).
- The scrubber details and how CLI snapshots are stored (one dir per command is fine).
- Which test files to merge or rename as modules move; total test count may drop only where tests covered deleted code.

## 11. Numeric strictness (owner decision, 2026-11-19)

"Tiny drift OK". Numbers may differ from the recorded lock by at most `rtol=1e-12` (with a matching small `atol` for values near zero), or the existing golden tolerance where looser. Human-readable output (CLI tables, report text, Markdown) must match exactly; raw CSV/JSON/YAML data files are compared numerically at that tolerance. This overrides the "bit-for-bit" wording in §2 and permits merging duplicated math (Kabsch, Hermite, rotations, vector helpers, scalar/batch helpers) into single implementations.
