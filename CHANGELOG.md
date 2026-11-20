# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Internal restructure: fewer, clearer modules and one implementation per piece of math. Python
  import paths changed (e.g. `suspkin.geom_math` → `suspkin.geometry.vectors`, `suspkin.sweeps.*` →
  `suspkin.sweeps`, `suspkin.cli` is now a package). CLI, YAML formats, web-app API and numerical
  results are unchanged (locked by `tests/regression/behaviour`).
- Documentation rewritten: Diátaxis structure with long concept pages (conventions, vehicle model,
  corner and vehicle kinematics, metrics, sweeps and reports, optimisation, surrogate, web app,
  architecture) that explain the theory and the exact code, rewritten guides and tutorial, a
  per-package API reference, ADRs 0002-0008, and Google-style docstrings across `src/`.
  `docs/DESIGN_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/performance.md` and `docs/webapp/PLAN.md`
  are removed; their still-true content moved into the new pages.
- The kinematics solver is about 2 times faster end to end (`run_standard_studies` at 61 points:
  336 ms to 162 ms; vehicle sweeps 2.0 to 2.6 times faster). The ground-contact secant builds the
  corner point table only for the converged pose and is warm-started from the neighbouring pose.
  The ARB drop-link solve and the hot 3-vector helpers use scalar math. Results are unchanged
  apart from differences within the ground-contact tolerance (wheel travel about 1e-11 m). See
  `docs/performance.md` and `scripts/bench_kinematics.py`.

### Added

- Optional numba fast path for the corner solver (extra `fast`, `numba>=0.60`): a compiled Newton
  solve and a compiled per-corner ground-contact secant. `run_standard_studies` at 61 points takes
  92 ms, against 163 ms in pure Python. It falls back to pure Python per corner on any kernel
  failure, and for the whole process when numba is missing or broken. `SUSPKIN_NO_NUMBA=1`
  forces the pure-Python path. The kernels compile once per process; with the on-disk cache this
  costs about 0.8 s.

### Removed

- Unused helpers: `_deprecation.deprecated`, `units.mm_to_m`, `units.deg_to_rad`,
  `units.format_quantity`, `intersect_line_plane`, `rotate_about_line`, `rotate_rocker_points`,
  `bundle_to_results`, and `metric_registry.{corner,vehicle,summary}_metric_names`.

## [1.1.0] - 2026-11-19

### Changed

- `suspkin optimize --surrogate ROOT` runs with `ROOT/models/` alone: without `ROOT/dataset/<axle>`
  the model metadata is checked against the study, fine-tuning is disabled for that axle
  (`skip_reason` `no_dataset`) and the networks stay as trained.
- Active learning fine-tuning is 5 to 6 times faster (about 50 s instead of about 315 s per
  iteration for both axles on the benchmark networks, decision S18): the replay rows and validation
  split are kept on the device between iterations, `active_learning.finetune` defaults are now
  `epochs` 60, `replay` 2048 (drawn once per run) and a new `patience` 30 (counted from the best
  epoch of the run). New optional keys `trigger` (1.0), `skip_after` (2), `skip_for` (2) and
  `stall_override` (3): an axle is fine-tuned only if the residual of the new verified point exceeds
  the trigger, and is skipped for a while after repeated fine-tunes that kept the previous weights.
  `al.csv` gets `skipped_front` and `skipped_rear`; older tables resume unchanged. Use
  `trigger: 0, skip_for: 0` for the previous always-fine-tune behaviour.
- Active learning no longer fine-tunes the networks on the CPU by default (`--device cpu`, or `auto`
  without CUDA): it takes minutes per iteration. New optional key `active_learning.finetune.on_cpu`
  (`skip` default, `run` to force); the reason is logged once and recorded per axle in the new
  `al.csv` columns `skip_reason_front` / `skip_reason_rear`; older tables resume unchanged.

- Surrogate inner loop about 5x faster (14 ms to under 3.3 ms per design): the design decode, vehicle build, cache keys and history rows are vectorised over the batch (`DesignSpace.points_batch`, `cache_keys`); results are bit-identical.
- Surrogate channels: the three swing angles are now one channel each holding the line angle wrapped
  to (-90, 90] degrees (tolerance 0.05 deg) instead of (cos, sin) of a direction, which jumped by
  180 degrees when the instant centre changed side. Channel counts: 29 front, 28 rear. Datasets and
  models of the old encoding are refused (sample and train again). The final model stores per-channel
  RMSE/tolerance ratios and the HPO report shows their median, max and the worst 5 channels.

### Added

- Report `docs/reports/surrogate-training.md`: first full surrogate training (Nov 2026), runs,
  problems and fixes, per-axle results and open points.
- Hardpoint grid: study key `resolution` (length, default `0.5 mm`, `null` = continuous) and a
  per-variable `resolution` override. Chosen designs are snapped to absolute multiples of it
  (`DesignSpace.quantise_x`, `points(quantise=)`, `to_vehicle(quantise=)`,
  `suspkin.optimize.quantise`); a domain without a grid node is a study error.

- Surrogate search report `models/hpo_report.md` (`suspkin.surrogate.report.write_hpo_report`, `scripts/surrogate_hpo_report.py`), `hpo_result.json`, and `hpo.max_epochs` / `hpo.subsample` study keys.
- Retrain the best configuration of an existing surrogate search without new trials (`suspkin.surrogate.hpo.retrain_best`, `scripts/surrogate_retrain_best.py`).
- Reloadable result bundles (`suspkin.report.bundle`: `save_bundle`, `load_bundle`); `build_report`
  writes one per design under `bundles/`.
- Declarative figure spec (`suspkin.plotting.spec`) built from the figure panels, and
  `interactive_figure` split out of `interactive_view`.
- Experimental local web app (FastAPI + React/TypeScript/Vite + Plotly.js, extra `app`): overlay
  of revisions, delta vs reference, 3D pose view, optimization tab; versioned `/api/v1` with
  generated TypeScript types.
- `suspkin app` and `suspkin app-export` (self-contained HTML snapshot), the web app guide and
  ADR 0001, and a CI `frontend` job.
- Surrogate optimisation guide (`docs/guides/surrogate.md`) for the experimental v1.1 NN surrogate.
- Setup scripts build the frontend when Node is available (`-NoFrontend` / `--no-frontend` skips).
- Experimental neural-network surrogate `suspkin.surrogate` (v1.1, see `docs/DESIGN_PLAN.md`
  §13 and `docs/ARCHITECTURE.md` section 15). Not exported in `suspkin.__all__`; its API may
  change in minor releases.
  - Latin-hypercube dataset of the left corner of each axle on a fixed grid, sharded and
    resumable; channel registry and corner extractor.
  - Per-axle MLP with a regression and a validity head, Optuna architecture search, and
    validation against a test split.
  - Analytic vehicle layer: roll and pitch travel from tyre contact, roll centre, anti-features,
    roll stiffness, TLLTD, steering and the vehicle summary, with a measured layer error.
  - `SurrogateProblem` (the v0.8 `Problem` on the networks) and an active-learning optimizer
    with a trust region, FOM verification of every reported design, fine-tuning and run folders.
- Study YAML: optional `surrogate` block (`grid`, `dataset`, `training`, `hpo`,
  `active_learning`); study schema 1 is otherwise unchanged.
- CLI (experimental): `suspkin surrogate sample`, `suspkin surrogate train`,
  `suspkin surrogate validate` (all with `--root`) and `suspkin optimize --surrogate DIR`
  with `--device auto|cpu|cuda`.
- Surrogate validation plots and active-learning figures (`suspkin.plotting.surrogate`).
- Benchmark study `configs/studies/fs_reference_surrogate.yaml` (90 variables) and
  `scripts/benchmark_surrogate.py`. The benchmark is not yet run.
- Optional `surrogate` extra (`torch`, `optuna`): CUDA wheels on Windows, CPU wheels on Linux.
- Vectorised metric kernels `suspkin.metrics.batch` (`corner_metric_arrays`,
  `joint_angle_arrays`): the same columns as `metrics_frame` / `joint_angles_frame`, computed
  over all states at once.
- `scripts/profile_fom.py`: timing breakdown of one full-order evaluation.

### Changed

- Optimisers now evaluate and report producible designs: every optimised hardpoint coordinate
  is a multiple of 0.5 mm by default (classic optimiser, surrogate optimisation, active-learning
  infill, best / Pareto designs and exported YAML). Set `resolution: null` for the old behaviour.
  The surrogate training dataset stays continuous and the dataset / model hashes ignore the
  resolution. The active-learning trust-region radius is at least one grid step, and the loop
  stops with `no_candidates` when no new grid design is left. Finite-difference steps are at
  least one grid step. Golden snapshots are unchanged (the baseline is evaluated as it is).
- Surrogate search trials are capped by `hpo.max_epochs` (default 300) and `hpo.subsample`; the best configuration is still retrained in full. Resuming a study under another trial budget is refused.
- `mr_progression` is shown in 1/mm in the figures (as in the CLI).
- `save_vehicle` writes LF line endings.
- `build_report` writes `bundles/` and raises an error on duplicate design slugs.
- About 3x faster full-order evaluation (0.75 s to 0.24 s on the reference study). Metric
  tables, `states_to_frame` and `vehicle_frame` are built from the vectorised kernels, and
  the corner pose and rocker use the scalar fast path. Results are unchanged (max deviation
  9e-16 against the per-state functions; golden snapshots untouched).
- Evaluator logic is faster: closed-form line fit and window fast paths.
- Convex-hull cache in `optimize/regions.py`.

## [1.0.0] - 2026-11-18

### Added

- One-command setup scripts `scripts/setup.ps1` and `scripts/setup.sh` (Python check, uv bootstrap,
  `uv sync --all-extras`, pre-commit hooks, smoke check; `-Minimal` and `-Test` flags).
- Golden regression snapshots (`tests/regression/golden`) of the reference car, rewritten with
  `pytest --update-golden`; the flag is refused on CI.
- Vehicle YAML `schema: 1` (unsupported schemas are rejected with a clear message).
- Stable public API `suspkin.__all__` (lazy imports), `suspkin.__version__`, the deprecation
  policy and the `suspkin.deprecated` helper (see `docs/guides/stability.md`).
- Examples in `examples/`, run by the `examples` test marker.
- MkDocs Material site with API reference and guides, deployed to GitHub Pages.
- CI: OS matrix (Ubuntu, Windows) x Python 3.11-3.13, coverage floor, docs build job and a
  separate examples job.
- Release workflow: tag checks, tests, `uv build` and a GitHub release with notes from this
  file.
- `.gitattributes` enforcing LF line endings.

### Changed

- `docs/ARCHITECTURE.md` refreshed for 1.0 (testing, tooling, stable surface).
- `suspkin optimize` and `suspkin.optimize` are marked experimental; their API may change in
  minor releases.
- Study schema error message: `unsupported study schema N; this suspkin supports 1`.

## [0.8.0] - 2026-11-18

### Added

- Objectives (`suspkin.optimize.evaluate`): `curve` (RMS error to a target curve), `target`
  (|value - target|), `minimize` / `maximize` (value, slope, max, min, max_abs or mean over a
  range, or a summary metric) and `linearity` (RMS residual of a straight-line fit). Each has
  a normalised `f = raw / scale`; `f_total` is the weighted sum in weighted mode.
- Optimizer block in the study YAML (`optimize.spec`): `mode` (`weighted` or `pareto`),
  `method`, `seed`, `workers` (int or `auto`), `budget` (evaluations and wall time), `start`
  and method `options`; `time` unit (s, min, h).
- Optimizer drivers (`optimize.drivers`): `nsga2` (pymoo, `opt` extra), `de` (scipy
  differential evolution), `slsqp`, `trust-constr` and `de+slsqp` (DE, then SLSQP polish).
- Parallel problem (`optimize.problem`): process pool (spawn), batch evaluation, cache keyed
  on the rounded design vector, evaluation and wall-time budget (`BudgetExhausted`), and a
  forward-difference Jacobian.
- Append-only history CSV (`optimize.history`) with a reader, the evaluation cache and
  resume (`--resume`): the cache skips evaluated points and the population is warm-started
  from history.
- Run outputs (`optimize.outputs`): run folder with `study.yaml`, `vehicle.yaml`,
  `meta.json`, `history.csv`, `best.yaml` or `pareto/` with `pareto.csv`, `result.json`,
  `figures/` and a before/after `report/`.
- Optimization plots (`plotting.optimization`): convergence, Pareto front, parallel
  coordinates and constraint margins.
- `suspkin optimize STUDY.yaml [--out] [--resume] [--workers] [--evaluations] [--time]
  [--method] [--no-report]` command.
- Demo study `configs/studies/fs_reference_optimize.yaml`.

## [0.7.0] - 2026-11-18

### Added

- Optimization study spec (`suspkin.optimize.spec`): study YAML (schema 1) with variables,
  sweeps and constraints; `load_study` with cross-checks against the vehicle and the metric
  registry.
- Unit strings in study YAML (`units.parse_quantity`), converted to SI and checked against the
  metric dimension.
- Regions (`optimize.regions`): trilinear hexahedron map with Jacobian check, Newton inverse,
  signed distances for box, convex hexahedron, sphere and cylinder.
- Design space (`optimize.design`): unit-cube vector to `Vehicle`, with symmetric and
  asymmetric variables.
- Evaluator (`optimize.evaluate`): band, slope, value, reach, summary, inside, outside,
  distance, relation and joint_angle constraints with normalised `g`; each sweep runs once
  per evaluation.
- Joint misalignment angles (`metrics.joints`): Kabsch body rotations, rod-end and ball-joint
  angles over a corner sweep.
- `suspkin check STUDY.yaml [--csv] [--plot] [--strict]` command.
- Region 3D plot (`plotting.regions.plot_study_3d`).
- Reference study `configs/studies/fs_reference_constraints.yaml`.

### Changed

- Faster corner solver: own damped Newton with an analytic Jacobian (scipy `hybr` kept as
  fallback), scalar fast path without numpy calls on 3-vectors, one `_Ctx` per sweep and a
  linear predictor in continuation. Results unchanged within 1e-9.
- Branch-jump guard: a guided solve that moves an angle by more than 0.05 rad from its
  guess is rejected and the continuation step is halved. `solve_corner` with a far guess
  now reaches the target by staged continuation from the guess.

### Fixed

- Timing tests no longer depend on the machine: work-count budgets plus wall-clock checks
  under the `perf` pytest marker.
- CLI `--help` shows option units again: `(mm)` / `(deg)` instead of `[mm]` / `[deg]`,
  which Rich markup swallowed.

## [0.6.0] - 2026-11-17

### Added

- 2D figure sets (`suspkin.plotting.figures`): alignment, wheel position, centres,
  actuation, steering/Ackermann, roll, heave/pitch, summary table; N-design overlays;
  shared colour-blind-safe style.
- 3D view at any solved pose, sweep animation (GIF/MP4) and interactive plotly HTML
  (`viz` extra); `suspkin show` pose options and `suspkin animate`.
- Report generator (`suspkin.report`): standard studies, CSV/JSON export, figures,
  self-contained `report.html` and `report.md`; `suspkin report --compare`.
- README usage section.

### Changed

- **Breaking (CLI):** all length inputs are now in mm (`sweep --travel/--rack/--rack-sweep`,
  `vsweep --heave/--rack/--base-heave/--base-rack/--pivot`); angles in deg. Files stay SI.

## [0.5.0] - 2026-11-17

### Added

- Vehicle-level solver (`solve_vehicle`): body heave, roll, pitch and rack;
  each corner's travel found so the tyre touches the tilted ground; ARBs per axle.
  Default rotation pivot is the vehicle CG.
- Vehicle sweeps incl. combined cases and `vehicle_frame` with per-corner metrics
  (ground contact under pitch), ground-frame camber/steer, axle roll centres.
- Vehicle formulas: anti-dive, anti-lift, anti-squat (brake bias, drive split,
  inboard/outboard), roll stiffness (springs, ARB, tyres), roll gradient, lateral
  load transfer and TLLTD, Ackermann, steering ratio, turning radius, pitch centre.
- `vehicle_summary` incl. roll camber, roll steer, RC migration (relative to the
  car centreline) and RC curvature.
- CLI: `suspkin vehicle` and `suspkin vsweep` (with `--pivot`).
- `VehicleParams`: `brake_bias_front`, `drive_split_front`, `unsprung_cg_height`.

## [0.4.0] - 2026-11-17

### Added

- Actuation metrics: pushrod angles, rocker rotation, coilover compression and
  remaining stroke, spring force.
- Motion ratios from the solver (spring, damper, pushrod, rocker, progression),
  wheel rate by virtual work (incl. spring-force term), ride rate with tyre.
- Anti-roll bars: `torsional` (bar twist in series with arm bending), `blade`
  (torsion tube + oriented blade arms) and `beam` (heave-free bending bar);
  ARB kinematics, stiffness, energy, roll-mode and single-wheel ARB wheel rates.

### Changed

- **Breaking:** ARB type `flexural` removed; use `blade` or `beam`.
- Reference car: front torsional ARB with bending arms, rear blade ARB, 0.15 m arms.

## [0.3.0] - 2026-11-17

### Added

- Alignment metrics (`suspkin.metrics`): camber, toe, steer angle, caster, KPI,
  scrub radius, mechanical trail, kingpin offset, spindle length.
- Wheel position metrics: half-track change, contact-patch shift, wheel-centre
  recession, wheelbase change contribution.
- Instant centres by the plane-section method: front/side-view IC, swing-arm
  lengths and signed swing angles (positive = anti-geometry), symmetric and
  left/right roll centre.
- Metrics tables with gradients (camber gain, bump steer, caster change,
  track change rate, RC migration, steer gain) and static summary.
- `suspkin sweep --metrics` and `suspkin summary` CLI commands.
- Line/plane geometry helpers in `suspkin.geom_math`.

### Changed

- Reference-vehicle tests use the metrics module instead of temporary helpers.

## [0.2.0] - 2026-11-17

### Added

- Corner kinematic solver (`suspkin.kinematics`): reduced 3-unknown formulation
  (UCA/LCA rotation + upright spin about the kingpin axis), closed-form rocker,
  toggle margin and coilover range checks.
- Pushrod mount options `upright`, `lca` and `uca`.
- Continuation sweeps over wheel travel and rack (`sweep_corner`, `sweep_rack`);
  cold solves step from the static pose to stay on the same branch.
- Tidy result tables (`states_to_frame`) and `suspkin sweep` CLI command (CSV export).
- Rotation helpers in `suspkin.geom_math`.
- Regression sweeps on the reference car.

### Changed

- Reference car: pushrod now mounted on the upper wishbone on both axles,
  rocker above the UCA, transverse coilover.
- Test fixture rocker re-placed away from toggle.

## [0.1.0] - 2026-11-17

### Added

- Geometry model (`suspkin.geometry`): pydantic models for hardpoints, wheel, coilover,
  torsional and flexural ARB, steering rack, axle and vehicle, with validation.
- Lossless YAML load/save (`load_vehicle`, `save_vehicle`).
- `build_corners`: left-to-right mirroring with per-point right-side overrides.
- `suspkin.geom_math` vector helpers (`mirror_y`, `unit`, `distance`).
- Static 3D vehicle view (`plot_vehicle_3d`) and `suspkin show` CLI command.
- Reference Formula Student vehicle `configs/vehicles/fs_reference.yaml`.

## [0.0.1] - 2026-11-17

### Added

- Package scaffold (`suspkin` with subpackages for geometry, math, kinematics, metrics, sweeps, plotting, optimize).
- CLI stub with `suspkin version`.
- Tooling: ruff, mypy, pytest, pre-commit.
- CI workflow (GitHub Actions) running lint, format check, type check and tests.
- Design plan in `docs/DESIGN_PLAN.md`.
