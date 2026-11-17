# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
