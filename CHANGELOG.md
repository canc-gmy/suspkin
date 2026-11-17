# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
