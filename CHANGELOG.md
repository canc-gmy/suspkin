# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
