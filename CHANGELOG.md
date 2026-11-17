# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
