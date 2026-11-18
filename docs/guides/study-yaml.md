# Study YAML reference

!!! warning "Experimental"
    The optimizer (`suspkin.optimize`, the `optimize` command) is **experimental**: its Python
    API may change in minor releases. The study file format, schema 1, is stable.

A study file describes *what may move* (variables), *what is measured* (sweeps) and *what is
required or wanted* (constraints and objectives). One vehicle can have many studies. Studies
are loaded with `suspkin.optimize.spec.load_study`, which returns `(Study, Vehicle)` or raises
`StudyConfigError`.

Reference files: `configs/studies/fs_reference_constraints.yaml` (constraints only; every
region, sweep and constraint type) and `configs/studies/fs_reference_optimize.yaml` (adds
objectives and an optimizer block).

## Top level

| Key | Meaning |
|---|---|
| `schema` | Must be `1`. |
| `vehicle` | Path to the vehicle YAML, relative to the study file. |
| `variables` | Hardpoints that may move, each with a domain. |
| `sweeps` | Named corner or vehicle sweeps that constraints and objectives refer to. |
| `constraints` | Requirements. |
| `objectives` | Optional goals. |
| `optimizer` | Optional optimizer settings. |

Values are SI numbers or strings with a unit (`"30 mm"`, `"-1.5 deg"`, `"0.5 deg/deg"`). A
block-level `units: mm` applies to every bare coordinate in that block. Durations take `s`,
`min` or `h`. Coordinates are ISO 8855 vehicle-frame positions on the left side (y > 0).
Point references look like `front.uca_front` or, for a single coordinate,
`front.uca_front.z`.

## `variables`

Each entry: `point: <axle>.<hardpoint>`, `domain` (exactly one of the below), optional
`asymmetric` and `side` (to optimise the right-hand point independently).

| Domain | Keys |
|---|---|
| `hexahedron` | `vertices`: 8 points in FEM order (0-3 bottom face counter-clockwise seen from +z, 4-7 above 0-3); must be convex. |
| `box` | `min`, `max` corner points. |
| `offset` | Per-axis ranges around the baseline point, e.g. `{x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}`. |

## `sweeps`

A mapping of name to definition.

| `kind` | Keys |
|---|---|
| `corner` | `corner` (FL, FR, RL, RR), `travel` `[min, max]`, `steps`. |
| `vehicle` | `input` (`heave`, `roll`, `pitch`, `rack`), `range` or `full`, `steps`, optional `base`. |

## `constraints`

A list. Common keys: `name`, `type`, optional `min`, `max` (where the type has limits),
`scale` (normalisation), `enabled` (default true), `note`.

| `type` | Checks |
|---|---|
| `band` | A sweep metric stays within `min`/`max` over a `where` range. |
| `slope` | The slope of a sweep metric over a `where` range is within limits. |
| `value` | A sweep metric at a point `at` is within limits. |
| `reach` | A corner can travel the given range with a minimum rocker toggle margin (`min_toggle_margin`) and optional coilover `stroke` check. |
| `summary` | A static vehicle-summary metric is within limits. |
| `inside`, `outside` | A hardpoint (optionally over a sweep, `over`) stays inside or outside a `region` (`box`, `hexahedron`, `sphere`, `cylinder`). |
| `distance` | Distance between points `a` and `b` is within limits. |
| `relation` | Linear relation between coordinates: `lhs op rhs` with `op` in `<=`, `>=`, `==` and an `offset`. |
| `joint_angle` | Misalignment angle of a spherical joint over a sweep stays within limits. |

Semantics of each type are given in [Design plan, section 7.4](../DESIGN_PLAN.md).

## `objectives`

| `type` | Goal |
|---|---|
| `curve` | RMS error of a metric to a target curve (`target: {x: [...], y: [...]}`). |
| `target` | `abs(value - target)`. |
| `minimize`, `maximize` | A reduction `of` the metric (`value`, `slope`, `max`, `min`, `max_abs`, `mean`) over `where`/`at`, or a summary metric without `sweep`. |
| `linearity` | RMS residual of a straight-line fit. |

Keys: `name`, `type`, `sweep`, `metric`, `where`, `at`, `of`, `target`, `scale`, `weight`
(default 1, used in weighted mode only), `note`.

## `optimizer`

| Key | Meaning |
|---|---|
| `mode` | `weighted` (single objective, weighted sum) or `pareto` (multi-objective). |
| `method` | `nsga2` (needs the `opt` extra), `de`, `slsqp`, `trust-constr`, `de+slsqp`. |
| `seed` | Random seed. |
| `workers` | Integer or `auto` (CPU count minus one). |
| `budget` | `evaluations` and/or `time` (`"15 min"`). |
| `start` | `baseline` or `best` (warm start from a previous run). |
| `options` | Pass-through method options (`pop_size`, `popsize`, `maxiter`, `ftol`, `xtol`, `gtol`, `fd_step`, ...). |

## Abridged example

```yaml
schema: 1
vehicle: ../vehicles/fs_reference.yaml

variables:
  - point: front.tie_inner
    domain: {box: {min: [70, 135, 139], max: [100, 165, 169], units: mm}}
  - point: rear.pushrod_outer
    domain: {offset: {x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}

sweeps:
  bump_f: {kind: corner, corner: FL, travel: [-30 mm, 30 mm], steps: 13}
  roll: {kind: vehicle, input: roll, range: [-2 deg, 2 deg], steps: 9}

constraints:
  - {name: camber_band_f, type: band, sweep: bump_f, metric: camber,
     where: [-25 mm, 25 mm], min: -4 deg, max: 0 deg, scale: 1 deg}
  - {name: mr_front, type: value, sweep: bump_f, metric: mr_spring, at: 0 mm, min: 0.9, max: 1.1}
  - {name: tie_len, type: distance, a: front.tie_inner, b: front.tie_outer, min: 400 mm}
```

Check a study's baseline with `suspkin check`, optimise with `suspkin optimize`
([Command line](cli.md)). Maths and file roles: [Architecture, section 14](../ARCHITECTURE.md).
