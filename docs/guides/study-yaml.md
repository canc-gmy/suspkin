# Study YAML reference

!!! warning "Experimental"
    The optimizer (`suspkin.optimize`, the `optimize` command) and the surrogate are
    **experimental**: the Python API may change in minor releases. The study file format,
    schema 1, is stable.

A study file says *what may move* (variables), *what is measured* (sweeps) and *what is
required or wanted* (constraints and objectives) for **one vehicle file**. One vehicle can
have many studies. The study is loaded by `suspkin.optimize.spec.load_study(path)`, which
returns `(Study, Vehicle)` or raises `suspkin.optimize.spec.StudyConfigError`. The models
live in the package `suspkin.optimize.spec`:

| Module | Content |
|---|---|
| `suspkin.optimize.spec.variables` | units parsing, regions, variables, sweeps |
| `suspkin.optimize.spec.criteria` | constraints, objectives, budget, optimizer block |
| `suspkin.optimize.spec.study` | the `Study` model, cross-checks, `load_study` |
| `suspkin.optimize.spec.surrogate` | the `surrogate:` block |

For how these inputs become a cost function and a search, read
[Optimization](../concepts/optimization.md); for the `surrogate:` block read
[Surrogate](../concepts/surrogate.md) and the [surrogate guide](surrogate.md). The vehicle
file is described in [Vehicle YAML](vehicle-yaml.md); the CLI commands that consume a study
(`check`, `optimize`, `surrogate ...`) are in [Command line](cli.md).

All blocks are frozen pydantic models with `extra="forbid"`: an unknown key is an error that
names the key (for example `optimizer.seeed: Extra inputs are not permitted`).

## Values, units and references

**Numbers.** A bare number is **SI** (m, rad, N, N/m, s). A string carries a unit:
`"30 mm"`, `"-1.5 deg"`, `"0.5 deg/deg"`, `"1.2 deg/g"`, `"35 N/mm"`, `"50 %"`, `"15 min"`.
The loader (`suspkin.units.parse_quantity`) converts everything to SI and **checks the
dimension** against the field or metric, so `"-25 deg"` in a travel window is an error:

```text
constraint 'camber_band_f', field 'where': unit 'deg' in '-25 deg' has dimension 'angle',
expected 'length' (display unit 'mm')
```

Unit symbols: `m cm mm` (length), `rad deg` (angle), `N kN` (force), `kg` (mass), `s min h`
(time), `g` (acceleration in g), `m/s^2`, `%` (percent; the SI value of `50 %` is `50.0`, as for
the percent metrics). Products, quotients and powers are accepted: `N*m/deg`, `deg/mm`,
`mm/mm`. `deg/deg` and `mm/mm` are both dimensionless.

**Region units.** A region mapping may carry `units: mm` (inside the region body or next to
the shape key). It applies to every *bare* number of that region only; strings keep their own
unit. Only length symbols are valid there.

**Point references.** `<axle>.<hardpoint>`, with axle `front` or `rear` and a hardpoint name from
[Vehicle YAML](vehicle-yaml.md#hardpoints) (including `rocker_arb`), for example
`front.uca_front`. A single coordinate adds `.x`, `.y` or `.z` (`front.uca_front.z`). A typo
gets a suggestion:
`unknown hardpoint 'tie_iner' in 'front.tie_iner'; did you mean 'tie_inner', 'tie_outer', ...`.

**Frame.** Regions and coordinates are ISO 8855 vehicle-frame positions on the **left** side
(y > 0); for an asymmetric *right* variable they are on the right side (y < 0).

## Top level

`suspkin.optimize.spec.study.Study`.

| Key | Type | Required / default | Meaning |
|---|---|---|---|
| `schema` | int | **required** | Must be `1`: `unsupported study schema N; this suspkin supports 1`. |
| `vehicle` | str | required | Path to the vehicle YAML, relative to the study file. |
| `resolution` | length or `null` | `0.5 mm` | Grid of the optimised hardpoint coordinates; `null` = continuous. Must be > 0. See [resolution](#resolution-producible-hardpoints). |
| `variables` | list | `[]` | Hardpoints that may move. |
| `sweeps` | mapping | `{}` | Named sweeps referenced by constraints and objectives. |
| `constraints` | list | `[]` | Hard requirements. |
| `objectives` | list | `[]` | Goals. |
| `optimizer` | mapping | none | Optimizer settings. |
| `surrogate` | mapping | none | Neural-network surrogate settings (experimental). |

Cross-checks run when the study loads (`Study` validator plus `check_against_vehicle`):

| Rule | Message (abridged) |
|---|---|
| Constraint names are unique. | `duplicate constraint name 'x'` |
| Objective names are unique and do not clash with a constraint name. | `duplicate objective name 'x'`, `objective name 'x' clashes with a constraint name` |
| `mode: pareto` needs >= 2 objectives, `mode: weighted` >= 1. | `optimizer: mode pareto needs at least 2 objectives` |
| No point is listed twice as a variable (even as a left and a right asymmetric variable). | `duplicate variable point(s): [...]` |
| Every `sweep:` named by an objective or constraint exists. | `unknown sweep 'x' (defined: [...]); did you mean ...` |
| Constraint/objective windows (`where`, `at`) lie inside the sweep range. | `constraint 'x': where[0] = -0.04 lies outside the sweep range [-0.03, 0.03] (SI)` |
| Every referenced point exists in the vehicle (`rocker_arb` only if the vehicle has it). | `point 'x' does not exist in the vehicle (not defined)` |
| A rack sweep needs steering (always true for a valid vehicle). | `sweep 'x': the vehicle has no steering` |
| Every gridded variable has at least one grid node in its domain. | `variable 'x': its domain contains no node of the 0.5 mm grid` |
| A symmetric variable cannot sit on a point that has a `right_overrides` entry. | `symmetric variable on point with right override: 'x' (declare it 'asymmetric: true' or drop the override)` |

## `variables`

Each entry (`suspkin.optimize.spec.variables.Variable`):

| Key | Type | Default | Rule |
|---|---|---|---|
| `point` | `<axle>.<hardpoint>` | required | The hardpoint to move. |
| `domain` | mapping | required | Exactly one shape key out of `hexahedron`, `box`, `offset`. |
| `asymmetric` | bool | `false` | `false`: the **left** point is optimised and the right mirrors it. `true`: only one side moves; the other is pinned in `right_overrides`. |
| `side` | `left` \| `right` | `left` when `asymmetric` | Which side an asymmetric variable moves. Only allowed with `asymmetric: true`: `'side' is only allowed together with asymmetric: true`. |
| `resolution` | length or `null` | study value | Per-variable grid; an explicit `null` means continuous. |

An asymmetric `side: right` variable must have its region at y < 0: `asymmetric right-side
region must lie at y < 0 (vehicle frame)`. The design-space label of such a variable is
`front.uca_front@right`.

### Domains

| Domain | Body | Meaning |
|---|---|---|
| `hexahedron` | `vertices: [8 points]` (or the bare list of 8 points) | Eight vertices in FEM order. |
| `box` | `min: [x, y, z]`, `max: [x, y, z]` | Axis-aligned box; `min < max` on every axis: `box min must be below max on every axis`. |
| `offset` | any of `x`, `y`, `z`, each `[lo, hi]` | Range around the **baseline** point. Only axes you list move; the others stay fixed. At least one axis; `lo <= hi`; `lo == hi` is an error (omit the axis instead). |

Hexahedron vertex order (FEM convention): vertices 0-1-2-3 are the bottom face, counter-clockwise
seen from +z; vertices 4-5-6-7 are the top face, 4 above 0, 5 above 1, and so on. The loader
rejects a hexahedron with a non-positive Jacobian at any corner or the centre (`hexahedron is
degenerate or twisted, check vertex order`). A hexahedron used in a **constraint** region must
in addition be convex.

`sphere` and `cylinder` shapes exist for constraint regions only; they are not valid
variable domains.

```yaml
variables:
  # skewed hexahedron around the baseline (130, 300, 325) mm
  - point: front.uca_front
    domain:
      hexahedron:
        units: mm
        vertices:
          - [112, 282, 307]
          - [148, 282, 307]
          - [148, 318, 307]
          - [112, 318, 307]
          - [116, 288, 343]
          - [144, 288, 343]
          - [144, 324, 343]
          - [116, 324, 343]
  # axis-aligned box
  - point: front.tie_inner
    domain: {box: {min: [70, 135, 139], max: [100, 165, 169], units: mm}}
  # offset from the baseline, x and z only
  - point: rear.pushrod_outer
    domain: {offset: {x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}
    resolution: 1 mm          # per-variable override
  # one side only
  - point: front.tie_outer
    asymmetric: true
    side: right
    domain: {box: {min: [0.07, -0.60, 0.14], max: [0.10, -0.57, 0.16]}}
```

(The last entry is an illustration, it is not part of the reference studies.)

### `resolution`: producible hardpoints

Every optimised hardpoint coordinate is an integer multiple of `resolution` (default
`0.5 mm`; **absolute** vehicle-frame coordinates, not an offset from the baseline). The
optimisers search continuously; each design is snapped to the nearest grid node before it is
evaluated, and the history, best design, Pareto designs and exported YAML hold the snapped
design.

- Box and offset domains: each free axis snaps to the nearest node inside the bounds (floor or
  ceil at an off-grid bound). A fixed axis of an offset domain stays at the baseline value.
- Hexahedron: if the rounded point leaves the region, the nearest grid node inside it is taken.
- A domain without any grid node is rejected when the study loads.
- The baseline design is evaluated as it is, even if some of its points are off the grid (for
  example `pushrod_outer` at 337.4 mm); candidate designs are on the grid.
- The surrogate training dataset is not snapped. Surrogate predictions and active-learning
  infill points use the snapped design.
- Only length-valued hardpoint coordinates are snapped.

## `sweeps`

A mapping `name: definition`. Constraints and objectives refer to sweeps by name. Each sweep is
solved at most once per design evaluation, however many criteria use it.

### `kind: corner`

`suspkin.optimize.spec.variables.CornerSweep`. One corner moved over wheel travel (bump
positive), rack at 0.

| Key | Type | Rule |
|---|---|---|
| `kind` | `corner` | |
| `corner` | `FL` \| `FR` \| `RL` \| `RR` | |
| `travel` | `[lo, hi]` length | `lo < hi`: `travel must be [lo, hi] with lo < hi`. |
| `steps` | int | >= 3 (gradient metrics need at least 3 points). |

The metrics available are the columns of the corner table (see
[Metric names](#metric-names)), including the gradients `camber_gain`, `bump_steer`, ...

### `kind: vehicle`

`suspkin.optimize.spec.variables.VehicleSweep`. The body pose is swept; the other inputs sit at
`base`.

| Key | Type | Rule |
|---|---|---|
| `kind` | `vehicle` | |
| `input` | `heave` \| `roll` \| `pitch` \| `rack` | The swept input. `heave`, `rack` are lengths, `roll`, `pitch` angles. |
| `range` | `[lo, hi]` or `full` | `lo < hi`. `full` only for `input: rack`: it means the steering `rack_travel` of the vehicle (`range: full is only allowed for input: rack`). |
| `steps` | int | >= 3. |
| `base` | mapping | Default `{}`. Fixed values of the *other* inputs, keys `heave`, `roll`, `pitch`, `rack` (with units). Must not contain the swept input: `base must not set the swept input 'roll'`. |

```yaml
sweeps:
  bump_f: {kind: corner, corner: FL, travel: [-30 mm, 30 mm], steps: 13}
  bump_r: {kind: corner, corner: RL, travel: [-30 mm, 30 mm], steps: 13}
  roll:   {kind: vehicle, input: roll, range: [-2 deg, 2 deg], steps: 9}
  steer:  {kind: vehicle, input: rack, range: full, steps: 9}
  roll_in_heave: {kind: vehicle, input: roll, range: [-2 deg, 2 deg], steps: 9,
                  base: {heave: 5 mm}}
```

## Metric names

Metric names are validated against `suspkin.optimize.registry`; a typo gets suggestions:
`unknown corner metric 'camber_gan'; did you mean 'camber_gain', 'camber', 'caster_change'?`.
Dimensions decide which unit strings are accepted for `min`, `max`, `scale`, `target`.

**Corner-sweep metrics** (`corner_metric_dim`, any corner sweep):

| Group | Names |
|---|---|
| State | `travel` `rack` `theta_uca` `theta_lca` `psi` `rocker_angle` `coilover_length` `toggle_margin` `max_residual` `spin_x` `spin_y` `spin_z` |
| Alignment | `camber` `toe` `steer_angle` `caster` `kpi` `scrub_radius` `mechanical_trail` `kingpin_offset_wc` `spindle_length` `half_track` `half_track_change` `contact_patch_dx` `contact_patch_dy` `wheel_center_recession` `wheelbase_change_contribution` |
| Centres | `fvic_y` `fvic_z` `fvsa` `fv_swing_angle` `svic_x` `svic_z` `svsa` `sv_swing_angle` `sv_wc_angle` `rc_height` |
| Actuation | `pushrod_angle` `pushrod_angle_fv` `rocker_rotation` `coilover_compression` `bump_stroke_remaining` `droop_stroke_remaining` `spring_force` `spring_rate` `tyre_stiffness` |
| Gradients | `camber_gain` `bump_steer` `caster_change` `track_change_rate` `rc_migration_rate` `steer_gain` `mr_spring` `mr_progression` `wheel_load_spring` `wheel_rate_linear` `wheel_rate` `ride_rate` |
| Point columns | `<hardpoint>_<x\|y\|z>` for any hardpoint, `rocker_arb` and `contact_patch` (for example `uca_outer_z`) |

**Vehicle-sweep metrics** (`vehicle_metric_dim`): `heave` `roll` `pitch` `rack` `rc_front_y`
`rc_front_z` `rc_rear_y` `rc_rear_z` `arb_front_elastic` `arb_rear_elastic` `wheelbase`
`steer_c_factor`, plus every corner metric above prefixed by the corner (`FL_camber`,
`RR_toe`, `FL_mr_spring`, ...) and the ground-frame columns `FL_cp_x`, `FL_cp_y`, `FL_cp_z`,
`FL_camber_ground`, `FL_steer_ground` (same for `FR`, `RL`, `RR`).

**Summary metrics** (`summary_metric_dim`; fields of the vehicle summary, no sweep):
`wheelbase` `sprung_mass` `sprung_cg_height` `roll_axis_height_at_cg` `roll_stiffness_total`
`roll_gradient` (deg/g) `ay_ref` `load_transfer_front` `load_transfer_rear` `tlltd_front`
`anti_dive_front` `anti_lift_rear` `anti_squat_rear` `anti_lift_front_accel` (percent)
`pitch_centre_x` `pitch_centre_z` `max_steer_inner` `max_steer_outer` `ackermann_5deg`
`ackermann_10deg` `ackermann_15deg` (percent) `steering_ratio` `turning_radius`, and per
axle `front.<f>` / `rear.<f>` with `<f>` in `track` `wheel_rate` `arb_wheel_rate_roll`
`roll_stiffness_springs` `roll_stiffness_arb` `roll_stiffness_tyres` `roll_stiffness` `rc_height`
`roll_camber` `roll_steer` `rc_migration` `rc_height_curvature`. Definitions and sign
conventions are on the [Metrics page](../concepts/metrics.md).

## `constraints`

A list of mappings, discriminated by `type`. A constraint is **satisfied** when its normalised
value $g \le 0$.

Common keys (`suspkin.optimize.spec.criteria`):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required, non-empty, unique | Label in tables and CSV. |
| `type` | str | required | One of the ten types below. |
| `enabled` | bool | `true` | `false` keeps the entry but skips it (status `OFF`). |
| `scale` | number or unit string | auto | > 0. Normalisation scale of $g$. Same dimension as the limits. `reach` rejects it (use `toggle_scale` / `stroke_scale`). |
| `note` | str | none | Free text. |

Types with limits (`band`, `slope`, `value`, `summary`, `distance`, `joint_angle`) take `min`
and/or `max`; at least one is required (`at least one of min, max is required`) and `min <= max`.

**Normalisation.** For a value $v$ and limits $[a, b]$:
$g = \max\big((v - b)/s,\ (a - v)/s\big)$ using the given limits only. The scale $s$ is the first
available of: your `scale`; $b - a$ when both limits are given and differ; a default per
dimension (length 1 mm, angle 0.1 deg, ratio 0.01, percent 1, force 10 N, stiffness 1 N/mm);
otherwise the absolute limit. If none applies (zero limit on a dimension without default) the
study is rejected: `constraint 'x': add 'scale' (no default for dim ... with zero limit)`.
$g$ is a margin in display-sized units, so it is not comparable across dimensions. See
[Optimization](../concepts/optimization.md) for how it enters the search.

### `band`

The metric stays within `min`/`max` over a window of a sweep.

| Key | Type | Meaning |
|---|---|---|
| `sweep` | str | Sweep name. |
| `metric` | str | Sweep metric. |
| `where` | `[lo, hi]` in the sweep input unit | Window; default the whole sweep. `lo < hi`, inside the sweep range. Values at the window ends are interpolated. |
| `min`, `max` | in the metric's unit | Limits. |

`value` and `g` are those of the worst point.

### `slope`

The least-squares slope $\mathrm d(\text{metric})/\mathrm d(\text{input})$ over the window is
within `min`/`max`. Same keys as `band`; limits are in metric unit per input unit
(`deg/mm`, `deg/deg`, ...).

### `value`

The metric at the sweep input `at` (linear interpolation) is within `min`/`max`. Keys `sweep`,
`metric`, `at` (input unit, inside the sweep range), `min`, `max`.

### `reach`

The corner solves over its own `travel` (13 evenly spaced steps) and the coilover and rocker
stay in range.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `corner` | `FL` \| `FR` \| `RL` \| `RR` | required | |
| `travel` | `[lo, hi]` length | required | `lo < hi`. |
| `min_toggle_margin` | angle | none | Minimum allowed rocker toggle margin. |
| `stroke` | bool | `true` | Also require positive remaining bump and droop coilover stroke. |
| `toggle_scale` | angle | `0.1 deg` | Scale of the toggle term. |
| `stroke_scale` | length | `1 mm` | Scale of the stroke term. |

$g$ is the largest of the active terms (`min_toggle_margin - min(toggle_margin)` over
`toggle_scale`; `-min(bump_stroke_remaining)` and `-min(droop_stroke_remaining)` over
`stroke_scale`). A solver failure marks it `UNSOLVED`. With no active term it only requires
the solve.

### `summary`

A static vehicle-summary metric within `min`/`max`: `metric` is a summary name
(`front.rc_height`, `roll_gradient`, `tlltd_front`, ...). A metric that is `None` for the
car is `UNSOLVED`.

### `inside` and `outside`

A hardpoint stays inside (`inside`) or outside (`outside`, keep-out) a region.

| Key | Type | Meaning |
|---|---|---|
| `point` | point ref | The hardpoint. |
| `region` | mapping | One of `box` (`min`, `max`), `hexahedron` (8 points, convex), `sphere` (`center`, `radius`), `cylinder` (`p0`, `p1`, `radius`: finite, capped). `units: mm` allowed. |
| `over` | corner-sweep name | If given, the point is taken from every state of that sweep (worst counts); otherwise the static point. |

$g$ is the signed distance to the region over the scale (default 1 mm), sign-flipped for
`outside`. Rules for `over`: it must name a **corner** sweep, on the same axle as the point,
and the region must lie on the same side as that corner (centroid y > 0 for a left corner,
y < 0 for a right corner, never exactly 0). Without `over` the static left point is used, or
the right point if the region lies at y < 0. `min`/`max` are not accepted by these types.

### `distance`

`|a - b|` between two points within `min`/`max`. Keys `a`, `b` (point refs), optional `over`
(corner sweep on the points' axle; worst over the sweep), `min`, `max` (lengths).

### `relation`

`lhs op rhs + offset` between static left-side coordinates.

| Key | Type | Meaning |
|---|---|---|
| `lhs` | coordinate ref | `front.uca_front.z`. |
| `op` | `>=` or `<=` | Quote it in YAML (`op: ">="`). |
| `rhs` | coordinate ref or length | Another coordinate or a constant (`0.1`, `"100 mm"`). |
| `offset` | length | Default `0`. |

$g = (\text{rhs} + \text{offset} - \text{lhs})/s$ for `>=` and the negative for `<=`. The check
`value` shown is `lhs`.

### `joint_angle`

The misalignment angle of a spherical joint over a corner sweep stays within `min`/`max`.

| Key | Type | Meaning |
|---|---|---|
| `corner` | `FL`..`RR` | Corner of the joint. |
| `joint` | one of `uca_outer` `lca_outer` `tie_inner` `tie_outer` `pushrod_inner` `pushrod_outer` `spring_chassis` `rocker_spring` | The joint. |
| `sweep` | str | A corner sweep **of the same corner**: `sweep 'x' must be a corner sweep of FL`. |
| `axis` | 3-vector | Optional joint axis override; non-zero. |
| `min`, `max` | angle | Limits. |

### Examples

```yaml
constraints:
  - {name: camber_band_f, type: band, sweep: bump_f, metric: camber, where: [-25 mm, 25 mm],
     min: -4 deg, max: 0 deg, scale: 1 deg}
  - {name: camber_gain_f, type: slope, sweep: bump_f, metric: camber,
     min: -0.05 deg/mm, max: 0.0 deg/mm}
  - {name: roll_camber_f, type: slope, sweep: roll, metric: FL_camber_ground,
     min: -1.1 deg/deg, max: -0.5 deg/deg}
  - {name: mr_front, type: value, sweep: bump_f, metric: mr_spring, at: 0 mm, min: 0.9, max: 1.1}
  - {name: travel_f, type: reach, corner: FL, travel: [-30 mm, 30 mm],
     min_toggle_margin: 20 deg, stroke: true}
  - {name: ubj_angle, type: joint_angle, corner: FL, joint: uca_outer, sweep: bump_f, max: 25 deg}
  - {name: rc_height_f, type: summary, metric: front.rc_height, min: 10 mm, max: 80 mm}
  - {name: roll_gradient, type: summary, metric: roll_gradient, max: 1.2 deg/g}
  - name: uca_front_keepout
    type: outside
    point: front.uca_front
    region: {sphere: {center: [0.13, 0.40, 0.46], radius: 30 mm}}
    over: bump_f
  - name: tie_inner_tube
    type: outside
    point: front.tie_inner
    region: {cylinder: {p0: [0.02, 0.10, 0.20], p1: [0.02, 0.30, 0.20], radius: 25 mm}}
  - name: uca_front_zone
    type: inside
    point: front.uca_front
    region: {box: {min: [0.09, 0.26, 0.28], max: [0.17, 0.34, 0.37]}}
  - {name: tie_len, type: distance, a: front.tie_inner, b: front.tie_outer, min: 400 mm}
  - {name: uca_above_lca, type: relation, lhs: front.uca_front.z, op: ">=",
     rhs: front.lca_front.z, offset: 100 mm}
```

Status values shown by `suspkin check`: `PASS`, `FAIL`, `UNSOLVED` (solver failure or missing
value), `INFEASIBLE` (the design itself is invalid, for example a point outside its domain),
`OFF` (`enabled: false`).

## `objectives`

A list of mappings, discriminated by `type`. Each has a raw SI value and a normalised
$f = \text{raw}/s$ (sign flipped for `maximize`). The weighted mode minimises
$f_\text{total} = \sum_i w_i f_i$; Pareto mode minimises the vector $(f_i)$.

Common keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | required, unique, no clash with a constraint | |
| `type` | str | required | `curve`, `target`, `minimize`, `maximize`, `linearity`. |
| `weight` | number | `1.0` | >= 0. Used in weighted mode only. |
| `scale` | number or unit string | auto | > 0. Same dimension as the raw value. |
| `note` | str | none | |

The scale is your `scale`, else the dimension default (length 1 mm, angle 0.1 deg, ratio 0.01,
percent 1, force 10 N, stiffness 1 N/mm), else `|target|` for a `target` objective. If none
exists the study is rejected: `objective 'x': add 'scale' (no default for dim ...)`.

| `type` | Keys | Raw value |
|---|---|---|
| `curve` | `sweep`, `metric`, `target: {x: [...], y: [...]}`, optional `where` | RMS of `metric - target(x)` over the window (linear interpolation of the target). |
| `target` | `metric`, `target`; with `sweep` + `at`, or a summary `metric` without sweep | `|value - target|`. |
| `minimize`, `maximize` | `metric`, optional `sweep`, `of`, `at`, `where` | A reduction `of` the metric. |
| `linearity` | `sweep`, `metric`, optional `where` | RMS residual of a least-squares line through the metric over the window. |

Rules in detail:

- `curve`: `target.x` (sweep-input unit) and `target.y` (metric unit) have the same length >= 2
  and `x` is strictly increasing. `where` must lie inside the target `x` range; without it the
  window is the overlap of the target range and the sweep (an empty overlap is an error).
- `target`: `sweep` and `at` are given together or not at all (`` `at` is required with a
  sweep``, `` `at` needs a sweep``). Without a sweep the metric is a summary metric.
- `minimize` / `maximize`: `of` is one of `value` (default), `slope`, `max`, `min`, `max_abs`,
  `mean` (trapezoidal). With a sweep: `of: value` needs `at` (and no `where`); every other
  `of` takes `where` (default whole sweep) and no `at`. Without a sweep (summary metric) only
  `of: value` is valid and `at`/`where` are not allowed. For `of: slope` the dimension is the
  metric per input unit.
- `where`: `[lo, hi]` with `lo < hi`, inside the sweep range.

```yaml
objectives:
  - name: camber_curve                  # follow a target camber curve
    type: curve
    sweep: bump_f
    metric: camber
    target: {x: [-30 mm, 0 mm, 30 mm], y: [-1.6 deg, -0.6 deg, 0.3 deg]}
    where: [-25 mm, 25 mm]
    weight: 1.0
    scale: 0.1 deg
  - {name: rc_f, type: target, metric: front.rc_height, target: 40 mm, scale: 5 mm}
  - {name: bump_steer_f, type: minimize, sweep: bump_f, metric: bump_steer, of: max_abs,
     where: [-25 mm, 25 mm], scale: 0.01 deg/mm, weight: 1.0}
  - {name: mr_linearity, type: linearity, sweep: bump_f, metric: mr_spring,
     where: [-25 mm, 25 mm], scale: 0.01, weight: 0.5}
  - {name: max_roll_gain, type: maximize, sweep: roll, metric: FL_camber_ground, of: slope,
     scale: 0.1 deg/deg}                # illustration
```

## `optimizer`

`suspkin.optimize.spec.criteria.Optimizer`. The whole block is optional: without it `optimize` uses
`Optimizer()`, that is the defaults below. The study still needs at least one objective.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `weighted` \| `pareto` | `weighted` | `weighted`: minimise $f_\text{total}$ (one design). `pareto`: multi-objective, needs >= 2 objectives. |
| `method` | `nsga2` \| `de` \| `slsqp` \| `trust-constr` \| `de+slsqp` | `de+slsqp` (`nsga2` if `mode: pareto` and no method given) | `pareto` requires `nsga2`: `optimizer: mode pareto requires method nsga2`. `nsga2` needs the `opt` extra (pymoo). |
| `seed` | int | `0` | Random seed (reproducible runs). |
| `workers` | int >= 1 or `auto` | `auto` | Worker processes; `auto` is the CPU count minus one (at least 1). |
| `budget` | mapping | `{evaluations: 2000}` | At least one of `evaluations` (int > 0) and `time` (duration, > 0, e.g. `"15 min"`). Both: the run stops at whichever comes first. |
| `start` | `baseline` \| `best` | `baseline` | Start point for `slsqp` / `trust-constr`: the baseline or the best design of an existing history. `de+slsqp` always polishes from the best design found by its DE phase. |
| `options` | mapping | `{}` | Method options, see below. Unknown keys are ignored. |

Methods: `de` = SciPy differential evolution (vectorised, no polish); `slsqp` and `trust-constr`
= SciPy local solvers with finite-difference Jacobians; `de+slsqp` = DE for 80 % of the
budget, then SLSQP from the best point; `nsga2` = pymoo NSGA-II (single- or multi-objective).

`options` keys read by the drivers (`suspkin.optimize.drivers`):

| Method | Keys |
|---|---|
| `nsga2` | `pop_size` (default `max(20, 4 * n_variables)`) |
| `de` (and the DE phase of `de+slsqp`) | `popsize` (population = `popsize * n`, at least 8), `maxiter`, `mutation`, `recombination`, `strategy`, `tol`, `atol` |
| `slsqp` | `maxiter` (default 100), `ftol`, `fd_step` |
| `trust-constr` | `maxiter` (default 100), `xtol`, `gtol`, `fd_step` |

Command-line options (`--evaluations`, `--time`, `--method`, `--workers`) override this block
for one run.

```yaml
optimizer:
  mode: weighted
  method: de+slsqp
  seed: 1
  workers: auto
  budget: {evaluations: 300, time: 15 min}
  start: baseline
  options: {}            # e.g. {popsize: 10, maxiter: 200, fd_step: 1e-4}
```

### Resuming

`suspkin optimize ... --resume` continues a run folder from its `history.csv`. The folder's
`meta.json` stores `study_hash`, the SHA-256 of the **constraints and objectives only**
(`suspkin.optimize.outputs.study_hash`). **`--resume` therefore detects a change only in
`constraints` and `objectives`.** Edits to variables, domains, sweeps, resolution, the
`optimizer` block, the `surrogate` block or the vehicle file are **not** detected, and
resuming then mixes designs evaluated under different problems. If you change any of those,
start a new run folder.

What happens on a detected change: a plain `optimize --resume` prints
`warning: constraints/objectives changed since the last run; cached results are dropped and old
points are re-evaluated on demand` and continues; the active-learning run
(`optimize --surrogate`) refuses with `the constraints/objectives changed since the last run;
start a new run`.

## `surrogate` (experimental)

`suspkin.optimize.spec.surrogate.SurrogateSpec`. Used by `suspkin surrogate sample`, `train`,
`validate` and by `optimize --surrogate`. The surrogate learns the **left** corner of each axle
on a fixed grid; only **symmetric** variables are supported (an asymmetric one is rejected:
`variable 'x' is asymmetric; the surrogate supports symmetric variables only`). An axle with no
design variable is skipped by `surrogate sample --axle both`. The inner optimiser of
active learning is the study's `optimizer` block (`method`, `mode`, `seed`).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `grid.travel` | mapping | **required** | `min`, `max` (lengths), `points` (int >= 3). Wheel-travel grid; `min < max` and it must contain 0 (`grid must contain 0 (the static pose)`). |
| `grid.rack` | mapping | none | Same keys. Rack grid, front axle only; omit to fix the rack at 0. |
| `dataset.n_initial` | int >= 1 | `20000` | Latin-hypercube samples per axle. |
| `dataset.seed` | int | `1` | LHS and train/val/test split seed. |
| `dataset.shard_size` | int >= 1 | `500` | Rows per shard file. |
| `dataset.lhs_optimization` | `auto` \| `random-cd` \| `none` | `auto` | Latin-hypercube optimisation: `random-cd` (centred-discrepancy optimised), `none` (plain LHS); `auto` uses `random-cd` up to 5000 samples and plain LHS above. |
| `training.device` | `auto` \| `cpu` \| `cuda` | `auto` | `auto`: CUDA if available, else CPU. |
| `training.max_epochs` | int >= 1 | `2000` | Epoch cap. |
| `training.patience` | int >= 1 | `100` | Early-stopping patience (epochs). |
| `training.num_threads` | int >= 1 or null | torch default | CPU threads of torch. |
| `training.finetune_lr_factor` | float > 0 | `0.1` | Learning-rate factor when fine-tuning a trained model. |
| `hpo.n_trials` | int >= 1 | `100` | Optuna trials per axle. |
| `hpo.timeout` | duration or null | `4 h` | Search time budget, e.g. `4 h`, `90 min`; `null` = none. |
| `hpo.parallel` | `auto` \| true \| false | `auto` | `auto`: axles in parallel on CUDA, sequential on CPU. |
| `hpo.max_epochs` | int >= 1 or null | `300` | Epoch cap of each search trial (`null`: use `training.max_epochs`; the smaller of the two applies). The final retrain still uses `training.max_epochs`. |
| `hpo.subsample` | int >= 1 or null | `null` (all) | Training rows each trial uses (same seeded subset for all trials; validation stays complete; the final retrain uses all rows). |
| `active_learning.max_iterations` | int >= 1 | `50` | Iteration cap. |
| `active_learning.fom_evaluations` | int >= 1 | `2000` | Budget of FOM (real model) evaluations of the whole run, baseline included. |
| `active_learning.time` | duration or null | none | Wall-time budget, e.g. `2 h`. |
| `active_learning.surrogate_evaluations` | int >= 1 | `20000` | Budget of each inner optimisation on the networks. |
| `active_learning.infill` | `top`, `perturb` | `0`, `4` | Extra candidates and random perturbation points per iteration besides the predicted optimum (both >= 0). |
| `active_learning.margin` | float >= 0 | `0.05` | Constraint margin on the normalised $g$ during the network search. |
| `active_learning.trust_region` | `initial`, `min`, `max` | `0.2`, `0.01`, `0.5` | Radius in the unit cube; each in (0, 1] and `min <= initial <= max`. |
| `active_learning.stop` | `eps_f`, `patience` | `0.01`, `5` | Stop on relative agreement `eps_f` (> 0) kept for `patience` (>= 1) iterations. |

`active_learning.finetune` (`FinetuneSpec`), all optional:

| Key | Default | Meaning |
|---|---|---|
| `epochs` (>= 1) | `60` | Cap of one fine-tune. |
| `weight` (> 0) | `5.0` | Sample weight of the new rows. |
| `replay` (>= 0) | `2048` | Earlier train rows replayed (drawn once per run, kept on the device). |
| `patience` (>= 1) | `30` | Stop after that many epochs without a new best score. |
| `trigger` (>= 0) | `1.0` | An axle is fine-tuned only if the residual of the new verified point exceeds it (tolerance units; 0 = always). |
| `skip_after` (>= 1) | `2` | After that many consecutive fine-tunes that kept the previous weights the axle is skipped. |
| `skip_for` (>= 0) | `2` | ... for that many iterations (`0` disables the rule), unless the residual exceeds `stall_override * trigger`. |
| `stall_override` (>= 1) | `3.0` | See above. |
| `on_cpu` | `skip` | `skip`: with the networks on the CPU no fine-tuning (it takes minutes per iteration); `run`: apply the rules above. |

Early stopping of a fine-tune scores the mean of the validation score and the score on all
active-learning rows.

```yaml
surrogate:
  grid:
    travel: {min: -30 mm, max: 30 mm, points: 41}
    rack: {min: -30 mm, max: 30 mm, points: 21}
  dataset: {n_initial: 20000, seed: 1, lhs_optimization: auto}
  training: {device: auto, max_epochs: 600, patience: 600}
  hpo: {n_trials: 60, timeout: 90 min, parallel: auto, max_epochs: 300, subsample: 5000}
  active_learning:
    max_iterations: 50
    fom_evaluations: 2000
    surrogate_evaluations: 20000
    infill: {top: 0, perturb: 4}
    finetune: {epochs: 60, weight: 5.0}
    margin: 0.05
    trust_region: {initial: 0.2, min: 0.01, max: 0.5}
    stop: {eps_f: 0.01, patience: 5}
```

## The reference studies

All three live in `configs/studies/` and point at `../vehicles/fs_reference.yaml`. The baseline
passes every constraint with margin (`suspkin check` shows `12/12` or `20/20 constraints pass`).

| File | Content |
|---|---|
| `fs_reference_constraints.yaml` | Constraints only, 20 of them, **every** region type, sweep type and constraint type. Four variables: `front.uca_front` and `front.uca_rear` (hexahedra), `front.tie_inner` (box), `rear.pushrod_outer` (offset). |
| `fs_reference_optimize.yaml` | The same variables and sweeps, 12 of the constraints, four objectives (`curve`, `target`, `minimize` with `max_abs`, `linearity`) and an `optimizer` block for a short demo run (300 evaluations, 15 min). |
| `fs_reference_surrogate.yaml` | The benchmark of the v1.1 surrogate: **every** hardpoint of both axles as a `+-10 mm` offset variable (`wheel_center` `+-5 mm`), the 12 constraints and four objectives of the optimize study, `optimizer` `de` with 2000 evaluations / 2 h, and a full `surrogate:` block (travel grid 41 points, rack grid 21 points). |

Annotated skeleton of `fs_reference_optimize.yaml` (variables shortened, see the file for the
full vertex lists):

```yaml
schema: 1
vehicle: ../vehicles/fs_reference.yaml     # relative to this file

variables:                                 # 11 free parameters in total
  - point: front.uca_front                 # hexahedron: 3 parameters (u, v, w in the unit cube)
    domain: {hexahedron: {units: mm, vertices: [...8 points...]}}
  - point: front.uca_rear
    domain: {hexahedron: {units: mm, vertices: [...8 points...]}}
  - point: front.tie_inner                 # box: 3 parameters
    domain: {box: {min: [70, 135, 139], max: [100, 165, 169], units: mm}}
  - point: rear.pushrod_outer              # offset: 2 parameters (x and z only)
    domain: {offset: {x: [-10 mm, 10 mm], z: [-8 mm, 8 mm]}}

sweeps:                                    # solved once per evaluation each
  bump_f: {kind: corner, corner: FL, travel: [-30 mm, 30 mm], steps: 13}
  bump_r: {kind: corner, corner: RL, travel: [-30 mm, 30 mm], steps: 13}
  roll:   {kind: vehicle, input: roll, range: [-2 deg, 2 deg], steps: 9}
  steer:  {kind: vehicle, input: rack, range: full, steps: 9}

constraints:                               # 12 entries, g <= 0 required
  - {name: camber_band_f, type: band, ...}
  - ...
objectives:                                # 4 entries, f = raw / scale
  - {name: camber_curve, type: curve, ...}
  - ...
optimizer: {mode: weighted, method: de+slsqp, seed: 1, workers: auto,
            budget: {evaluations: 300, time: 15 min}, start: baseline, options: {}}
```

Check it before running anything:

```console
$ suspkin check configs/studies/fs_reference_optimize.yaml
name            type           value      min     max  unit           g  status  detail
camber_band_f   band          -0.629   -4.000   0.000  deg       -0.629  PASS
...
12/12 constraints pass, feasible: yes
```

See [Command line](cli.md#check) for the complete output and the other commands.
