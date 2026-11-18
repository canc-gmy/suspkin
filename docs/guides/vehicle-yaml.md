# Vehicle YAML reference

A vehicle file describes the car at **static ride height**. It is validated on load
(`suspkin.load_vehicle`, pydantic, unknown keys are rejected) and errors name the offending
field. The complete worked example is
[`configs/vehicles/fs_reference.yaml`](https://github.com/canc-gmy/suspkin/blob/main/configs/vehicles/fs_reference.yaml).

**Conventions.** ISO 8855: x forward, y left, z up; origin on the ground at the mid front
axle. SI units only: m, rad, kg, N, N/m, Pa. Hardpoints are `[x, y, z]` in metres on the
**left side** (y > 0); the right side is mirrored by the loader. Vectors that are directions
(`rocker_axis`, `bar_axis`, ...) are normalised on load.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema` | int | Must be `1` (default 1). Other values are rejected. |
| `vehicle` | mapping | Car-level parameters. |
| `front` | mapping | Front axle (must have `steering`). |
| `rear` | mapping | Rear axle (must not have `steering`). |

Cross-checks: front `wheel_center` x is 0 and greater than the rear one; each axle's
`wheel_center` z matches `wheel.tyre_radius` within 20 mm.

## `vehicle`

| Field | Unit | Meaning |
|---|---|---|
| `name` | text | Label used in reports. |
| `mass` | kg | Total mass including driver (> 0). |
| `cg_height` | m | Centre-of-gravity height (> 0). |
| `front_mass_ratio` | - | Static front weight fraction, 0 to 1 (exclusive). |
| `drive` | `rwd` / `fwd` / `awd` | Driven axle (default `rwd`). |
| `brake_bias_front` | - | Front share of brake force, 0 to 1 (default 0.6). |
| `drive_split_front` | - | Front share of drive force, 0 to 1. Default: 0 (rwd), 1 (fwd), 0.5 (awd). |
| `unsprung_cg_height` | m | Optional height of the unsprung-mass CG; default the axle's `tyre_radius`. |

## `front` / `rear` (axle)

| Field | Unit | Meaning |
|---|---|---|
| `brakes` | `outboard` / `inboard` | Brake location (default `outboard`); affects anti-geometry. |
| `pushrod_mount` | `upright` / `lca` / `uca` | Member that carries `pushrod_outer` (default `upright`). |
| `unsprung_mass` | kg | Unsprung mass per corner (> 0). |
| `hardpoints` | mapping | See below. |
| `wheel` | mapping | See below. |
| `coilover` | mapping | See below. |
| `arb` | mapping | Optional anti-roll bar, discriminated by `type`. |
| `steering` | mapping | Rack and pinion; required on the front, forbidden on the rear. |
| `right_overrides` | mapping | Optional explicit right-side points (name to `[x, y, z]`, y < 0) for asymmetric cars. Keys are hardpoint names. |

### `hardpoints` (left side)

| Name | Meaning |
|---|---|
| `uca_front`, `uca_rear` | Upper wishbone chassis pivots (define its axis). |
| `uca_outer` | Upper ball joint (must be above `lca_outer`). |
| `lca_front`, `lca_rear` | Lower wishbone chassis pivots. |
| `lca_outer` | Lower ball joint. |
| `tie_inner`, `tie_outer` | Tie rod / toe link ends; `tie_inner` is the rack end on the steered axle. |
| `pushrod_outer` | Pushrod end on the upright or wishbone (see `pushrod_mount`). |
| `pushrod_inner` | Pushrod end on the rocker. |
| `rocker_pivot` | A point on the rocker axis. |
| `rocker_axis` | Rocker axis direction (unit vector, e.g. `[1, 0, 0]`). |
| `rocker_spring` | Spring/damper pin on the rocker. |
| `spring_chassis` | Spring/damper chassis mount. |
| `wheel_center` | Wheel centre; z is the loaded radius. |
| `rocker_arb` | Optional anti-roll-bar drop-link pickup on the rocker (required when `arb` is set). |

Validators: `y > 0` for outer points and wishbone pivots (`y >= 0` for inboard points such as
`tie_inner`, rocker points, `spring_chassis`); every link longer than 10 mm; wishbone axes and
rocker arms non-degenerate (pickups at least 10 mm from the rocker axis).

### `wheel`

| Field | Unit | Meaning |
|---|---|---|
| `tyre_radius` | m | Unloaded tyre radius. |
| `rim_width` | m | Rim width. |
| `static_camber` | rad | Static camber, negative = top inward (between -0.2 and 0.2). |
| `static_toe` | rad | Static toe, positive = toe-in (between -0.1 and 0.1). |
| `tyre_vertical_stiffness` | N/m | Vertical tyre rate. |

### `coilover`

| Field | Unit | Meaning |
|---|---|---|
| `spring_rate` | N/m | Spring rate. |
| `free_length` | m | Spring free length; static force is `spring_rate * (free_length - static length) + preload`. |
| `preload` | N | Extra static preload (>= 0). |
| `length_min`, `length_max` | m | Allowed coilover length range; the static length (`rocker_spring` to `spring_chassis`) must lie inside. |

### `steering`

| Field | Unit | Meaning |
|---|---|---|
| `c_factor` | m/rev | Rack travel per pinion revolution. |
| `rack_travel` | m | `[min, max]` with min < 0 < max. |
| `rack_axis` | - | Rack direction (default `[0, 1, 0]`). Positive travel moves the tie rod ends toward +y (steer left). |

### `arb` (anti-roll bar)

Common to the tube types: `arm_end` (drop-link end of the arm, left side), `bar_pivot` (where
the arm meets the bar), `bar_axis` (default `[0, 1, 0]`), `bar_length` m (must equal
`2 * bar_pivot.y` within 10 mm), `outer_diameter`, `inner_diameter` m (inner < outer),
`shear_modulus` Pa.

| `type` | Extra fields |
|---|---|
| `torsional` | `arm_diameter` m (round solid arms; omit for rigid arms), `arm_youngs_modulus` Pa (default 200e9). |
| `blade` | `blade_width`, `blade_thickness` m, `blade_youngs_modulus` Pa, `blade_angle` rad (0 = flat/soft, pi/2 = edge-on/stiff). |
| `beam` | Transverse bending bar, no tube fields: `beam_center` (on the centreline, y = 0), `beam_axis`, `arm_end`, `beam_width`, `beam_thickness` m, `beam_youngs_modulus` Pa, `beam_angle` rad. |

## Minimal skeleton

```yaml
schema: 1
vehicle: {name: demo, mass: 290.0, cg_height: 0.28, front_mass_ratio: 0.47, drive: rwd}
front:
  unsprung_mass: 10.0
  hardpoints: {uca_front: [0.13, 0.30, 0.325], ...}   # all 14 hardpoints + rocker_axis
  wheel: {tyre_radius: 0.228, rim_width: 0.178, static_camber: -0.0175, static_toe: 0.0,
          tyre_vertical_stiffness: 100000.0}
  coilover: {spring_rate: 35000.0, free_length: 0.2164, preload: 0.0,
             length_min: 0.16, length_max: 0.24}
  steering: {c_factor: 0.09, rack_travel: [-0.03, 0.03]}
rear:
  unsprung_mass: 10.0
  ...
```

Save a modified model with `suspkin.save_vehicle`; errors are raised as
`suspkin.VehicleConfigError`. Full details of the model are in
[Architecture & math, section 4](../ARCHITECTURE.md).
