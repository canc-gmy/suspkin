# Vehicle YAML reference

A vehicle file describes the whole car at **static ride height**: car-level data, and for each
axle the left-side hardpoints, wheel, coilover, optional anti-roll bar (ARB) and, on the
front, the steering. This page lists every field, its type, unit, default and validation
rule. For *why* the model is built this way (rocker, pushrod, mirroring maths) read
[Vehicle model](../concepts/vehicle-model.md); for the frame and sign conventions read
[Conventions](../concepts/conventions.md).

The file is read by `suspkin.geometry.io.load_vehicle` into the frozen pydantic model
`suspkin.geometry.model.Vehicle`. Unknown keys anywhere are rejected (`extra="forbid"`), so a
typo is an error, not a silently ignored field.

## Conventions in one paragraph

ISO 8855: x forward, y left, z up. Origin on the ground plane, at the middle of the front axle
at static ride height. **SI units only** in the file: m, rad, kg, N, N/m, Pa. A point is a list
of three numbers `[x, y, z]` in metres. All hardpoints are for the **left** side of the car
(y > 0); the right side is derived by mirroring $y \to -y$. A direction (`rocker_axis`,
`bar_axis`, `beam_axis`, `rack_axis`) is any non-zero 3-vector; it is normalised to unit length
on load (a zero vector is rejected with `<field> must be non-zero`).

## How errors look

`load_vehicle` raises `suspkin.geometry.io.VehicleConfigError` (a `ValueError`) whose message
starts with the file path, then the pydantic location and text. The CLI prints it on stderr and
exits with code 1. Real examples, from `fs_reference.yaml` with one edit each:

```text
front.hardpoints.uca_frnt
  Extra inputs are not permitted [type=extra_forbidden, ...]          # typo in a key
front
  Value error, an ARB requires hardpoints.rocker_arb to be set [...]   # arb without rocker_arb
front
  Value error, static coilover length 0.2000 m outside [0.16, 0.19]    # length_max set to 0.19
  Value error, rear axle must not have steering [...]
front
  Value error, torsional ARB: |2 * bar_pivot.y - bar_length| must be < 0.01 m (bar runs between
  the mirrored pivots), got 0.0800 [...]
schema
  Value error, unsupported vehicle schema 2; this suspkin supports 1 [...]
```

Numeric range violations (`gt=0` and so on) use the standard pydantic wording, for example
`Input should be greater than 0`. In the tables below, **Rule** gives the constraint and, where
the code raises its own message, that message.

Write a vehicle back with `suspkin.geometry.io.save_vehicle(vehicle, path)` (tuples written as
lists, model key order, LF line endings) or get the text with `suspkin.geometry.io.vehicle_yaml`.

## File layout

```yaml
schema: 1          # optional, default 1
vehicle: {...}     # car-level data
front: {...}       # axle block, steering required
rear: {...}        # axle block, steering forbidden
```

### Top level (`suspkin.geometry.model.Vehicle`)

| Key | Type | Required | Default | Rule |
|---|---|---|---|---|
| `schema` | int | no | `1` | Must be `1`: `unsupported vehicle schema N; this suspkin supports 1`. |
| `vehicle` | mapping | yes | - | See [`vehicle`](#vehicle). |
| `front` | mapping | yes | - | See [axle](#front-and-rear-axle). Must have `steering`: `front axle requires steering`. |
| `rear` | mapping | yes | - | Must not have `steering`: `rear axle must not have steering`. |

Cross-axle rules (checked after both axles are valid):

- `front.hardpoints.wheel_center` x must be greater than the rear one:
  `front wheel_center x must be greater than rear wheel_center x`.
- Front `wheel_center` x must be 0 (|x| < 5 mm): `front wheel_center x must be 0 (origin at mid
  front axle)`.
- For each axle, `wheel_center` z must match `wheel.tyre_radius` within 20 mm:
  `front wheel_center z (...) must match tyre_radius (...) within 20 mm`. The hardpoint z is the
  *loaded* radius, so it may sit a little below the unloaded `tyre_radius`.

Derived read-only values: `Vehicle.wheelbase` (front minus rear `wheel_center` x),
`Vehicle.front_track` and `Vehicle.rear_track` (2 x `wheel_center` y).

## `vehicle`

`suspkin.geometry.model.VehicleParams`.

| Field | Type | Unit | Required / default | Rule |
|---|---|---|---|---|
| `name` | str | - | required | Label used in reports and the web app. |
| `mass` | float | kg | required | > 0. Total mass including driver. |
| `cg_height` | float | m | required | > 0. CG height above the ground. |
| `front_mass_ratio` | float | - | required | 0 < x < 1. Static front weight fraction. |
| `drive` | `rwd` \| `fwd` \| `awd` | - | `rwd` | Driven axle(s). |
| `brake_bias_front` | float | - | `0.6` | 0 to 1 inclusive. Front share of braking force. |
| `drive_split_front` | float | - | none | 0 to 1. Front share of drive force. If omitted: 0 for `rwd`, 1 for `fwd`, 0.5 for `awd` (`VehicleParams.effective_drive_split_front`). |
| `unsprung_cg_height` | float | m | none | > 0. Height of the unsprung-mass CG used by the anti-geometry maths; if omitted the axle's `wheel.tyre_radius` is used. |

## Front and rear axle

`suspkin.geometry.model.Axle`. Same schema on both axles.

| Field | Type | Unit | Required / default | Rule |
|---|---|---|---|---|
| `brakes` | `outboard` \| `inboard` | - | `outboard` | Brake torque reaction path; changes anti-dive / anti-lift / anti-squat geometry. |
| `pushrod_mount` | `upright` \| `lca` \| `uca` | - | `upright` | Which member carries `pushrod_outer`: the upright, the lower wishbone or the upper wishbone. |
| `unsprung_mass` | float | kg | required | > 0. Per corner. |
| `hardpoints` | mapping | m | required | See [hardpoints](#hardpoints). |
| `wheel` | mapping | - | required | See [wheel](#wheel). |
| `coilover` | mapping | - | required | See [coilover](#coilover). |
| `arb` | mapping | - | none | See [arb](#arb-anti-roll-bar). Needs `hardpoints.rocker_arb`: `an ARB requires hardpoints.rocker_arb to be set`. |
| `steering` | mapping | - | none | See [steering](#steering). Required on `front`, forbidden on `rear`. |
| `right_overrides` | mapping name to `[x, y, z]` | m | `{}` | Explicit right-side points for an asymmetric car, see [mirroring](#mirroring-and-right_overrides). |

Axle-level cross-checks:

- Static coilover length, the distance `rocker_spring` to `spring_chassis`, must lie in
  `[coilover.length_min, coilover.length_max]`:
  `static coilover length 0.2000 m outside [0.16, 0.19]`.
- Torsional and blade ARB: `|2 * arb.bar_pivot.y - arb.bar_length|` must be < 0.01 m.

### Hardpoints

`suspkin.geometry.model.CornerHardpoints`. 14 mandatory points, one optional point and the
rocker axis. All `[x, y, z]` in metres, left side. The names below are the same names used by
study files (`front.uca_front`) and in result tables (`uca_front_x`, ...).

| Name | Meaning | y rule |
|---|---|---|
| `uca_front`, `uca_rear` | Upper wishbone chassis pivots (they define its rotation axis). | y > 0 |
| `uca_outer` | Upper ball joint on the upright. | y > 0 |
| `lca_front`, `lca_rear` | Lower wishbone chassis pivots. | y > 0 |
| `lca_outer` | Lower ball joint. | y > 0 |
| `tie_inner` | Tie-rod end on the rack (steered axle) or the chassis (toe link). | y >= 0 |
| `tie_outer` | Tie-rod end on the upright (steering arm). | y > 0 |
| `pushrod_outer` | Pushrod end on the upright or wishbone, see `pushrod_mount`. | y > 0 |
| `pushrod_inner` | Pushrod end on the rocker. | y >= 0 |
| `rocker_pivot` | Any point on the rocker rotation axis. | y >= 0 |
| `rocker_spring` | Coilover pin on the rocker. | y >= 0 |
| `spring_chassis` | Coilover chassis mount. | y >= 0 |
| `wheel_center` | Wheel centre; z is the loaded radius. | y > 0 |
| `rocker_arb` | Optional. ARB drop-link pickup on the rocker. Required when `arb` is set. | y >= 0 |
| `rocker_axis` | Direction of the rocker axis (any non-zero vector, normalised). Required. | - |

Violations read `<name> must have y > 0 (left side), got <y>` (or `y >= 0`).

Geometry validators (all lengths in the same message style):

| Rule | Message |
|---|---|
| Each of the six links `uca_front-uca_outer`, `uca_rear-uca_outer`, `lca_front-lca_outer`, `lca_rear-lca_outer`, tie rod, pushrod is longer than 10 mm. | `link <name> shorter than 0.01 m` |
| `uca_outer` z above `lca_outer` z. | `uca_outer z must be above lca_outer z` |
| Each wishbone axis is longer than 10 mm. | `uca_front and uca_rear closer than 0.01 m (degenerate axis)` (same for lca) |
| `pushrod_inner`, `rocker_spring` and `rocker_arb` (if set) are at least 10 mm from the rocker axis line. | `<name> lies within 0.01 m of the rocker axis (degenerate rocker arm)` |

### Wheel

`suspkin.geometry.model.Wheel`.

| Field | Type | Unit | Rule |
|---|---|---|---|
| `tyre_radius` | float | m | > 0. Unloaded radius. |
| `rim_width` | float | m | > 0. |
| `static_camber` | float | rad | -0.2 < x < 0.2. Negative = top of the wheel inward. |
| `static_toe` | float | rad | -0.1 < x < 0.1. Positive = toe-in. |
| `tyre_vertical_stiffness` | float | N/m | > 0. Vertical tyre rate. |

All five are required. Remember the file is in radians: -1 deg is `-0.017453`.

### Coilover

`suspkin.geometry.model.Coilover`. The spring force at a coilover length $L$ is
$F = F_\text{preload} + k\,(L_\text{free} - L)$ (`metrics.arrays` and `kinematics.batch` both use
exactly this expression). Preload is therefore an *extra constant force*, not an offset of the
free length.

| Field | Type | Unit | Rule |
|---|---|---|---|
| `spring_rate` | float | N/m | > 0. $k$. |
| `free_length` | float | m | > 0. $L_\text{free}$. |
| `preload` | float | N | >= 0. Extra static force. The field is required: write `0.0` for none. |
| `length_min` | float | m | > 0. |
| `length_max` | float | m | Must exceed `length_min`: `length_max must be greater than length_min`. |

With `free_length: 0.2164` and a static length of 0.200 m the reference car has
$35000 \times 0.0164 = 574$ N of static spring force, which is exactly what
`suspkin summary` prints as `spring_force` for FL.

### Steering

`suspkin.geometry.model.Steering`. Rack-and-pinion.

| Field | Type | Unit | Rule |
|---|---|---|---|
| `c_factor` | float | m/rev | > 0. Rack travel per pinion revolution (used for the steering ratio). |
| `rack_travel` | `[min, max]` | m | Required, `min < 0 < max`: `rack_travel must satisfy min < 0 < max`. |
| `rack_axis` | 3-vector | - | Default `[0, 1, 0]`, normalised. |

Positive rack travel moves both `tie_inner` points along `+rack_axis`, that is toward +y (to
the left). With a front-steer rack (tie rods ahead of the axle) this steers left, which is
positive steer. The right corner uses the *same* global `rack_axis` (it is **not** mirrored)
because the rack is one rigid part. `range: full` in a study vehicle sweep means
`rack_travel`.

### ARB (anti-roll bar)

Three bar types, selected by the mandatory `type` key (pydantic discriminated union). The bar
is a left-side description; the right half is its mirror image. Common rule for all types: the
axle must define `hardpoints.rocker_arb`, the drop-link pickup on the rocker. The drop link
runs from `rocker_arb` to `arm_end`.

#### `type: torsional` (`TorsionalARB`)

A torsion tube between the two mirrored `bar_pivot` points with round solid arms.

| Field | Type | Unit | Default | Rule |
|---|---|---|---|---|
| `arm_end` | `[x, y, z]` | m | required | Drop-link end of the arm (left side). |
| `bar_pivot` | `[x, y, z]` | m | required | Point where the arm is clamped to the tube (on the bar axis). |
| `bar_axis` | 3-vector | - | `[0, 1, 0]` | Tube axis direction, normalised. |
| `bar_length` | float | m | required | > 0. Must satisfy `abs(2*bar_pivot.y - bar_length) < 0.01`. |
| `outer_diameter` | float | m | required | > 0. |
| `inner_diameter` | float | m | required | >= 0 and `< outer_diameter`: `inner_diameter must be smaller than outer_diameter`. `0` is a solid bar. |
| `shear_modulus` | float | Pa | required | > 0. Steel is about 7.9e10. |
| `arm_diameter` | float or null | m | `null` | > 0. Round solid arm diameter. `null` means **rigid arms**; otherwise the arms bend as cantilevers. |
| `arm_youngs_modulus` | float | Pa | `200e9` | > 0. |

Tube rule: the arm (distance of `arm_end` from the bar axis line through `bar_pivot`) must be
longer than 10 mm: `ARB arm shorter than 0.01 m (arm_end on the bar axis)`.

#### `type: blade` (`BladeARB`)

Same tube fields as torsional (`arm_end`, `bar_pivot`, `bar_axis`, `bar_length`,
`outer_diameter`, `inner_diameter`, `shear_modulus`, with the same rules) but with flat blade
arms from `bar_pivot` to `arm_end`.

| Field | Type | Unit | Default | Rule |
|---|---|---|---|---|
| `blade_width` | float | m | required | > 0. |
| `blade_thickness` | float | m | required | > 0. |
| `blade_youngs_modulus` | float | Pa | required | > 0. |
| `blade_angle` | float | rad | `0.0` | `0` = blade flat, bends about its thin direction (soft); $\pi/2$ = edge-on (stiff). |

#### `type: beam` (`BeamARB`)

A transverse bending bar with a central vertical slider (free in heave, clamped in rotation);
roll bends each half as a cantilever. There are no tube fields.

| Field | Type | Unit | Default | Rule |
|---|---|---|---|---|
| `beam_center` | `[x, y, z]` | m | required | Must be on the centreline: `abs(y) < 1 mm`, else `beam_center must lie on the centreline (|y| < 1 mm)`. |
| `beam_axis` | 3-vector | - | `[0, 1, 0]` | Normalised. |
| `arm_end` | `[x, y, z]` | m | required | Drop-link end (left). The projection of `arm_end - beam_center` on `beam_axis` is the cantilever half-length `a`; it must exceed 10 mm (`beam half-length (arm_end along beam_axis) must exceed 0.01 m`). |
| `beam_width` | float | m | required | > 0. |
| `beam_thickness` | float | m | required | > 0. |
| `beam_youngs_modulus` | float | Pa | required | > 0. |
| `beam_angle` | float | rad | `0.0` | As `blade_angle`. |

## Mirroring and `right_overrides`

`suspkin.geometry.corners.build_corners` builds the four corners `FL, FR, RL, RR`:

1. Left corners use the hardpoints as written.
2. Right corners mirror every hardpoint ($y \to -y$) and the `rocker_axis` direction.
3. Then each entry of `right_overrides` **replaces** the mirrored value of that point. Override
   values are in right-side coordinates, so `y <= 0`.

`right_overrides` rules (`Axle` validator):

| Rule | Message |
|---|---|
| Key must be one of the 14 hardpoint names or `rocker_arb` (not `rocker_axis`). | `right_overrides key 'x' is not a hardpoint name` |
| `y <= 0` for every value. | `right_overrides['uca_outer'] must have y <= 0 (right side)` |
| `y < 0` strictly for `uca_outer`, `lca_outer`, `tie_outer`, `pushrod_outer`, `wheel_center`. | `... must have y < 0 (right side)` |
| `rocker_arb` key needs `hardpoints.rocker_arb`. | `right_overrides has rocker_arb but hardpoints.rocker_arb is unset` |

!!! note "ARB points always mirror"
    `right_overrides` only touches hardpoints. The ARB's own points (`arm_end`, `bar_pivot`,
    `beam_center`; function `suspkin.geometry.corners.arb_points`) are always the mirror of the
    left ones; there is no way to make the bar itself asymmetric. The drop-link pickup
    `rocker_arb` *is* a hardpoint and can be overridden.

Example, a wider right-hand wheel centre and a lowered right upper chassis pivot:

```yaml
front:
  right_overrides:
    wheel_center: [0.0, -0.615, 0.228]    # left is [0, 0.610, 0.228]
    uca_front:    [0.130, -0.300, 0.320]
```

Only one value is stored per overridden point; the left side is unaffected. A study with a
*symmetric* variable on an overridden point is refused (`declare it 'asymmetric: true' or drop
the override`), see [Study YAML](study-yaml.md).

## Springs and preload in practice

- Choose `free_length` so the static spring force is what you want:
  $F_\text{static} = F_\text{preload} + k\,(L_\text{free} - L_\text{static})$, with
  $L_\text{static} = \lVert \texttt{rocker\_spring} - \texttt{spring\_chassis} \rVert$.
- Use `preload` for a pre-compression that is *not* part of the spring free length (a
  preload ring on a real spring). It raises the force at every length by the same amount and
  does not change the rate.
- `length_min` / `length_max` are the hard stroke limits of the unit. They are used by the
  `bump_stroke_remaining`, `droop_stroke_remaining` metrics and by `reach` constraints; the
  static length must lie inside them.
- The rate seen at the wheel includes the motion ratio and the ARB:
  see `wheel_rate`, `ride_rate` in the [metrics page](../concepts/metrics.md).

## Minimal annotated example

`tests/fixtures/minimal_vehicle.yaml` is the smallest valid car used by the test suite (front:
torsional ARB with rigid arms, rear: beam ARB). Every field is annotated here.

```yaml
schema: 1                       # optional; must be 1
vehicle:
  name: minimal_fs              # report label
  mass: 280.0                   # kg, with driver
  cg_height: 0.28               # m
  front_mass_ratio: 0.47        # 0 < x < 1
  drive: rwd                    # rwd | fwd | awd (drive_split_front defaults to 0 for rwd)
front:
  brakes: outboard              # default
  pushrod_mount: upright        # pushrod_outer sits on the upright
  unsprung_mass: 11.0           # kg per corner
  hardpoints:                   # LEFT side, metres, [x, y, z]
    uca_front: [0.10, 0.28, 0.30]      # upper wishbone chassis pivots
    uca_rear: [-0.10, 0.28, 0.29]
    uca_outer: [0.0, 0.55, 0.30]       # upper ball joint, z above lca_outer
    lca_front: [0.15, 0.22, 0.12]      # lower wishbone chassis pivots
    lca_rear: [-0.15, 0.22, 0.12]
    lca_outer: [0.0, 0.58, 0.13]       # lower ball joint
    tie_inner: [0.08, 0.15, 0.14]      # rack end
    tie_outer: [0.08, 0.56, 0.15]      # steering arm
    pushrod_outer: [0.0, 0.56, 0.27]   # on the upright (pushrod_mount: upright)
    pushrod_inner: [0.0, 0.22, 0.42]   # on the rocker
    rocker_pivot: [0, 0.1998, 0.3743]  # a point on the rocker axis
    rocker_spring: [0.05, 0.2456, 0.3541]    # coilover pin on the rocker
    spring_chassis: [-0.0075, 0.2456, 0.5456] # coilover chassis mount
    wheel_center: [0.0, 0.60, 0.2032]  # x = 0 on the front axle; z ~ tyre_radius
    rocker_arb: [-0.04, 0.2364, 0.3581]      # ARB drop-link pickup (needed because arb is set)
    rocker_axis: [1.0, 0.0, 0.1]       # normalised on load
  wheel:
    tyre_radius: 0.2032         # m
    rim_width: 0.18             # m
    static_camber: -0.03        # rad (-1.7 deg)
    static_toe: 0.0             # rad
    tyre_vertical_stiffness: 90000.0   # N/m
  coilover:
    spring_rate: 35000.0        # N/m
    free_length: 0.200          # m
    preload: 0.0                # N, required field
    length_min: 0.150           # m
    length_max: 0.250           # m; static length (pin to chassis mount) must be inside
  arb:
    type: torsional
    arm_end: [-0.04, 0.20, 0.40]       # drop-link end of the arm
    bar_pivot: [-0.10, 0.20, 0.40]     # arm clamp on the tube; arm length 0.06 m
    bar_axis: [0.0, 1.0, 0.0]          # default
    bar_length: 0.40                   # = 2 * bar_pivot.y
    outer_diameter: 0.016              # m
    inner_diameter: 0.0                # solid bar
    shear_modulus: 7.9e10              # Pa
    # arm_diameter omitted -> rigid arms
  steering:                     # required on the front
    c_factor: 0.090             # m/rev
    rack_travel: [-0.030, 0.030]   # m, min < 0 < max
rear:
  brakes: outboard
  pushrod_mount: upright
  unsprung_mass: 12.0
  hardpoints:
    uca_front: [-1.50, 0.26, 0.30]
    uca_rear: [-1.70, 0.26, 0.29]
    uca_outer: [-1.60, 0.52, 0.30]
    lca_front: [-1.45, 0.22, 0.12]
    lca_rear: [-1.75, 0.22, 0.12]
    lca_outer: [-1.60, 0.55, 0.13]
    tie_inner: [-1.72, 0.20, 0.14]     # toe link, behind the axle
    tie_outer: [-1.72, 0.54, 0.15]
    pushrod_outer: [-1.60, 0.53, 0.27]
    pushrod_inner: [-1.60, 0.22, 0.42]
    rocker_pivot: [-1.6, 0.1982, 0.375]
    rocker_spring: [-1.55, 0.2432, 0.3532]
    spring_chassis: [-1.6075, 0.2432, 0.5448]
    wheel_center: [-1.55, 0.58, 0.2032]   # x = -wheelbase (1.55 m)
    rocker_arb: [-1.64, 0.2342, 0.3576]
    rocker_axis: [1.0, 0.0, 0.1]
  wheel:
    tyre_radius: 0.2032
    rim_width: 0.18
    static_camber: -0.026
    static_toe: 0.0
    tyre_vertical_stiffness: 90000.0
  coilover:
    spring_rate: 40000.0
    free_length: 0.200
    preload: 0.0
    length_min: 0.150
    length_max: 0.250
  arb:
    type: beam
    beam_center: [-1.64, 0.0, 0.40]    # on the centreline (|y| < 1 mm)
    beam_axis: [0.0, 1.0, 0.0]
    arm_end: [-1.64, 0.2342, 0.45]     # half-length a = 0.2342 m along the beam
    beam_width: 0.02                   # m
    beam_thickness: 0.003              # m
    beam_youngs_modulus: 2.1e11        # Pa
    beam_angle: 0.0                    # flat = soft
  # no steering block on the rear
```

## Annotated reference car

`configs/vehicles/fs_reference.yaml` is the fictional-but-consistent car used by every example
in these docs. Its header comment states the design intent; the numbers below are the part
that matters when you copy it.

```yaml
schema: 1
vehicle:
  name: fs_reference
  mass: 290.0              # kg, including driver
  cg_height: 0.28          # m
  front_mass_ratio: 0.47
  drive: rwd               # brake_bias_front defaults to 0.6; drive_split_front to 0 (rwd)

front:
  brakes: outboard
  pushrod_mount: uca       # pushrod_outer is on the upper wishbone, 50 mm inboard of uca_outer
  unsprung_mass: 10.0
  hardpoints:
    uca_front: [0.130, 0.300, 0.325]     # chassis upper rail
    uca_rear: [-0.130, 0.300, 0.325]
    uca_outer: [-0.009, 0.567, 0.338]    # upper ball joint
    lca_front: [0.150, 0.220, 0.125]
    lca_rear: [-0.150, 0.220, 0.125]
    lca_outer: [0.010, 0.586, 0.118]     # lower ball joint
    tie_inner: [0.085, 0.150, 0.154]     # rack end, ahead of the axle (front steer)
    tie_outer: [0.085, 0.585, 0.150]
    pushrod_outer: [-0.0087, 0.5175, 0.3374]   # on the UCA (pushrod_mount: uca)
    pushrod_inner: [0.000, 0.437, 0.509]       # rocker pickup, above the UCA
    rocker_pivot: [0.000, 0.385, 0.479]
    rocker_spring: [0.000, 0.385, 0.5575]      # directly above the pivot
    spring_chassis: [0.000, 0.185, 0.5575]     # horizontal transverse coilover, 0.200 m
    wheel_center: [0.000, 0.610, 0.228]        # track 1.220 m
    rocker_arb: [-0.090, 0.260, 0.479]         # drop-link pickup on the rocker
    rocker_axis: [1.0, 0.0, 0.0]               # rocker turns about x
  wheel:
    tyre_radius: 0.228
    rim_width: 0.178
    static_camber: -0.017453   # -1.0 deg
    static_toe: 0.0
    tyre_vertical_stiffness: 100000.0
  coilover:
    spring_rate: 35000.0
    free_length: 0.2164        # static length 0.200 m -> 35000 * 0.0164 = 574 N
    preload: 0.0
    length_min: 0.160
    length_max: 0.240
  arb:
    type: torsional
    arm_end: [-0.090, 0.260, 0.579]      # vertical drop link of 0.10 m to rocker_arb
    bar_pivot: [-0.240, 0.260, 0.579]    # arm length 0.15 m
    bar_axis: [0.0, 1.0, 0.0]
    bar_length: 0.52                     # = 2 * bar_pivot.y
    outer_diameter: 0.012
    inner_diameter: 0.010                # thin-wall tube
    shear_modulus: 7.93e10
    arm_diameter: 0.012                  # round solid arms bending in series with the tube
    arm_youngs_modulus: 2.0e11
  steering:
    c_factor: 0.090
    rack_travel: [-0.030, 0.030]
```

The rear axle repeats the layout at `x = -1.550` (wheelbase 1.550 m, track 1.180 m,
`static_toe: 0.001745` = +0.1 deg toe-in, `static_camber: -0.026180` = -1.5 deg,
`spring_rate: 40000.0`, `free_length: 0.2174`) with a **blade** ARB instead of the torsional
one:

```yaml
  arb:
    type: blade
    arm_end: [-1.640, 0.260, 0.579]
    bar_pivot: [-1.790, 0.260, 0.579]    # blade length 0.15 m
    bar_axis: [0.0, 1.0, 0.0]
    bar_length: 0.52
    outer_diameter: 0.016
    inner_diameter: 0.012
    shear_modulus: 7.93e10
    blade_width: 0.025
    blade_thickness: 0.003
    blade_youngs_modulus: 2.0e11
    blade_angle: 0.0                     # 0 = flat (soft), pi/2 = edge-on (stiff)
```

The rear has no `steering` block. The toe link `tie_inner` is a chassis point behind the
axle at `[-1.640, 0.150, 0.153]`.

## Checklist when you build your own car

1. Measure left-side hardpoints in the ISO frame at static ride height, origin at the front
   axle centre on the ground.
2. `wheel_center` z = loaded radius; set `tyre_radius` within 20 mm of it.
3. Make the static spring length (`rocker_spring` to `spring_chassis`) fit `[length_min, length_max]`.
4. If you add an ARB, add `rocker_arb`, then choose `type` and check `bar_length = 2 * bar_pivot.y`
   (torsional, blade) or `beam_center.y = 0` (beam).
5. Run `suspkin summary your.yaml` and `suspkin vehicle your.yaml`; both fail with a located
   message if something is off. See [Command line](cli.md).
