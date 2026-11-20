# Conventions: frame, units and signs

Every number that `suspkin` prints, stores or plots depends on three choices: the coordinate
frame, the unit system and the sign of each quantity. This page fixes all three. Read it once
before the other concept pages; they refer back to it instead of repeating the rules.

The rules are enforced in code, not just documented. Each sign below names the function that
implements it, and the last section shows a short script that checks the main ones on the
reference car (`configs/vehicles/fs_reference.yaml`).

## 1. Coordinate frame (ISO 8855)

The vehicle frame is the right-handed ISO 8855 frame:

| Axis | Direction |
|---|---|
| $x$ | forward |
| $y$ | left |
| $z$ | up |

The **origin is on the ground, at the mid-point of the front axle**: $x = 0$ at the front axle,
$y = 0$ on the car centreline, $z = 0$ on the ground plane at static ride height. The rear
axle is therefore at $x = -L$ with $L$ the wheelbase.

```text
          top view (z up, out of the page; the car faces up the page)

                       x (forward)
                       ^
          FL           |           FR
        (y > 0)        |        (y < 0)
    ---- O ------------+------------ O ----   front axle, x = 0
                       |
                       |  CG at x = -a
                       |
    ---- O ------------+------------ O ----   rear axle, x = -L
          RL           |           RR
                       |
        y <------------+
        (origin: front axle, centreline, ground)
```

Seen from above with the car facing up the page, $y$ points to the left of the page, so the
left corners FL and RL have $y > 0$.

The loader enforces the origin (`suspkin.geometry.model.Vehicle`):

* the front `wheel_center` has $|x| < 5$ mm (the origin is at the front axle) and is ahead of
  the rear `wheel_center` ($x_f > x_r$);
* each `wheel_center` height is within 20 mm of its `tyre_radius` (the ground is at $z = 0$
  and the wheel centre sits one loaded radius above it);
* `Vehicle.wheelbase` is $x_{wc,front} - x_{wc,rear}$; `Vehicle.front_track` and
  `Vehicle.rear_track` are $2\,y_{wc}$.

The YAML hardpoints describe the **static ride-height pose**: wheel travel 0, rack 0, body
heave, roll and pitch 0. All solved quantities are displacements from that pose.

### Body frame and ground frame

The solvers work in the **body frame**: the car body is fixed and the ground moves relative to
it (`suspkin.kinematics.corner`, `suspkin.kinematics.vehicle`). A vehicle pose (heave, roll,
pitch) is turned into a tilted, shifted ground plane expressed in the body frame, and each
wheel's travel is found so that the tyre touches that plane. Where a quantity has to be
measured against the *real* ground (camber to the road, steer in the ground plane, roll-centre
height) the result is rotated back with $R = R_y(\theta)R_x(\varphi)$; those columns carry the
suffix `_ground` or are named `rc_*_z` in `suspkin.sweeps.vehicle_frame`.

## 2. Corners, mirroring and the per-side sign

### Names

The four corners are `FL`, `FR`, `RL`, `RR` (front/rear, left/right); the type is
`suspkin.geometry.corners.CornerId`. `Corner.axle` is `"front"` or `"rear"`, `Corner.side` is
`"left"` or `"right"`.

### Mirroring rule

Hardpoints are entered **for the left side only**, with $y > 0$ for the outboard points (and
$y \ge 0$ for the inboard ones that may sit on the centreline: `tie_inner`, `rocker_pivot`,
`rocker_spring`, `spring_chassis`, `pushrod_inner`, `rocker_arb`). The loader rejects a
right-side coordinate typed into the left block. `suspkin.geometry.corners.build_corners`
builds the right corners by the reflection

$$
\mathbf p_R = M\,\mathbf p_L,\qquad M = \operatorname{diag}(1,-1,1)\quad (y \to -y)
$$

using `suspkin.geometry.vectors.mirror_y`. The rocker axis and the ARB arm, pivot and axis
are mirrored the same way (`suspkin.geometry.corners.arb_points`). The one exception is an
explicit asymmetric design: `Axle.right_overrides` replaces named right-side points, given
directly in right-side coordinates ($y < 0$). The rack axis is **not** mirrored (there is one
rigid rack, see section 5.1). ARB points (`suspkin.geometry.corners.arb_points`) always
mirror and ignore `right_overrides`.

### Why angles carry a per-side sign

A right corner is the reflection of a left corner. Reflection is an improper map
($\det M = -1$), and for a rotation $R(\mathbf a,\theta)$ about axis $\mathbf a$

$$
M\,R(\mathbf a,\theta)\,M = R(M\mathbf a,\,-\theta).
$$

So the mirror image of "rotate by $\theta$ about $\mathbf a$" is "rotate by **$-\theta$** about
the mirrored axis $M\mathbf a$". The code stores the mirrored axes and multiplies every
solver angle by $s = \pm 1$ (`suspkin.kinematics.rocker.side_sign`: $+1$ left, $-1$ right), so
that a left and a right corner doing mirror-image motion report **identical** angles. The
physical rotation is $s \cdot \text{angle}$ about the stored axis. This applies to the upper
and lower wishbone angles $\theta_{uca}$, $\theta_{lca}$, the upright spin $\psi$, the rocker
angle and the ARB arm angles $\beta$.

## 3. Units

### Internal units are SI

All internal values and all files (vehicle YAML, study YAML, result CSV/JSON, bundles) use SI:
metres, radians, newtons, N/m, N m/rad, kilograms, seconds. Angles are radians everywhere
inside the package.

| Quantity | SI unit | Display unit |
|---|---|---|
| length, hardpoint coordinate | m | mm |
| angle | rad | deg |
| force | N | N |
| stiffness, wheel rate, ride rate | N/m | N/mm |
| roll stiffness | N m/rad | N m/deg |
| angle per length (bump steer, camber gain) | rad/m | deg/mm |
| roll gradient | rad per g | deg/g |
| percentages (anti-dive, anti-lift, anti-squat, Ackermann) | the percent number itself (50 % is `50.0`) | % |
| front TLLTD (`tlltd_front`) | a fraction in $[0,1]$ | % (factor 100) |

Two quantities need a remark. Percent metrics are stored as the percent number (anti-dive 50 %
is `50.0`), **except** `tlltd_front` in the vehicle summary, which is a fraction in $[0,1]$ and
is shown with factor 100. And the roll gradient is stored in rad per g of lateral
acceleration.

### Quantity strings in YAML

A plain number in a study YAML is SI. A string with a unit is converted on load to SI and the
dimension is checked: `"30 mm"`, `"-1.5 deg"`, `"0.5 deg/deg"`, `"1.2 deg/g"`, `"34 N/mm"`,
`"50 %"`, `"30 min"`. `suspkin.units.parse_quantity(v, dim)` does this:

```python
from suspkin.units import parse_quantity
parse_quantity("30 mm", "length")        # 0.03
parse_quantity("-1.5 deg", "angle")      # -0.02617993877991494
parse_quantity("1.2 deg/g", "angle/gload")  # 0.020943951023931952 (rad per g)
parse_quantity("34 N/mm", "stiffness")   # 34000.0
parse_quantity("5 deg", "length")        # ValueError: unit 'deg' ... has dimension 'angle', expected 'length'
```

A dimension is a product of atoms (`length`, `angle`, `force`, `accel`, `gload`, `mass`,
`percent`, `time`) with integer exponents. `"ratio"` is the dimensionless one (`deg/deg` and
`mm/mm` are both a ratio, SI value unchanged). `suspkin.units.display_unit(dim)` and
`suspkin.units.display_factor(dim)` give the display unit and the SI-to-display factor of any
dimension (`length`: `mm`, 1000; `angle`: `deg`, 57.2958; `stiffness`: `N/mm`, 0.001).
`suspkin.units.slope_dim(metric_dim, input_dim)` gives the dimension of a derivative, e.g.
`angle/length` for bump steer.

### Where SI is converted to display units

Conversion happens only at the boundary where a human reads or types a number. There are
several such boundaries, each with its own table:

| Boundary | Direction | Code |
|---|---|---|
| CLI flags (`--heave`, `--rack`, `--travel`, `--pivot` in mm; `--roll`, `--pitch` in deg) | display to SI | `suspkin.cli.vehicle` (`_pose_kwargs` and the sweep commands divide by 1000 or use `math.radians`) |
| CLI metric tables (`suspkin sweep`, `summary`) | SI to display | `suspkin.cli.vehicle._display`, built on `suspkin.units.m_to_mm` and `suspkin.units.rad_to_deg` |
| `suspkin vehicle` table | SI to display | `suspkin.plotting.units.VEHICLE_CLI_UNITS` |
| Summary table (figure, report, web spec) | SI to display | `suspkin.plotting.units.SUMMARY_ROWS` |
| 2D figures | SI to display | `suspkin.plotting.figures` (`_KINDS`: `mm`, `deg`, `N/mm`, `1/mm`, ...) |
| Figure spec (web app) | SI to display | `suspkin.plotting.spec`, using `suspkin.units.display_factor` |
| Web app pose inputs | display to SI | `suspkin.app.routes` (`PoseInput.factor`, from `suspkin.units.display_factor`) |
| Study constraint/objective tables | SI to display | `suspkin.optimize.evaluate` (`display_factor`, `display_unit` per `dim`) |
| Study YAML values | display to SI | `suspkin.units.parse_quantity` (via `suspkin.optimize.spec.variables.parse_scalar`) |

Result tables written by the package (`sweep` CSV, `states_to_frame`, `vehicle_frame`,
`metrics_frame`) are **always SI**; the plotting layer converts when it draws.

Two display tables exist on purpose. `SUMMARY_ROWS` and `VEHICLE_CLI_UNITS` in
`suspkin.plotting.units` differ in unit spelling (`N m/deg` vs `N*m/deg`) and in a few extra
fields, because the `suspkin vehicle` CLI output is a stable surface and must not change. See
[ADR 0008](../adr/0008-2026-restructure.md).

!!! note "Dead override in `suspkin.units`"
    `_DISPLAY_OVERRIDE` is keyed `"force*length/angle"`, but `suspkin.units.normalize_dim`
    produces the canonical name `"length*force/angle"` (atoms in the fixed order of
    `_ATOMS`). The override therefore never matches and the roll-stiffness dimension displays
    as `mm*N/deg` (factor 17.4533, numerically correct: 1 N m/rad = 17.45 N mm/deg) instead
    of the intended `N*m/deg`. It is cosmetic. `suspkin vehicle` prints `N*m/deg` through
    `VEHICLE_CLI_UNITS`, the figure and web summary prints `N m/deg` through `SUMMARY_ROWS`, and
    `suspkin summary` has no roll-stiffness row. Only callers of the generic
    `display_unit` / `display_factor` on that dimension show `mm*N/deg`: the constraint and
    objective tables of `suspkin.optimize.evaluate` for the summary roll-stiffness metrics
    (dimensions come from `suspkin.optimize.registry`) and `suspkin.plotting.spec`.

## 4. Sign conventions at a glance

All signs below are the same for left and right corners unless stated ("L/R-symmetric": a
mirror-image motion gives equal values). The sections that follow derive each one.

| Quantity | Positive means |
|---|---|
| wheel travel $z$ | bump: wheel moves **up** relative to the body |
| body heave $h$ | body **up** (wheels droop, travel < 0) |
| body roll $\varphi$ | right-hand rotation about $+x$: the **left** side goes **up** (the roll of a left turn) |
| body pitch $\theta$ | right-hand rotation about $+y$: nose **down** |
| rack displacement $r$ | both inner tie-rod points move toward $+y$ (left); steers **left** |
| steer angle $\delta$ | wheel heading points **left** (ISO yaw) |
| camber $\gamma$ | top of the wheel leans **away** from the car (negative = toward the centre) |
| toe | **toe-in**: wheel front points toward the centreline |
| caster | top of the steering axis **rearward** |
| kingpin inclination (KPI) | top of the steering axis **inboard** |
| scrub radius | steering-axis ground intercept **inboard** of the contact patch |
| mechanical trail | steering-axis ground intercept **ahead** of the contact patch |
| bump steer | $d(\text{toe})/dz$, toe-in per metre of bump |
| swing angle, front view | FVIC **above** the contact patch, horizontal axis positive **inboard** |
| swing angle, side view | **anti-geometry** (resists dive / lift / squat) |
| anti-dive, anti-lift, anti-squat | anti-geometry, i.e. the suspension resists the pitch motion |
| coilover compression | spring **shorter** than static (bump) |
| motion ratio | bump compresses the spring |
| ARB twist | $\beta_L - \beta_R$ in the per-side sign (zero in pure heave) |
| joint angle | **unsigned** misalignment angle, $\ge 0$ |

## 5. Signs in detail

### 5.1 Motion inputs

**Wheel travel $z$.** Positive = bump = the wheel centre moves up relative to the body. The
corner solver prescribes the wheel-centre height: the third residual is
$z_{lbj} + (\text{wheel-centre offset})_z - z_{wc,0} - z$, so $z > 0$ raises the wheel centre
(`suspkin.kinematics.corner`, `_eval`). Public entry: `suspkin.kinematics.corner.solve_corner`.

**Heave, roll, pitch, rack.** Defined by `suspkin.kinematics.vehicle`. A body point maps to
the world as

$$
\mathbf x_w = R(\mathbf x_b - \mathbf c) + \mathbf c + h\,\mathbf e_z,\qquad
R = R_y(\theta)\,R_x(\varphi),
$$

with pivot $\mathbf c$ (default the vehicle CG, `suspkin.kinematics.vehicle.cg_pivot`:
$(-a, 0, h_{cg})$ with $a = L(1 - f_{mass})$) and the right-hand rotation matrices of
`suspkin.geometry.vectors.body_rotation` (and `body_rotation_batch`). Consequences:

* $R_x(\varphi)$ maps $(0,1,0) \mapsto (0,\cos\varphi,\sin\varphi)$, so **positive roll lifts the
  left side**. The body leans to the right, which is the roll of a left turn. The outer wheel
  for positive roll is therefore the right wheel (FR/RR).
* $R_y(\theta)$ maps $(1,0,0) \mapsto (\cos\theta, 0, -\sin\theta)$, so **positive pitch puts
  the nose down** (braking dive is positive pitch, acceleration squat is negative).
* Positive heave lifts the body, so wheel travel is **negative** (the wheels droop).

The roll is applied first, then the pitch about the fixed ground-aligned $y$ axis. The pivot
applies to roll and pitch, not to heave.

**Rack displacement $r$.** `suspkin.geometry.model.Steering` documents it: positive rack travel
moves both `tie_inner` points along `+rack_axis` (default $(0,1,0)$, i.e. left), so for
front-steer geometry (tie rods ahead of the axle) it steers left. In code
`suspkin.kinematics.corner._tie_inner` returns `tie_inner + rack * rack_axis`. The rack is one
rigid part, so the same global `rack_axis` is used on both corners; mirroring it would make the
right wheel steer the opposite way. `rack` must be 0 on an axle without steering. Rack travel
is limited by `rack_travel = (min, max)` with `min < 0 < max`.

Worked check on the reference car (the script is in section 6):

| Input | FL | FR | RL | RR |
|---|---|---|---|---|
| heave +10 mm, travel [mm] | -9.99 | -9.99 | -9.98 | -9.98 |
| roll +1 deg, travel [mm] | -10.74 | +10.67 | -10.41 | +10.34 |
| pitch +1 deg, travel [mm] | +14.31 | +14.31 | -12.70 | -12.70 |

Positive roll droops the left wheels (left side up) and bumps the right wheels; positive pitch
bumps the front (nose down) and droops the rear.

### 5.2 Wheel alignment

All alignment columns come from `suspkin.metrics.arrays.corner_arrays` (and its public wrapper
`corner_metric_arrays`, which is what `suspkin.metrics.table.metrics_frame` returns). Let
$\mathbf n$ be the unit wheel **spin axis**, pointing **outboard**
(`suspkin.geometry.vectors.wheel_spin_axis`: $(0,1,0)$ for the left wheel at zero camber and
toe, $(0,-1,0)$ for the right), and $o = +1$ on left corners, $-1$ on right corners. Let
$\mathbf u = \mathbf p_{uca,outer} - \mathbf p_{lca,outer}$ be the steering-axis (kingpin)
direction.

**Camber.** Negative = the top of the wheel leans toward the car centre.

$$
\gamma = -\arcsin(n_z)
$$

For a left wheel with its top leaning inboard, the wheel plane normal tilts so that $n_z > 0$,
hence $\gamma < 0$. The sign is the same on both sides because $n$ points outboard on both.
`wheel_spin_axis` builds the static axis as $(0,\cos\gamma_0,-\sin\gamma_0)$ then applies the
toe rotation, and mirrors it for the right side. Code: `out["camber"]` in
`suspkin.metrics.arrays.corner_arrays`. In vehicle sweeps `<id>_camber_ground` applies the
same formula to the spin axis rotated into the ground frame.

**Toe.** Positive = toe-in (the front of the wheel points toward the centreline), equal on
both sides:

$$
\text{toe} = \operatorname{atan2}(n_x,\; o\,n_y)
$$

`wheel_spin_axis` rotates the axis by $-\text{toe}$ about $z$, so a positive toe on the left
gives $n_x > 0$. Check on the reference rear axle: static toe +0.1 deg gives `toe` = +0.100 deg
on both RL and RR.

**Steer angle.** The road-wheel heading in the ISO yaw sense, positive = wheel points left:

$$
\delta = \operatorname{atan2}(-o\,n_x,\; o\,n_y)
$$

Hence $\delta = -\text{toe}$ on a left wheel and $\delta = +\text{toe}$ on a right wheel
(rear axle at the reference car: RL steer = -0.100 deg, RR = +0.100 deg). Scalar form:
`suspkin.metrics.corner.steer_angle`; column `steer_angle` in `corner_arrays`; ground-plane
form `<id>_steer_ground` in `suspkin.sweeps.vehicle_frame`. This is the quantity differentiated
for `steer_gain` (section 5.8).

**Caster.** Positive = the top of the steering axis is rearward of the bottom:

$$
\text{caster} = \operatorname{atan2}(-u_x,\; u_z)
$$

**Kingpin inclination (KPI, SAI).** Positive = the top of the axis is inboard:

$$
\text{KPI} = \operatorname{atan2}(-o\,u_y,\; u_z)
$$

**Scrub radius.** Let $\mathbf g_{cp}$ be the point where the steering-axis line meets the
height of the contact patch (`ground()` in `corner_arrays`: the line from the LBJ through the
UBJ evaluated at $z = z_{cp}$). Then

$$
r_s = o\,(y_{cp} - y_{g})
$$

Positive = the axis intercept is **inboard** of the contact patch (the usual positive scrub).

**Mechanical (caster) trail.**

$$
t_m = x_{g} - x_{cp}
$$

Positive = the intercept is **ahead** of the contact patch (positive trail, self-aligning).

**Kingpin offset at wheel centre.** $o\,(y_{wc} - y_{g,wc})$ with $\mathbf g_{wc}$ the axis
point at wheel-centre height. Positive = the wheel centre is outboard of the axis.

**Spindle length.** Always $\ge 0$: the perpendicular distance from the wheel centre to the
steering axis.

Static values on the reference front-left corner (from `corner_metric_arrays` at travel 0):

| Metric | Value |
|---|---|
| camber | -1.000 deg |
| toe | 0.000 deg |
| caster | 4.936 deg |
| KPI | 4.936 deg |
| scrub radius | 17.79 mm |
| mechanical trail | 20.19 mm |

### 5.3 Wheel position

All relative to the static pose of the same corner; $o$ as above.

| Column | Definition | Positive means |
|---|---|---|
| `half_track` | $\lvert y_{cp}\rvert$ | always positive |
| `half_track_change` | $\lvert y_{cp}\rvert - \lvert y_{cp,0}\rvert$ | track gets wider (**outboard**) |
| `contact_patch_dy` | $o\,(y_{cp} - y_{cp,0})$ | contact patch moves **outboard** |
| `contact_patch_dx` | $x_{cp} - x_{cp,0}$ | contact patch moves **forward** |
| `wheel_center_recession` | $x_{wc} - x_{wc,0}$ | wheel centre moves **forward** (despite the name, positive is not rearward) |
| `wheelbase_change_contribution` | $+\Delta x_{cp}$ front, $-\Delta x_{cp}$ rear | this corner **lengthens** the wheelbase |

### 5.4 Instant centres and swing angles

The front-view instant centre (FVIC, in the $(y,z)$ plane) and side-view instant centre (SVIC,
in the $(x,z)$ plane) come from the plane method: each wishbone plane passes through its two
chassis pivots and its outer ball joint; both planes are cut with the plane $x = x_{cp}$ (front
view) or $y = y_{cp}$ (side view) and the two section lines are intersected
(`suspkin.kinematics.ic_arrays.ic_arrays`; scalar reference `suspkin.metrics.corner.front_view_ic`
and `side_view_ic`). Parallel section lines mean an instant centre at infinity: the coordinates
are NaN, the common direction is kept, and nothing raises.

A **swing angle** is the signed angle of the line reference point $\to$ IC:

$$
\alpha = \operatorname{atan2}\!\bigl(z_{ic} - z_{ref},\; \sigma\,(h_{ic} - h_{ref})\bigr),
\qquad \alpha \in (-\pi,\pi]
$$

where $h$ is $y$ (front view) or $x$ (side view) and $\sigma = \pm 1$ chooses the positive
horizontal direction (code: `suspkin.kinematics.ic_arrays.swing_angle` and the scalar
`suspkin.metrics.corner._angle`). Positive $\alpha$ always means the IC is **above** the
reference point; $\lvert\alpha\rvert > \pi/2$ means the IC is on the opposite horizontal side.
For an IC at infinity the line direction is oriented to the $\sigma$ side, so the convention
continues smoothly.

| Column | Reference point | $\sigma$ |
|---|---|---|
| `fv_swing_angle` | contact patch | $-1$ on left corners, $+1$ on right: positive horizontal = **inboard** |
| `sv_swing_angle` | contact patch | $-1$ front axle (rearward positive), $+1$ rear axle (forward positive) |
| `sv_wc_angle` | wheel centre | as `sv_swing_angle` |

The side-view choice makes a positive angle mean **anti-geometry** on either axle: on the front
axle the SVIC rearward of and above the reference point, on the rear axle forward of and above
it (`suspkin.metrics.corner._anti_sign`, `inboard` and `anti` in `corner_arrays`). Swing-arm
lengths `fvsa` and `svsa` are $\lvert h_{ic} - h_{cp}\rvert$ (always $\ge 0$, `inf` at infinity).

The per-corner `rc_height` column is the height at $y = 0$ of the line contact patch $\to$ FVIC
(a single-wheel roll centre); the true axle roll centre needs both sides (section 5.6).

### 5.5 Anti-features

Implemented in `suspkin.metrics.vehicle`; $L$ wheelbase, $h_{cg}$ CG height, $b_f$ front brake
bias, $s_f$ front drive-torque share, $\alpha$ the swing angle of section 5.4:

$$
\begin{aligned}
\text{anti-dive (front)} &= \tan\alpha_{cp}\; b_f\, \tfrac{L}{h_{cg}} \cdot 100\\
\text{anti-lift (rear)} &= \tan\alpha_{cp}\;(1-b_f)\,\tfrac{L}{h_{cg}} \cdot 100\\
\text{anti-squat (rear)} &= \tan\alpha_{wc}\,(1-s_f)\,\tfrac{L}{h_{cg}} \cdot 100
\end{aligned}
$$

(`anti_dive_front`, `anti_lift_rear`, `anti_squat_rear`; `anti_lift_front_accel` is the front
axle under acceleration.) Positive percent = the suspension geometry **resists** the pitch
motion; 100 % would cancel it completely. Which angle applies is chosen by
`suspkin.metrics.vehicle.anti_angle_for`: braking uses the contact-patch angle for outboard
brakes and the wheel-centre angle for inboard brakes; drive always uses the wheel-centre angle.
On the reference car the side-view lines are horizontal, so every anti percentage is 0.

### 5.6 Roll centre, roll and load transfer

**Roll-centre height and lateral position.** For each wheel the line contact patch $\to$ FVIC
is drawn in the front view; the roll centre of the axle is the intersection of the left and
right lines (`suspkin.metrics.corner.roll_centre_pair`, vectorised in `suspkin.sweeps`).
`rc_front_y`, `rc_rear_y` are the lateral position in the **body frame**, positive toward $+y$
(left); `rc_front_z`, `rc_rear_z` are the height above the **ground**, i.e. rotated into the
ground frame. NaN if undefined (parallel distinct lines). Static values on the reference car:
front 33.72 mm, rear 40.00 mm.

**RC migration** `rc_migration` $= d y_{rc}/d\varphi$ [m/rad]: positive = the roll centre moves
toward $+y$ (left) for left-up roll. **RC height curvature** $= d^2 z_{rc}/d\varphi^2$ (the
first derivative vanishes by symmetry). Both are cubic least-squares slopes at zero roll
(`suspkin.metrics.vehicle._slope0`, `_curv0`).

**Roll camber** `roll_camber` $= d\gamma_{ground}/d\varphi$ of the **outer** wheel for positive
roll, which is the right wheel (`FR_camber_ground`, `RR_camber_ground`). It includes the body
roll itself: rolling the body by $\varphi$ tilts the wheel top outward by $\varphi$, and the
suspension camber gain (more negative camber in bump) only partly cancels it. On the reference
car the slope is $+0.81$ deg/deg at the front and $+0.76$ deg/deg at the rear, so the outer
wheel *loses* negative camber relative to the ground as the body rolls. **Roll steer**
`roll_steer` $= d\,\overline{\delta_{ground}}/d\varphi$, the mean of left and right ground steer angles.

**Roll gradient** [rad per g], with $K = K_f + K_r - m_s g\,d$ and $d = h_{cg} - h_{ra}$:

$$
\frac{\varphi}{a_y/g} = \frac{m_s g\, d}{K}
$$

(`suspkin.metrics.vehicle.roll_gradient`; positive = the body rolls **outward** in a turn;
raises `ValueError` if $K \le 0$). **Axle roll stiffness** $K_i = k_w t^2/2$ for springs, ARB
and tyres, combined as (springs + ARB) in series with the tyres.

**Lateral load transfer** `LoadTransfer`: each value is the per-wheel shift $\Delta W$; the
outside wheel gains $\Delta W$ and the inside wheel loses it, so the outside-minus-inside load
difference of the axle is $2\Delta W$. `tlltd_front` $= \Delta W_f/(\Delta W_f + \Delta W_r)$ is
a fraction in $[0,1]$ (summary tables show it times 100).

### 5.7 Actuation and rates

Pushrod, rocker and spring quantities (columns of `corner_arrays`):

| Column | Definition | Positive means |
|---|---|---|
| `pushrod_angle` | $\operatorname{atan2}(r_z, \sqrt{r_x^2+r_y^2})$, $\mathbf r = \mathbf p_{inner} - \mathbf p_{outer}$ | the rod rises from the outer to the inner end (reference car: 64.7 deg) |
| `pushrod_angle_fv` | $\operatorname{atan2}(r_z, \lvert r_y\rvert)$ | same, in the front view |
| `rocker_rotation` | $\chi - \chi_{static}$ in the per-side sign | the same rotation on both sides for mirror motion |
| `coilover_compression` | $L_{static} - L$ | spring **shorter** than static (bump) |
| `bump_stroke_remaining` | $L - L_{min}$ | stroke left before bump stop |
| `droop_stroke_remaining` | $L_{max} - L$ | stroke left before droop stop |
| `spring_force` | $F_{pre} + k(L_0 - L)$ | compressive; never clamped, so it can be negative |

The **motion ratio** is $MR = -dL/dz = d(\text{coilover\_compression})/dz$, positive when bump
compresses the spring (column `mr_spring`, from `suspkin.metrics.table.add_gradients`). Virtual
work then gives

$$
F_w = F_s\,MR,\qquad k_w = k\,MR^2 + F_s\,\frac{dMR}{dz},\qquad
\frac{1}{k_{ride}} = \frac{1}{k_w} + \frac{1}{k_t}
$$

(`suspkin.metrics.corner.wheel_load_spring`, `wheel_rate`, `ride_rate`; `mr_progression` is
$dMR/dz$). Rates are in N/m internally. The **toggle margin** (`CornerState.toggle_margin`) is
$\min(a, \pi - a)$ with $a$ the angle between pushrod and rocker lever in the plane normal to
the rocker axis; it is $\ge 0$ and measures how far the rocker is from its dead centre.

### 5.8 Gradients

`suspkin.metrics.table.add_gradients` adds $d(\text{metric})/d(x)$ columns by `np.gradient`
(central differences, one-sided at the ends). The map is `suspkin.kinematics.batch.GRADIENTS`:

| Sweep variable | Metric | Column | Unit (SI) |
|---|---|---|---|
| travel | `camber` | `camber_gain` | rad/m |
| travel | `toe` | `bump_steer` | rad/m (positive = toe-in per metre of bump) |
| travel | `caster` | `caster_change` | rad/m |
| travel | `half_track_change` | `track_change_rate` | m/m |
| travel | `rc_height` | `rc_migration_rate` | m/m (per-wheel RC height vs travel; not the roll `rc_migration`) |
| rack | `steer_angle` | `steer_gain` | rad/m |

The sweep variable must be strictly monotonic with at least 3 rows. `suspkin.metrics.table.static_summary`
reads each column at $x = 0$ by linear interpolation. The CLI shows `bump_steer` and
`camber_gain` in deg/mm.

### 5.9 Steering

* **Inner and outer wheel.** `suspkin.metrics.vehicle.inner_outer(delta_left, delta_right)`:
  if the mean steer is $\ge 0$ (left turn) the left wheel is inner, else the right wheel.
* **Ackermann percent.** With the ideal inner angle from
  $\cot\delta_o - \cot\delta_i = t/L$:

  $$
  A = \frac{\delta_i - \delta_o}{\delta_{i,ideal} - \delta_o}\cdot 100
  $$

  100 = true Ackermann, 0 = parallel steer, negative = anti-Ackermann; angles are mirrored
  so the turn is positive; NaN when $\lvert\delta_o\rvert < 10^{-4}$ rad
  (`suspkin.metrics.vehicle.ackermann_percent`). Summary values at 5, 10 and 15 deg of outer
  steer are interpolated on the lock side.
* **Steering ratio** = handwheel angle / mean road-wheel angle, with handwheel angle
  $= r/c_{factor}\cdot 2\pi$ and `c_factor` the rack travel per pinion revolution (m/rev)
  (`suspkin.metrics.vehicle.steering_ratio`). Reported dimensionless, "deg/deg".
* **Turning radius** $= L/\tan\bigl(\tfrac12(\delta_i + \delta_o)\bigr)$ to the rear-axle
  centre (`inf` for zero steer).
* The steer angles in the summary are relative to the straight-ahead pose.

### 5.10 Anti-roll bar

Defined in `suspkin.kinematics.arb`. For `torsional` and `blade` bars each side's arm rotates
about the bar axis by $\beta$ chosen so the drop link keeps its static length. The reported
angles use the per-side sign of section 2 (physical rotation $= s\,\beta$ about the stored,
mirrored axis), so **symmetric heave gives $\beta_L = \beta_R$**. The elastic coordinate is

$$
q = \text{twist} = \beta_L - \beta_R \quad[\text{rad}],
$$

zero in pure heave. For the `beam` type the beam ends move vertically, left
$\mathbf a + (c+\delta)\mathbf e_z$ and right $M\mathbf a + (c-\delta)\mathbf e_z$, with slider
heave $c$ and elastic coordinate $\delta$ [m]. The static pose is the stress-free state of the
bar.

Sign check on the reference car: roll +1 deg (left up, left wheel in droop) gives front twist
$-0.223$ rad; pure heave +10 mm gives exactly 0.

Motion ratios (`suspkin.metrics.arb.arb_rates`, central differences with $h = 10^{-5}$ m):
`motion_ratio_roll` is $dq/dz$ for left $+z$ and right $-z$; `motion_ratio_single` for left $+z$
with the right wheel fixed. With bar energy $E = \tfrac12 k q^2$, virtual work gives the
per-wheel rates

$$
k_{w,roll} = \tfrac12\,k\,MR_{roll}^2,\qquad k_{w,single} = k\,MR_{single}^2 .
$$

On the reference front bar: $MR_{roll} = 20.82$ rad/m, $k_{w,roll} = 32.3$ N/mm. Stiffness
units: N m/rad for twist bars, N/m for the beam.

### 5.11 Solver and joint angles

* **Solver unknowns** $\theta_{uca}$, $\theta_{lca}$, $\psi$ and the **rocker angle** $\chi$ are
  right-hand rotations about the *left* corner's axes, reported with the per-side sign of
  section 2 (`CornerState.theta_uca`, `theta_lca`, `psi`, `rocker_angle`).
* **Joint misalignment angles** (`joint_<name>` columns, `suspkin.metrics.arrays.joint_angle_arrays`)
  are **unsigned**: $\operatorname{atan2}(\lvert\mathbf a\times\mathbf b\rvert,\ \mathbf a\cdot\mathbf b)\in[0,\pi]$
  (`suspkin.metrics.corner.angle_between`), the angle between a link (or ball-joint stud) and
  the direction carried by its housing body, measured against the static pose with body
  rotations from a Kabsch fit. Joints: `tie_inner`, `tie_outer`, `pushrod_outer`,
  `pushrod_inner`, `spring_chassis`, `rocker_spring`, `uca_outer`, `lca_outer`.

### 5.12 Angle ranges

* Swing angles: $(-\pi, \pi]$ (`suspkin.kinematics.ic_arrays.swing_angle`); the surrogate
  re-expresses them as line angles in $(-\pi/2,\pi/2]$, see [ADR 0007](../adr/0007-surrogate-model.md).
* `suspkin.geometry.vectors.wrap_angle`: $[-\pi,\pi)$.
* Camber, toe, steer, caster, KPI: small angles, no wrapping needed in normal ranges
  (`Wheel.static_camber` is limited to $\pm 0.2$ rad, `static_toe` to $\pm 0.1$ rad).

## 6. Checking the signs yourself

This script (run from the repository root) reproduces the table of section 5.1 and the rack
signs:

```python
import math

from suspkin.geometry import load_vehicle
from suspkin.geometry.corners import build_corners
from suspkin.kinematics import solve_corner, solve_vehicle
from suspkin.metrics.arrays import corner_metric_arrays

v = load_vehicle("configs/vehicles/fs_reference.yaml")
deg = math.radians(1.0)
for label, pose in [("heave +10 mm", {"heave": 0.010}),
                    ("roll  +1 deg", {"roll": deg}),
                    ("pitch +1 deg", {"pitch": deg})]:
    s = solve_vehicle(v, **pose)
    print(label, {k: round(c.travel * 1000, 2) for k, c in s.corners.items()}, "travel [mm]")

cs = build_corners(v)
for cid in ("FL", "FR"):
    m = corner_metric_arrays([solve_corner(cs[cid], 0.0, rack=0.010)], cs[cid])
    print(cid, "rack +10 mm: steer %.2f deg, toe %.2f deg"
          % (math.degrees(m["steer_angle"][0]), math.degrees(m["toe"][0])))
```

Output on the reference car:

```text
heave +10 mm {'FL': -9.99, 'FR': -9.99, 'RL': -9.98, 'RR': -9.98} travel [mm]
roll  +1 deg {'FL': -10.74, 'FR': 10.67, 'RL': -10.41, 'RR': 10.34} travel [mm]
pitch +1 deg {'FL': 14.31, 'FR': 14.31, 'RL': -12.7, 'RR': -12.7} travel [mm]
FL rack +10 mm: steer 7.41 deg, toe -7.41 deg
FR rack +10 mm: steer 7.40 deg, toe 7.40 deg
```

A positive rack steers both wheels left (steer $> 0$). Because toe-in is positive, the left
wheel turning left toes **out** ($-7.41$ deg) while the right wheel toes **in** ($+7.40$ deg):
`steer` $= -\text{toe}$ on the left and $+\text{toe}$ on the right. The small left/right
difference is the real Ackermann effect of the geometry, not a sign error.

## Where it is tested

* `tests/unit/` holds the sign checks of the metric columns against analytic constructions
  (camber, toe, steer, caster, KPI, swing angles, anti-features).
* `tests/regression/golden/` freezes the reference-car corner and vehicle sweeps, so any sign
  change shows up as a golden diff.
* `tests/regression/behaviour/` locks the CLI output, including the display conversions.

## See also

* [Vehicle model](vehicle-model.md): hardpoints, corner construction and mirroring in detail.
* [ADR 0003](../adr/0003-frame-units-conventions.md): why the frame, units and signs are what they are.
* [ADR 0001](../adr/0001-webapp-architecture.md) and the other decision records in `docs/adr/`.
