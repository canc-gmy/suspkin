# ADR 0002: Scope and modelling level

Status: accepted. Date: 2026-11-17 (updated 2026-11-18).

## Context

`suspkin` exists to design and compare Formula Student double-wishbone suspensions. The team
needs hardpoint-level kinematics (alignment, centres, rates, anti-features, roll behaviour)
quickly and repeatably, and an optimiser that can call it thousands of times. A full
multibody or K&C model would cost far more to build and to run, and the team has no
compliance data to feed it. Possible real-car hardpoints may exist later and must be able to
drop in without a schema change.

## Decision

* **Vehicle class:** Formula Student.
* **Architecture:** double wishbone, pushrod, rocker and coilover on both axles; a steering
  rack on the front axle and a fixed-length toe link on the rear.
* **Physics level:** rigid-body kinematics only. Links are rigid, joints are ideal, there are
  no forces and no compliance. Wheel and ride rates come from virtual work on the kinematic
  motion ratio, not from a force solver.
* **Motions:** corner bump and rebound, heave, roll, pitch, steer and any combination.
* **Metrics:** the full set (alignment, wheel position, instant centres and roll centre,
  anti-features, actuation and rates, anti-roll bar, roll, steering). They are listed in
  `suspkin.metrics`.
* **Pushrod mount:** the solver supports a pushrod on the `upright`, the `lca` or the `uca`
  (`Axle.pushrod_mount`). The reference car uses the **UCA on both axles**, with the rod in
  compression going up to a rocker mounted above the UCA (changed on 2026-11-17 from the
  upright).
* **Anti-roll bar:** both axles, type chosen per axle. `torsional` (round arms; bar twist
  $GJ/L$ in series with arm bending $3EI/L^3$), `blade` (torsion tube plus flat blade arms
  with an orientation angle, flat = soft, edge-on = stiff) and `beam` (a transverse bending
  bar with a heave-free central slider, clamped in rotation; roll bends the two halves as
  cantilevers). This replaced an earlier single `flexural` type.
* **Brakes and drive:** outboard brakes and rear-wheel drive by default; both are
  configurable (`Axle.brakes`, `VehicleParams.drive`, `brake_bias_front`,
  `drive_split_front`).
* **Tyre:** a rigid disc. The contact point is the lowest point of the tyre circle along the
  ground normal. Only a vertical stiffness is used, for the ride rate.
* **Validation data:** none external for now. The YAML schema accepts user hardpoints
  directly, so a real car can be added later as a regression case. Validation rests on
  analytic tests and golden snapshots (ADR 0006).

Deferred, not rejected: finite-element analysis of members, a force solver, and elasto-kinematics
(K&C compliance). A neural-network surrogate of the *kinematics* (not of stresses) was first
planned for a later release and was then brought forward; see ADR 0007.

## Consequences

* Cheap, deterministic solves (well under a millisecond per corner solve (0.46 ms measured, historical)) that an optimiser can use.
* Results are valid only where compliance is negligible: no steer or camber compliance, no
  tyre loaded-radius change under load transfer.
* Adding a suspension layout other than double wishbone is a larger change than adding a
  metric. The pushrod mount, brake and drive layout are already switches, so those variants
  need no code.
* The ARB types and the pushrod mount are part of the vehicle schema and of the stable
  surface (ADR 0006).
