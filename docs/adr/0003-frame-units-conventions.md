# ADR 0003: Frame, units, symmetry and body-pose conventions

Status: accepted. Date: 2026-11-17.

## Context

Suspension quantities are easy to misread: camber and toe have opposite natural signs on the
two sides, "roll" can mean either direction, and mm/deg are what engineers read while the
maths wants m/rad. Mixing them silently gives plausible but wrong numbers. The package needs
one frame, one internal unit system and one rule for the two sides that every module follows.

## Decision

* **Frame:** ISO 8855, $x$ forward, $y$ left, $z$ up. The origin is on the ground at the
  mid-point of the front axle, so the rear axle is at $x = -L$. The loader enforces it
  (`suspkin.geometry.model.Vehicle`).
* **Units:** SI everywhere internally and in all files (m, rad, N, N/m, N m/rad, kg, s).
  Display units (mm, deg, N/mm) exist only at the boundary where a person reads or types a
  number, and are always converted through a named table (`suspkin.units`,
  `suspkin.plotting.units`; see [Conventions](../concepts/conventions.md)).
* **Quantities with units in YAML:** a value may be a string with a unit, `"30 mm"`,
  `"-1.5 deg"`, `"0.5 deg/deg"`, `"34 N/mm"`. A plain number means SI. Everything is converted
  to SI on load and checked against the dimension of the field (decided with the study format,
  2026-11-18).
* **Symmetry:** hardpoints are given for the **left** side and the right side is the
  reflection $y \to -y$. `right_overrides` replaces named right-side points for an asymmetric
  design. Reflection flips the sense of rotations, so solver angles carry a per-side sign
  (`suspkin.kinematics.rocker.side_sign`) and a mirror-image motion reports identical values.
* **Corner names:** `FL`, `FR`, `RL`, `RR`.
* **Sign conventions:** camber negative = top of wheel toward the car centre; toe-in
  positive; wheel travel positive = bump; roll positive = right-hand rotation about $+x$
  (left side up); pitch positive = right-hand rotation about $+y$ (nose down); heave positive
  = body up; steer and rack positive = left. The full table is in
  [Conventions](../concepts/conventions.md).
* **Body pivot:** roll and pitch rotate the body about the vehicle **CG**
  ($x = -a$, $y = 0$, $z = h_{cg}$) by default (decided 2026-11-17). The vehicle solve, sweep and
  summary functions accept an explicit `pivot`.
* **Solver frame:** solvers work in the body frame (the body is fixed, the ground moves);
  results that need the real ground are rotated back and carry a `_ground` suffix.

## Consequences

* A reader can trust any table column to be SI unless it is explicitly a display table.
* Left and right outputs are directly comparable (mirror-image motion gives equal values),
  which simplifies tests (`test_mirror_symmetry` style invariants) and optimisation
  (a design variable moves the left point and the right side follows).
* There are several display tables (CLI, figures, summary, web spec, study tables). They are
  named in [Conventions](../concepts/conventions.md) and kept in sync by tests, not by one
  global table: the CLI output is a stable surface and cannot change spelling (ADR 0008).
* A latent inconsistency is recorded there: the roll-stiffness dimension displays as
  `mm*N/deg` because an override in `suspkin.units` is keyed with a non-canonical name.
