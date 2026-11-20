# ADR 0004: Corner solver: damped Newton with analytic Jacobian

Status: accepted. Date: 2026-11-17.

## Context

Each corner is a closed-loop mechanism. With the upper and lower wishbones as rotations about
their chassis axes, the upright pose follows from two angles and a spin, and three closing
equations remain (ball-joint distance, tie-rod length, prescribed wheel-centre height). The
solver runs inside sweeps, vehicle poses and optimisation loops, so cost per solve dominates
total run time. The first implementation used `scipy.optimize.root` (`hybr`) with finite
differences: about 15 residual evaluations per solve, with numpy overhead on 3-vectors.

## Decision

* The solver is **our own**, built on the formulation of `suspkin.kinematics.corner`: unknowns
  $\theta_{uca}$, $\theta_{lca}$, $\psi$; residuals as above; the rocker, spring length and
  toggle margin follow in closed form (`suspkin.kinematics.rocker`).
* The primary method is a **damped Newton iteration with an analytic Jacobian** on plain
  Python floats and tuples (no numpy on the hot path). Convergence tolerance on the
  residual is $10^{-12}$ with a stall tolerance of $10^{-10}$.
* **`scipy` `hybr`**, with the same analytic Jacobian, stays as the fallback when Newton
  fails.
* Far guesses and large steps go through **staged continuation** from the guess, with a
  linear predictor and a branch-jump guard: a step that changes an angle by more than
  0.05 rad is halved (up to six times).
* Vehicle poses solve each corner for the travel at which the tyre touches the (tilted) ground
  with a 1-D secant iteration, warm-started from the neighbouring pose.
* An **optional numba fast path** (`suspkin.kinematics._numba`, extra `fast`) is a deliberate
  twin of the numpy-free solver; a test asserts that both agree. A batched solver
  (`suspkin.kinematics.batch`) serves the surrogate dataset generator.

## Consequences

* Historical (measured 2026-11-17): about 3 Newton iterations per solve on the reference car
  instead of about 15 residual evaluations; results agreed with the old solver to $10^{-9}$.
* The branch-jump guard adds work near steering lock in rack sweeps, in exchange for never
  silently jumping to the other assembly branch.
* The Jacobian is hand-derived and long; `tests/unit/test_corner_jacobian.py` checks it
  against finite differences, and `tests/unit/test_fast_kernel.py` keeps the numba kernel
  honest.
* Anything that changes the solver's maths must keep the golden snapshots unchanged
  (ADR 0006).
