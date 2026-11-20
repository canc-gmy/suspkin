# ADR 0007: Kinematic surrogate model and active learning

Status: accepted (experimental). Date: 2026-11-18, amended 2026-11-19 (released in 1.1.0).

## Context

One full-order-model (FOM) design evaluation cost about 240 ms (historical, measured 2026-11-18/19, DESIGN_PLAN section 13), so a search of tens of
thousands of designs takes hours. The kinematics are smooth in the hardpoints, so a neural
network can predict the corner curves much faster. The risk is trusting a network where it is
wrong: a reported design must be the true result, not a prediction.

## Decision

**Scope and go/no-go.** The surrogate covers kinematics only (no stresses) and only symmetric
design variables (asymmetric ones are deferred). Before building it, the FOM was profiled; the
plan went ahead only if a surrogate evaluation was at least 100 times cheaper. Result: go
(the decision used the 0.75 s per design FOM profile of the time; a batched forward pass costs microseconds per design on a GPU). **Every reported design is
verified by the FOM**; the surrogate only proposes designs.

**Structure.**

* Two independent models, **front and rear**. The inputs are the axle's design variables in the
  unit cube; vehicle data are not inputs. A trained model is bound to one vehicle baseline,
  design space and output grid, checked by hashes.
* Outputs are the **left-corner curves on a fixed grid** (front: travel $\times$ rack; rear:
  travel) from a channel registry (`suspkin.surrogate.channels`), with a validity mask meaning
  "solved and finite". `toggle_margin` and the solver coordinates $\theta_{uca}$, $\theta_{lca}$,
  $\psi$ are channels; there is no ARB or wheel-centre-height channel.
* **Swing angles as line angles** (2026-11-19, replacing an earlier (cos, sin) encoding): the
  three swing-angle channels hold the angle of the line reference point $\to$ instant centre,
  wrapped to $(-\pi/2,\pi/2]$ with a 0.05 deg tolerance. A direction flips by $\pi$ when the
  instant centre changes side (parallel wishbones), a step a network cannot fit. Roll centre,
  pitch centre and anti-features need only the line. `toggle_margin` $= \min(a,\pi-a)$ has a
  kink at $a = \pi/2$; predicting $a$ and deriving the margin would be smoother and was not done.
* **Derived quantities:** rates (gains, motion ratios, wheel and ride rates) are finite
  differences of predicted curves. **Vehicle metrics** (roll centre, roll stiffness, TLLTD,
  anti-features, steering) come from an analytic, vectorised layer built on the curves
  (`suspkin.surrogate.vehicle_layer`), never from the network. Its error is measured
  separately from the network error and is far below the tolerances (for example roll-centre
  height about $10^{-13}$ mm; historical, measured 2026-11-18/19, DESIGN_PLAN section 13). Curves are interpolated with local cubic Hermite splines;
  queries outside the grid or touching a masked node give NaN, never an extrapolation.

**Data.** Latin-hypercube sampling (`random-cd` optimisation up to 5000 samples, none above),
seeded, baseline as row 0, 20 000 samples per axle by default, infeasible samples kept. Stored
as append-only `.npz` shards with a `meta.json` holding vehicle, design-space and grid hashes;
a fixed, frozen 80/10/10 train/validation/test split by a hash of (seed, row).

**Network and search.** A PyTorch MLP (extra `surrogate`) with a regression head and a
feasibility head, trained with a masked MSE plus a BCE term and optional POD output
compression. CUDA when available, CPU fallback (`device: auto`). Architecture search with
Optuna (TPE sampler, Hyperband pruner, SQLite storage); the objective is the validation RMSE
normalised by per-channel engineering tolerances, with per-channel ratios stored so one bad
channel cannot hide. Search trials are screening runs capped at `hpo.max_epochs` (default 300)
on a seeded subsample; the best configuration is retrained with the full budget, and a resume
with a different budget is refused. Fine-tuning never returns a worse model.

**Active learning (variant A).** Optimise on the networks inside a trust region around the
best FOM-verified design, run the FOM at the predicted optimum (plus optional infill), append
the FOM points to the dataset with a weight and fine-tune from the current weights. Safeguards:
a box trust region with a ratio test, and a constraint margin $g \le -\delta$ on the surrogate.
Stopping: model-FOM agreement for $p$ iterations, stagnation of the best verified $f$, the
minimum radius, no new candidates, or the budgets. Fine-tuning is gated to keep iterations
affordable: an axle is fine-tuned only if the residual of the new verified point exceeds a
trigger, is skipped after repeated no-gain fine-tunes, and is skipped by default when the
networks run on the CPU (`finetune.on_cpu: skip`); with the cheaper schedule both axles take
about 50 s worst case instead of about 315 s (historical, measured 2026-11-19). A frozen-base-plus-correction model (variant B)
is not implemented.

**Interface.** `SurrogateProblem` subclasses the optimiser `Problem` and keeps its
`evaluate_batch` interface, so every existing driver runs unchanged (ADR 0005). Commands:
`suspkin surrogate sample`, `train`, `validate` (each with `--root`; `--out` remains a hidden
alias of `--root` for `sample`) and `suspkin optimize --surrogate DIR`. Hardpoint grid snapping
(ADR 0005) applies: the network input is the snapped design re-encoded, the dataset stays
continuous, and active learning verifies snapped candidates and clamps the trust-region radius
to one grid step.

Superseded: the first plan encoded swing angles as (cos, sin) pairs and listed the
surrogate as a later (v3.x) feature, then as v0.9.0; it shipped as 1.1.0.

## Consequences

* Historical measurements (2026-11-18/19, DESIGN_PLAN section 13): a surrogate evaluation was
  about 2.5 to 3 ms per design (batch 1024; network plus constraint logic, not the forward
  pass alone) against about 240 ms for the FOM, and the vehicle layer about 0.46 ms per
  design (batch 256).
* A search can never return an unverified design, but a bad network can waste FOM
  evaluations; the trust region and the residual-triggered fine-tuning bound that waste.
* The torch/optuna dependency stays optional. The modules that need only numpy
  (`suspkin.surrogate.normalise`, `suspkin.surrogate.report`) deliberately do not import
  torch (ADR 0008).
* Asymmetric variables and a surrogate of ARB or wheel-centre quantities need new channels
  and a new dataset.
