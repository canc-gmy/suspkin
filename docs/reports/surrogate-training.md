# Surrogate training report (Nov 2026)

Report on the first full training of the v1.1 surrogate on the benchmark study
`configs/studies/fs_reference_surrogate.yaml`, 18-19 Nov 2026. It lists the numbers
of each run, the three problems found on the way and the open points. See the
[surrogate guide](../guides/surrogate.md) for how the workflow is used and
[the surrogate concept page](../concepts/surrogate.md) and [ADR 0007](../adr/0007-surrogate-model.md) for the design.

All scores are on the **validation split**. The test split has not been used yet.

## Summary

| item | front | rear |
|---|---|---|
| Channels | 29 | 28 |
| Search trials | 28 (10 complete, 18 pruned), stopped by the 90 min timeout | 60 (14 complete, 46 pruned), all used |
| Search wall time | 1 h 30 m 57 s | 26 m 39 s |
| Best trial score | 1.044 (trial 26) | 1.138 (trial 9) |
| Final model, first retrain (2000 epoch cap, patience 100) | 2.615 (218 epochs) | 0.969 (941 epochs) |
| **Final model, 600 epochs, patience 600** | **0.864** | **0.473** |
| Median / max channel ratio of the final model | 0.421 / 4.52 | 0.267 / 3.56 |
| Training time of the final model | 51 m 26 s | 9 m 55 s |
| Worst channels | toggle_margin, psi, coilover_length | toggle_margin, coilover_length |

Score = mean over channels of (masked validation RMSE in physical units / channel
tolerance). 1.0 means the average channel is exactly at its tolerance. The mean can hide
single bad channels, so the median, the maximum and the worst five are reported as well.

Both final models are below 1.0 on average, but `toggle_margin` and `coilover_length`
(both axles) and `psi` (front) are still 2.6 to 4.5 times their tolerance.

## Setup

| item | value |
|---|---|
| Study | `configs/studies/fs_reference_surrogate.yaml` |
| Design space | every hardpoint of both axles as an offset variable (45 parameters per axle), +-10 mm per coordinate (wheel centre +-5 mm) |
| Grid | travel -30 to 30 mm, 41 points; rack -30 to 30 mm, 21 points |
| Dataset | 20000 samples per axle, seed 1, LHS (`lhs_optimization: auto`), 100 % feasible |
| Split | train 15957 / val 2058 / test 1985 (package rule) |
| Sampling time | front 1487 s (24 m 47 s), rear 119.5 s |
| Search | `hpo`: 60 trials, timeout 90 min, trial cap 300 epochs, subsample 5000 train rows, trial patience 30 |
| Final training | `training`: `max_epochs: 600`, `patience: 600` (full cosine schedule, best weights kept) |
| Hardpoint grid | 0.5 mm absolute grid for optimised hardpoints (S16); the training dataset stays continuous |
| Hardware | one GTX 1650 (4 GB), front and rear trained one after the other in the second run |

Channels per axle are the metric tables on the sweep grid (camber, toe, caster, kpi,
scrub radius, trail, contact-patch and wheel-centre shifts, three swing-angle channels,
coilover length, rocker and pushrod angle, toggle margin) plus the solver coordinates
`theta_uca`, `theta_lca`, `psi` and the joint angles. Tolerances are defined in
`src/suspkin/surrogate/channels.py`; for example 0.05 deg for the swing angles and
`toggle_margin`, 0.05 mm for `coilover_length`, 0.01 deg for `theta_*` and `psi`.

## Timeline of runs

| when | run | outcome |
|---|---|---|
| 18 Nov | First attempt, old code (HPO trials up to 2000 epochs, patience 100, full train split, front and rear in parallel on one GPU) | Rear: 32 trials in 94 min, best 14.63. Front: first trial still running after 42 min (epoch 967). Run stopped. |
| 18 Nov | Fix 1 (S15) | Trial cap 300 epochs on 5000 rows; final retrain separate. |
| 18-19 Nov | Diagnosis of the rear plateau (CPU, read only) | Swing-angle encoding identified as the cause. |
| 19 Nov | Fix 2 (S17) and S16 | Line-angle channels; datasets regenerated, old data deleted. |
| 19 Nov 12:51-13:18 | Sampling, both axles | 2 x 20000 samples, 100 % feasible. |
| 19 Nov 13:18-14:01 | Rear search and retrain | 60 trials in 26 m 39 s; retrain 0.969 after 941 epochs (15 m 44 s). |
| 19 Nov 14:01-15:38 | Front search | 28 trials, stopped by the 90 min timeout. |
| 19 Nov 15:38-15:58 | Front retrain (2000 epoch cap) | 2.615 after 218 epochs. Worse than its trial. |
| 19 Nov 16:06-16:16 | Rear retrain, 600 epochs | 0.473. |
| 19 Nov 16:16-17:08 | Front retrain, 600 epochs | 0.864. |

The front retrain was interrupted once by a low-memory stop while the full front
dataset was loading (about 2 GB uncompressed; the background job was reaped). It was rerun
with `scripts/surrogate_retrain_best.py`, which retrains the best stored trial without
a new search.

## Problems and fixes

### 1. Trial budget (S15)

In the first attempt every trial could run up to 2000 epochs on the full train split.
On a 4 GB GPU shared by two axles this gave a plateau on the rear (seven very different
architectures between 14.6 and 14.9, best 14.63) and almost no trials on the front (the
first one was at epoch 967 after 42 min, so 1 to 2 trials fit in the 90 min budget).

Fix: a search trial is a screening run. It is capped at `hpo.max_epochs` (300) epochs on a
seeded subsample of `hpo.subsample` (5000) train rows, scored on the full validation
split, and the best configuration is retrained separately with the full budget.
A resume with a different trial budget is refused.

### 2. Swing-angle encoding (S17)

The plateau at about 15 was not an architecture effect. A read-only per-channel diagnosis
(`results/benchmark_surrogate/diagnosis/rear_channels.md`) showed that the channels
`sv_swing_angle` and `sv_wc_angle`, encoded as cos and sin of the direction from the
contact patch to the instant centre (IC), are bimodal:

- About 48 % of rear designs have the IC behind the patch. The direction flips by 180 deg
  and the cosine jumps from +1 to -1. Per sample, all-negative 46 %, all-positive 47 %,
  mixed along travel 7 %.
- The branch is set by the z coordinates of the rear wishbone pivots, so it is a step
  function of the design.
- With a tolerance of 0.001 on a quantity of unit spread, the ratio std/tol is about 1000
  for these channels. They pinned the mean score near 15.
- No bug was found in scaling, loss weights or units.

Diagnosis numbers for the rear (RMSE / tolerance of simple reference models on the
validation split, mean over channels):

| model | mean ratio |
|---|---|
| Predict the mean | 99.9 |
| Linear (ridge, 45 parameters) | 49.3 |
| Quadratic (all pairwise features) | 46.7 |
| POD with 51 modes | 6.57 |
| POD with 75 modes | 4.47 |
| POD with 124 modes | 2.39 |

Even a 124-mode POD floor is 2.39 on that encoding, and about 85 % of it came from four
swing-angle cos/sin channels plus `fv_swing_angle_cos` and `toggle_margin`.

Fix: each swing angle is one channel holding the angle of the line (reference point to
IC), wrapped to (-90, 90] deg, tolerance 0.05 deg. The vehicle layer needs only the line
(roll and pitch centre intersections, tangent for the anti-features). Channel counts
became 29 front and 28 rear. Datasets and models of the old encoding are refused.

### 3. Learning-rate schedule versus early stopping

With the new encoding the first final retrains were uneven: rear 0.969 after 941 epochs,
front 2.615 after only 218 epochs, which is worse than the front best trial (1.044).

Cause: the cosine learning-rate schedule spans `training.max_epochs` (2000). At epoch
about 200 the learning rate is still about 97 % of its initial value. Early stopping with
patience 100 fires on a noisy high-rate plateau before the rate has annealed. Search
trials (300 epoch cap) do anneal fully, which is why the retrain was worse than the trial.

Fix: `training.max_epochs: 600` and `patience: 600`, so the cosine schedule runs to the
end and the best weights are kept. Result: rear 0.473, front 0.864.

## Results per axle

### Front

| item | value |
|---|---|
| Rows train / val / test | 15957 / 2058 / 1985 |
| Trials | 28 total, 10 complete, 18 pruned |
| Search wall time | 1 h 30 m 57 s |
| Best trial | 26, score 1.0443, 300 epochs, 9 m 13 s |

Best parameters:

| param | value |
|---|---|
| activation | silu |
| n_layers | 7 |
| width | 639 |
| residual | True |
| layernorm | False |
| dropout | 0.0331 |
| lr | 0.001284 |
| weight_decay | 0.000865 |
| batch_size | 134 |
| lam_feas | 0.6374 |
| output | raw |

Top completed trials:

| trial | score | n_layers | width | activation | residual | worst channel (ratio) |
|---|---|---|---|---|---|---|
| 26 | 1.044 | 7 | 639 | silu | True | psi (5.08) |
| 15 | 1.174 | 7 | 823 | silu | True | psi (5.97) |
| 13 | 1.263 | 5 | 831 | silu | True | psi (6.39) |
| 17 | 1.378 | 8 | 731 | silu | True | psi (8.04) |
| 9 | 1.958 | 6 | 943 | silu | True | psi (9.36) |

The search favours deep, wide residual networks with `silu`, no layer norm and low
dropout. The four worst complete trials are small or non-residual networks (scores 2.3 to
26.5).

Final model (600 epochs, 51 m 26 s): mean 0.864, median 0.421, max 4.52.

| worst channel | ratio |
|---|---|
| toggle_margin | 4.521 |
| psi | 4.134 |
| coilover_length | 2.613 |
| toe | 2.051 |
| steer_angle | 2.051 |

### Rear

| item | value |
|---|---|
| Rows train / val / test | 15957 / 2058 / 1985 |
| Trials | 60 total, 14 complete, 46 pruned |
| Search wall time | 26 m 39 s |
| Best trial | 9, score 1.1377, 300 epochs, 1 m 44 s |

Best parameters:

| param | value |
|---|---|
| activation | silu |
| n_layers | 6 |
| width | 943 |
| residual | True |
| layernorm | False |
| dropout | 0.0973 |
| lr | 0.000786 |
| weight_decay | 5.09e-05 |
| batch_size | 308 |
| lam_feas | 0.0576 |
| output | raw |

Top completed trials:

| trial | score | n_layers | width | activation | residual | worst channel (ratio) |
|---|---|---|---|---|---|---|
| 9 | 1.138 | 6 | 943 | silu | True | toggle_margin (8.85) |
| 49 | 1.240 | 3 | 297 | silu | False | toggle_margin (8.28) |
| 45 | 1.269 | 2 | 324 | gelu | False | toggle_margin (8.32) |
| 33 | 1.373 | 3 | 316 | gelu | False | coilover_length (7.87) |
| 44 | 1.505 | 2 | 293 | gelu | False | toggle_margin (10.9) |

Final model (600 epochs, 9 m 55 s): mean 0.4726, median 0.267, max 3.56.

| worst channel | ratio |
|---|---|
| toggle_margin | 3.558 |
| coilover_length | 2.732 |
| rocker_angle | 0.704 |
| pushrod_angle | 0.666 |
| scrub_radius | 0.605 |

### Score progression

| step | front | rear |
|---|---|---|
| Old encoding, 2000 epoch trials, full train split | not completed (1 to 2 trials possible) | 14.63 (plateau, 14.6 to 14.9) |
| New encoding, best search trial (300 epochs, 5000 rows) | 1.044 | 1.138 |
| Final retrain, 2000 epoch cap, patience 100 | 2.615 | 0.969 |
| Final retrain, 600 epochs, patience 600 | **0.864** | **0.473** |
| Same, median channel ratio | 0.421 | 0.267 |
| Same, max channel ratio | 4.52 (toggle_margin) | 3.56 (toggle_margin) |

The final models are better than their best trials because they train on the full train
split (15957 rows against 5000 in a trial).

## Hardware and timings

| step | front | rear |
|---|---|---|
| Sampling 20000 designs | 24 m 47 s | 2 m 0 s |
| Search | 1 h 30 m 57 s (28 trials) | 26 m 39 s (60 trials) |
| Single trial | 3 to 9 m | 0.8 to 2 m |
| Final retrain, 2000 epoch cap | 18 m 54 s (218 epochs) | 15 m 44 s (941 epochs) |
| Final retrain, 600 epochs | 51 m 26 s | 9 m 55 s |

Hardware: one GTX 1650 (4 GB). With one trial at a time the GPU is only 30 to 50 %
utilised. The front dataset is much larger than the rear one (about 101 kB per sample
against 5 kB, 2.0 GB against 0.1 GB uncompressed) and the front networks are bigger.

## Open points and next steps

1. **toggle_margin** is `min(a, pi - a)`, which has a kink at a = 90 deg (the best-margin
   point) and is the worst channel on both axles (4.52 front, 3.56 rear). Proposal:
   predict the angle `a` and derive the margin in the evaluator (noted in S17, not done).
2. **coilover_length** has a tolerance of 0.05 mm. The ratio of 2.6 to 2.7 corresponds to
   about 0.13 mm, which may be stricter than the optimiser needs; the tolerance could be
   relaxed.
3. **psi on the front** (tolerance 0.01 deg, ratio 4.13) is the second worst front
   channel and the worst channel in all good front trials.
4. **GPU utilisation** is 30 to 50 % with one trial at a time. Running several trials in
   parallel per axle should shorten the front search, which hit the timeout with 28 of 60
   trials.
5. **Not done yet**: validation on the test split (`suspkin surrogate validate`) and the
   active-learning benchmark against direct differential evolution
   (`scripts/benchmark_surrogate.py`).

## Reproduce

All commands run from the repository root. Paths follow the benchmark layout.

```bash
# dataset (both axles), then search + final training, then validation
python -m uv run suspkin surrogate sample configs/studies/fs_reference_surrogate.yaml --root results/benchmark_surrogate/surrogate --workers auto
python -m uv run suspkin surrogate train configs/studies/fs_reference_surrogate.yaml --root results/benchmark_surrogate/surrogate --device cuda
python -m uv run suspkin surrogate validate configs/studies/fs_reference_surrogate.yaml --root results/benchmark_surrogate/surrogate

# retrain the best stored trial without a new search (one axle)
python -m uv run python scripts/surrogate_retrain_best.py configs/studies/fs_reference_surrogate.yaml --root results/benchmark_surrogate/surrogate --axle rear --device cuda

# rebuild the HPO report from the stored studies
python -m uv run python scripts/surrogate_hpo_report.py --root results/benchmark_surrogate/surrogate

# whole benchmark (sample, train, validate, active learning vs direct DE)
python -m uv run python scripts/benchmark_surrogate.py --device cuda --workers auto
```

Sources of the numbers: `results/benchmark_surrogate/surrogate/models/hpo_report.md`
(current), `hpo_report_2000ep.md` (previous final models), `models/{front,rear}/hpo_result.json`,
`results/benchmark_surrogate/diagnosis/rear_channels.md` and `results/benchmark_surrogate/train_log.txt`.
The first attempt of 18 Nov (old code) is stated from the session notes; its logs were
not kept.
