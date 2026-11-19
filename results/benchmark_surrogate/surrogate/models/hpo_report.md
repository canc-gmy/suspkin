# Surrogate architecture search report

## Front axle

### Dataset and budget

| item | value |
|---|---|
| Rows train / val / test | 15957 / 2058 / 1985 |
| Rows per search trial (train) | 5000 |
| Device | cuda |
| Trials requested | n/a |
| Timeout | none / n/a |
| Trial epoch cap | 300 |
| Trial patience | 30 |

Score = mean over channels of (masked RMSE in physical units / channel tolerance) on the complete validation split; lower is better, 1.0 means every channel is at its tolerance. The mean can hide a single bad channel: see the median / max channel ratio and the worst channels of the final model.

### Trials

- Trial counts by state: COMPLETE: 10, PRUNED: 18 (total 28).
- Search wall time (first start to last finish in the database): 1h 30m 57s.
- Final model: **retrain only** (best trial retrained, no new trials run).

### Best trial

Trial 26, score 1.0443.

| param | value |
|---|---|
| activation | silu |
| batch_size | 134 |
| dropout | 0.03312 |
| lam_feas | 0.6374 |
| layernorm | False |
| lr | 0.001284 |
| n_layers | 7 |
| output | raw |
| residual | True |
| weight_decay | 0.0008653 |
| width | 639 |

### Top 10 completed trials

| trial | score | epochs | duration | worst channel | n_layers | width | activation | residual | layernorm | dropout | lr | batch_size | pod_k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26 | 1.0443 | 300 | 9m 13s | psi (5.08) | 7 | 639 | silu | True | False | 0.03312 | 0.001284 | 134 |  |
| 15 | 1.1744 | 300 | 8m 24s | psi (5.97) | 7 | 823 | silu | True | False | 0.04559 | 0.001387 | 205 |  |
| 13 | 1.2627 | 300 | 7m 10s | psi (6.39) | 5 | 831 | silu | True | False | 0.06357 | 0.001643 | 254 |  |
| 17 | 1.3775 | 300 | 8m 30s | psi (8.04) | 8 | 731 | silu | True | False | 0.00283 | 0.001377 | 255 |  |
| 9 | 1.9580 | 300 | 7m 42s | psi (9.36) | 6 | 943 | silu | True | False | 0.09727 | 0.0007864 | 308 |  |
| 11 | 1.9658 | 300 | 9m 05s | psi (10) | 6 | 786 | silu | True | False | 0.09502 | 0.0004719 | 161 |  |
| 2 | 2.3461 | 300 | 4m 59s | psi (11.3) | 4 | 428 | tanh | True | True | 0.08958 | 0.002198 | 283 |  |
| 1 | 5.0497 | 300 | 3m 07s | psi (24) | 2 | 110 | gelu | False | True | 0.03397 | 0.001982 | 411 |  |
| 3 | 18.5081 | 266 | 3m 01s | psi (105) | 3 | 133 | relu | False | False | 0.1389 | 0.0004091 | 565 |  |
| 0 | 26.5264 | 300 | 3m 18s | coilover_length (160) | 4 | 471 | gelu | False | False | 0.08384 | 0.001028 | 1460 | 75 |

### Pruning

18 pruned trial(s) (number@epochs reached): #4@4, #5@10, #6@10, #7@4, #8@82, #10@10, #12@10, #14@10, #16@28, #18@28, #19@4, #20@10, #21@28, #22@82, #23@4, #24@28, #25@82, #27@244.

### Final model

- Validation score: 0.8639.
- Epochs trained: 600.
- Training time: 51m 26s.
- Trained on the full train split with the full epoch cap and patience.
- Channel ratios (RMSE / tolerance): mean 0.8639, median 0.4207, max 4.521.

Worst 5 channels:

| channel | ratio |
|---|---|
| toggle_margin | 4.521 |
| psi | 4.134 |
| coilover_length | 2.613 |
| toe | 2.051 |
| steer_angle | 2.051 |

## Rear axle

### Dataset and budget

| item | value |
|---|---|
| Rows train / val / test | 15957 / 2058 / 1985 |
| Rows per search trial (train) | 5000 |
| Device | cuda |
| Trials requested | 60 |
| Timeout | 1h 30m 00s |
| Trial epoch cap | 300 |
| Trial patience | 30 |

Score = mean over channels of (masked RMSE in physical units / channel tolerance) on the complete validation split; lower is better, 1.0 means every channel is at its tolerance. The mean can hide a single bad channel: see the median / max channel ratio and the worst channels of the final model.

### Trials

- Trial counts by state: COMPLETE: 14, PRUNED: 46 (total 60).
- Search wall time (first start to last finish in the database): 26m 39s.
- Final model: **retrain only** (best trial retrained, no new trials run).

### Best trial

Trial 9, score 1.1377.

| param | value |
|---|---|
| activation | silu |
| batch_size | 308 |
| dropout | 0.09727 |
| lam_feas | 0.05758 |
| layernorm | False |
| lr | 0.0007864 |
| n_layers | 6 |
| output | raw |
| residual | True |
| weight_decay | 5.088e-05 |
| width | 943 |

### Top 10 completed trials

| trial | score | epochs | duration | worst channel | n_layers | width | activation | residual | layernorm | dropout | lr | batch_size | pod_k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 1.1377 | 300 | 1m 44s | toggle_margin (8.85) | 6 | 943 | silu | True | False | 0.09727 | 0.0007864 | 308 |  |
| 49 | 1.2395 | 300 | 1m 53s | toggle_margin (8.28) | 3 | 297 | silu | False | True | 0.01322 | 0.001771 | 187 |  |
| 45 | 1.2694 | 300 | 1m 33s | toggle_margin (8.32) | 2 | 324 | gelu | False | True | 0.007685 | 0.002355 | 175 |  |
| 33 | 1.3726 | 300 | 1m 22s | coilover_length (7.87) | 3 | 316 | gelu | False | True | 0.001754 | 0.002021 | 274 |  |
| 44 | 1.5046 | 300 | 1m 13s | toggle_margin (10.9) | 2 | 293 | gelu | False | True | 0.006659 | 0.002767 | 228 |  |
| 22 | 1.5053 | 300 | 1m 35s | coilover_length (8.93) | 4 | 242 | gelu | False | True | 0.001162 | 0.002966 | 256 |  |
| 2 | 1.5740 | 300 | 1m 22s | toggle_margin (8.78) | 4 | 428 | tanh | True | True | 0.08958 | 0.002198 | 283 |  |
| 36 | 2.2454 | 300 | 1m 28s | coilover_length (14.2) | 3 | 302 | gelu | False | True | 0.07277 | 0.002669 | 244 |  |
| 1 | 3.0447 | 300 | 0m 48s | toggle_margin (22.4) | 2 | 110 | gelu | False | True | 0.03397 | 0.001982 | 411 |  |
| 25 | 3.0566 | 300 | 1m 25s | coilover_length (17.8) | 6 | 398 | gelu | False | True | 0.05248 | 0.00194 | 339 |  |

### Pruning

46 pruned trial(s) (number@epochs reached): #3@10, #7@28, #8@28, #10@4, #11@10, #12@4, #13@4, #14@4, #15@82, #16@4, #17@10, #18@10, #19@10, #20@4, #21@4, #23@10, #24@28, #26@4, #27@28, #28@10, #29@28, #30@4, #31@28, #32@4, #34@10, #35@4, #37@244, #38@4, #39@10, #40@10, #41@82, #42@10, #43@4, #46@82, #47@10, #48@4, #50@10, #51@82, #52@244, #53@10, #54@244, #55@82, #56@10, #57@10, #58@4, #59@244.

### Final model

- Validation score: 0.4726.
- Epochs trained: 600.
- Training time: 9m 55s.
- Trained on the full train split with the full epoch cap and patience.
- Channel ratios (RMSE / tolerance): mean 0.4726, median 0.2674, max 3.558.

Worst 5 channels:

| channel | ratio |
|---|---|
| toggle_margin | 3.558 |
| coilover_length | 2.732 |
| rocker_angle | 0.7041 |
| pushrod_angle | 0.6657 |
| scrub_radius | 0.6051 |
