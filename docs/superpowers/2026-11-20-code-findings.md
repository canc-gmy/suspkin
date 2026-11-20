# Code findings from the 2026-11 refactor and docs rewrite

Found while documenting the code. None of these were changed, because the refactor had to keep
behaviour identical. Each one is a candidate for a separate fix, and each fix would need a
deliberate golden or behaviour-lock update.

## Likely bugs (worth fixing)

1. **Stale cache on `optimize --resume`.** `suspkin.optimize.outputs.study_hash` hashes only the
   constraints and objectives. If you change the variable domains, sweeps, resolution or vehicle
   and then resume, cached evaluations from the old setup are reused without a warning. A reviewer
   reproduced it: changing `bump_f` travel reused 22 stale rows.
2. **Resume after adding, removing, renaming or disabling a constraint or objective** raises
   `HistoryError` (exit 2), not the "warn and drop cache" path. The warn-and-drop path only triggers
   when limits, scale or weight change. `--resume` also overwrites the run folder's `study.yaml`,
   `vehicle.yaml` and `meta.json` start/end.
3. **Two variables on one hardpoint are always rejected** (`Study._cross_check`), even a left plus
   right asymmetric pair. The nearby `sides` tuple suggests the pair was meant to be allowed.
4. **ARB points ignore `right_overrides`.** `geometry.corners.arb_points` always mirrors, while
   `_drop_length` honours `right_overrides["rocker_arb"]`.
5. **Web app edited-dataset id ignores parent and label** (`_short("edited", digest, steps)` in
   `app/datasets.py`). The same edit made from another parent, or with another label, returns the
   first revision with its original parent and label.

## Inconsistencies and naming

6. The roll-stiffness display override in `suspkin.units._DISPLAY_OVERRIDE` is keyed
   `"force*length/angle"`, but `normalize_dim` returns `"length*force/angle"`, so it never matches.
   The optimiser constraint tables and `plotting.spec` show `mm*N/deg`. Label and scale are
   consistent, so this is cosmetic.
7. `wheel_center_recession` is positive forward, the opposite of what the name suggests.
8. `rc_migration_rate` (per-wheel d(rc_height)/d(travel)) and summary `rc_migration` (lateral RC
   movement in roll) look related but measure different things.
9. `tlltd_front` is a fraction in [0, 1], while the other percent metrics store the percent number.
10. The `report` Markdown rounds Ackermann to 0 %, while `summary` prints 0.359 %. The figure uses
    contact-patch spacing as the track; the summary uses `front_track`.
11. The CLI `_display` has no entry for `steer_gain` (unreached today).
12. The `surrogate sample` refusal message says "--out" although `--out` is the hidden alias of
    `--root`.
13. `metrics/vehicle.py` defines `_ACKERMANN_AT` twice; `Wheel.rim_width` is unused.

## Behaviour worth knowing

14. The `rack_travel` limit is not enforced anywhere; it only sets sweep ranges.
15. The evaluation budget is checked only at the start of a batch: `--evaluations 40` ran 44. With a
    small budget the SLSQP stage of `de+slsqp` gets nothing, and NSGA-II with a budget below
    2 × pop_size runs only its initial sample.
16. When a sweep step fails, the error message reports the last halved sub-step's travel or rack, not
    the requested value.
17. The batched surrogate kinematics (`kinematics.batch.rebuild`) does not enforce the ball-joint
    distance; `max_residual` is 0 there.
18. Hexahedron variable domains only need a positive Jacobian; convexity is checked only for
    constraint regions.
19. `POST /export` returns 404, not 500, when the single-file frontend build is missing.
20. The `perf` pytest marker is registered but CI does not deselect it.
21. `fs_reference.yaml` comment "toggle margin ~67 deg" is the +30 mm bump value; static is 85.15 deg.
22. The committed `frontend/src/api/openapi.json` says version 1.0.0 while the app reports 1.1.0;
    run `npm run gen:api` to refresh.
23. The front benchmark `model.pt` is a Git LFS pointer in a fresh worktree; run `git lfs pull`.
