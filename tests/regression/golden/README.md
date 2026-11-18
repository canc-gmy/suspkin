# Golden snapshots

Frozen outputs of the reference car (`configs/vehicles/fs_reference.yaml`) and its two
reference studies, all SI (m, rad, N, N/m):

- `corner_{FL,FR,RL,RR}.csv`: travel sweep -30..30 mm, 25 steps, metrics + gradients.
- `rack_FL.csv`: rack sweep at zero travel (90 % of the rack limits), 21 steps.
- `vehicle_{heave,roll,pitch,rack}.csv`: vehicle sweeps, 11 steps each.
- `vehicle_summary.json`: flattened `vehicle_summary` (None is null).
- `study_constraints.csv`, `study_optimize.csv`, `study_objectives.csv`: baseline `evaluate`.

Tests: `tests/regression/test_golden.py` (rtol 1e-6, atol 1e-9; atol 1e-8 for gradient columns).

Regenerate only after a deliberate, reviewed model change:

    python -m uv run pytest tests/regression --update-golden

Review the diff of these files in the same commit and mention the change in `CHANGELOG.md`.
A failing comparison without such a change is a regression, not a reason to regenerate.
