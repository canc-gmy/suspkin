# suspkin examples

Run from the repository root, e.g. `python -m uv run python examples/01_load_and_sweep.py`.
Every script takes `--out DIR` (default `results/examples/<script name>`).

- `01_load_and_sweep.py`: load the reference car, FL bump sweep, metrics table, static values, CSV.
- `02_vehicle_roll.py`: roll sweep, vehicle summary (roll gradient, roll centres, TLLTD), camber vs roll plot.
- `03_compare_designs.py`: modify a hardpoint, compare two designs and write a comparison report.
- `04_custom_plot.py`: 3D view at a solved pose and a custom matplotlib curve from a result table.
- `05_optimize_study.py` (experimental): constraint table of a study baseline and a tiny optimisation run.
