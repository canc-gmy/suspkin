# suspkin examples

Five runnable scripts that show the Python API on the reference car
(`configs/vehicles/fs_reference.yaml`). Run them from the repository root:

```bash
python -m uv run python examples/01_load_and_sweep.py          # or: uv run python ...
python -m uv run python examples/01_load_and_sweep.py --out my_results/01
```

Every script takes `--out DIR` (default `results/examples/<script name>`, created if missing)
and writes there. They need only the core install; 05 is experimental.

| Script | What it shows | Writes | Key output (expected) |
|---|---|---|---|
| `01_load_and_sweep.py` | load a vehicle, FL bump sweep, metrics table, static values | `fl_bump_sweep.csv` | static camber -1.000 deg, RC height 33.7 mm, camber gain -0.0175 deg/mm |
| `02_vehicle_roll.py` | roll sweep, vehicle summary, camber vs roll plot | `camber_vs_roll.png` | roll gradient 0.629 deg/g, TLLTD 50.3 % |
| `03_compare_designs.py` | modify a hardpoint, compare two designs, write a report | `report.html`, `report.md`, `figures/`, ... | base RC 33.7 mm, with front LCA +10 mm 59.0 mm |
| `04_custom_plot.py` | 3D view at a solved pose and a custom curve from a table | `vehicle_3d.png`, `camber_curve.png` | two "saved ..." lines |
| `05_optimize_study.py` | constraint table of a study baseline and a tiny DE run (experimental) | run folder with `best.yaml`, `history.csv` | baseline f_total 11.3504, best 11.0953 after 44 evaluations |

The whole set takes about 30 seconds. Full descriptions, the source of each script and the
complete expected output are in the docs: `docs/guides/examples.md`
(published as "Examples" under Guides). CI runs every script
(`pytest -m examples`), so they are always in a working state.
