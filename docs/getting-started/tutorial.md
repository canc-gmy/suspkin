# Tutorial

All commands run from the repository root on the reference car
`configs/vehicles/fs_reference.yaml`. Prefix with `uv run` (or `python -m uv run`).
Command-line inputs are in mm and degrees; written files are SI.

## 1. Look at the car

```bash
suspkin show configs/vehicles/fs_reference.yaml --out car.png
suspkin show configs/vehicles/fs_reference.yaml --roll 2 --heave 5 --out pose.png
```

Without `--out` a matplotlib window opens. `--labels` annotates hardpoints, `--html` writes an
interactive plotly view.

## 2. Sweep one corner

```bash
suspkin sweep configs/vehicles/fs_reference.yaml --corner FL --travel -30 30 --metrics --out fl.csv
```

The CSV holds the solved hardpoint positions and, with `--metrics`, camber, toe, caster,
roll centre, motion ratio and their gradients against wheel travel.

## 3. Static summary

```bash
suspkin summary configs/vehicles/fs_reference.yaml
suspkin vehicle configs/vehicles/fs_reference.yaml
```

`summary` prints every static metric and gradient of the front-left and rear-left corners.
`vehicle` prints car-level numbers: ride and roll rates, roll centres, roll gradient, anti
geometry and steering.

## 4. Body-pose sweep

```bash
suspkin vsweep configs/vehicles/fs_reference.yaml --roll -2 2 --out roll.csv
```

Exactly one of `--heave`, `--roll`, `--pitch`, `--rack` is swept; the others stay at their
`--base-*` values.

## 5. Report

```bash
suspkin report configs/vehicles/fs_reference.yaml --out report/
```

Writes the standard studies as figures and a `report/report.html`. Add
`--compare other.yaml` to overlay a second design.

## 6. Python API

The same pipeline in code (this is `examples/01_load_and_sweep.py`):

```python
--8<-- "examples/01_load_and_sweep.py"
```

The core functions are pure: `load_vehicle` returns a frozen `Vehicle`; `sweep_corner`
returns solved states; `states_to_frame` and `metrics_frame` turn them into pandas tables.
Plotting reads those tables and never calls the solver. See the [Python API](../reference/api.md)
and the other [examples](../guides/examples.md).
