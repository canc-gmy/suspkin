# Examples

Scripts in `examples/`. Run from the repository root, e.g.
`uv run python examples/01_load_and_sweep.py`. Every script takes `--out DIR`
(default `results/examples/<script name>`).

## 01 Load and sweep

Load the reference car, sweep the front-left corner in bump/droop, print key metrics.

```python
--8<-- "examples/01_load_and_sweep.py"
```

## 02 Vehicle roll

Whole-car roll sweep, vehicle summary, camber versus roll plot.

```python
--8<-- "examples/02_vehicle_roll.py"
```

## 03 Compare designs

Modify a hardpoint, compare two designs and write a comparison report.

```python
--8<-- "examples/03_compare_designs.py"
```

## 04 Custom plot

3D view at a solved pose and a custom matplotlib curve from a result table.

```python
--8<-- "examples/04_custom_plot.py"
```

## 05 Optimize a study (experimental)

Constraint table of a study baseline and a tiny optimisation run.

```python
--8<-- "examples/05_optimize_study.py"
```
