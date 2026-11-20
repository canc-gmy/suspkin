# suspkin (stable API)

The names in `suspkin.__all__` are the stable public API. They are imported lazily, so `import suspkin` is cheap. Background: [Architecture](../concepts/architecture.md).

```python
import suspkin as sk

vehicle = sk.load_vehicle("configs/vehicles/fs_reference.yaml")
```

`suspkin.__version__` holds the installed package version.

::: suspkin.Vehicle

::: suspkin.VehicleConfigError

::: suspkin.load_vehicle

::: suspkin.save_vehicle

::: suspkin.build_corners

::: suspkin.solve_corner

::: suspkin.sweep_corner

::: suspkin.sweep_rack

::: suspkin.solve_vehicle

::: suspkin.KinematicsError

::: suspkin.sweep_vehicle

::: suspkin.vehicle_frame

::: suspkin.states_to_frame

::: suspkin.metrics_frame

::: suspkin.add_gradients

::: suspkin.vehicle_summary

::: suspkin.build_report

::: suspkin.run_standard_studies

::: suspkin.plot_vehicle_3d
