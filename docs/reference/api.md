# Python API

The stable public API is the set of names in `suspkin.__all__`. They are imported lazily, so
`import suspkin` is cheap. Typical use:

```python
import suspkin as sk

vehicle = sk.load_vehicle("configs/vehicles/fs_reference.yaml")
```

The reference below documents each name at its defining module.

## Geometry

::: suspkin.geometry.models.Vehicle

::: suspkin.geometry.io
    options:
      members:
        - VehicleConfigError
        - load_vehicle
        - save_vehicle

::: suspkin.geometry.corners.build_corners

## Kinematics

::: suspkin.kinematics.corner
    options:
      members:
        - solve_corner
        - sweep_corner
        - sweep_rack

::: suspkin.kinematics.vehicle.solve_vehicle

::: suspkin.kinematics.rocker.KinematicsError

## Sweeps and tables

::: suspkin.sweeps.vehicle_sweeps
    options:
      members:
        - sweep_vehicle
        - vehicle_frame

::: suspkin.sweeps.corner_sweeps.states_to_frame

## Metrics

::: suspkin.metrics.table
    options:
      members:
        - metrics_frame
        - add_gradients

::: suspkin.metrics.vehicle_summary.vehicle_summary

## Reports and plots

::: suspkin.report.builder.build_report

::: suspkin.report.studies.run_standard_studies

::: suspkin.plotting.geometry3d.plot_vehicle_3d
