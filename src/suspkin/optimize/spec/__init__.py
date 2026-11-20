"""Study specification: design variables, sweeps and constraints (YAML, schema 1).

Frozen pydantic models (``extra="forbid"``). Values in the YAML may carry units
(``"30 mm"``, ``"-1.5 deg"``, ``"0.5 deg/deg"``); everything is converted to SI on load and
the unit dimension is checked against the metric registry. Regions are in the vehicle frame
on the left side (y > 0); for an asymmetric right point they are on the right side.

Hexahedron vertex order: bottom face 0-1-2-3 counter-clockwise seen from +z, then the top
face 4-5-6-7 above 0-1-2-3 (see :mod:`suspkin.optimize.regions`).

Submodules: ``variables`` (regions, variables, sweeps), ``constraints``, ``objectives``
(objectives, budget, optimizer) and ``study`` (the :class:`Study` model and the loader). The
surrogate block lives in :mod:`suspkin.surrogate.spec` and is re-exported here.
"""

from __future__ import annotations

from suspkin.optimize.spec.constraints import (
    BandConstraint,
    Constraint,
    DistanceConstraint,
    InsideConstraint,
    JointAngleConstraint,
    OutsideConstraint,
    ReachConstraint,
    RelationConstraint,
    SlopeConstraint,
    SummaryConstraint,
    ValueConstraint,
)
from suspkin.optimize.spec.objectives import (
    Budget,
    CurveObjective,
    CurveTarget,
    ExtremumObjective,
    LinearityObjective,
    Objective,
    Optimizer,
    TargetObjective,
    objective_dim,
    objective_scale,
)
from suspkin.optimize.spec.study import (
    Study,
    check_against_vehicle,
    check_grid,
    load_study,
    region_centroid,
    variable_resolution,
)
from suspkin.optimize.spec.variables import (
    DEFAULT_RESOLUTION,
    JOINT_NAMES,
    POINT_NAMES,
    BoxRegion,
    CornerSweep,
    CylinderRegion,
    Domain,
    HexahedronRegion,
    OffsetRegion,
    Pair,
    Region,
    SphereRegion,
    StudyConfigError,
    Sweep,
    Variable,
    Vec3,
    VehicleSweep,
    default_scale,
    parse_coord_ref,
    parse_point_ref,
    sweep_input_dim,
    sweep_metric_dim,
    sweep_range,
)
from suspkin.surrogate.spec import (
    ActiveLearningSpec,
    AxisGrid,
    DatasetSpec,
    FinetuneSpec,
    GridSpec,
    HpoSpec,
    InfillSpec,
    StopSpec,
    SurrogateSpec,
    TrainingSpec,
    TrustRegionSpec,
)

__all__ = [
    "DEFAULT_RESOLUTION",
    "JOINT_NAMES",
    "POINT_NAMES",
    "ActiveLearningSpec",
    "AxisGrid",
    "BandConstraint",
    "BoxRegion",
    "Budget",
    "Constraint",
    "CornerSweep",
    "CurveObjective",
    "CurveTarget",
    "CylinderRegion",
    "DatasetSpec",
    "DistanceConstraint",
    "Domain",
    "ExtremumObjective",
    "FinetuneSpec",
    "GridSpec",
    "HexahedronRegion",
    "HpoSpec",
    "InfillSpec",
    "InsideConstraint",
    "JointAngleConstraint",
    "LinearityObjective",
    "Objective",
    "OffsetRegion",
    "Optimizer",
    "OutsideConstraint",
    "Pair",
    "ReachConstraint",
    "Region",
    "RelationConstraint",
    "SlopeConstraint",
    "SphereRegion",
    "StopSpec",
    "Study",
    "StudyConfigError",
    "SummaryConstraint",
    "SurrogateSpec",
    "Sweep",
    "TargetObjective",
    "TrainingSpec",
    "TrustRegionSpec",
    "ValueConstraint",
    "Variable",
    "Vec3",
    "VehicleSweep",
    "check_against_vehicle",
    "check_grid",
    "default_scale",
    "load_study",
    "objective_dim",
    "objective_scale",
    "parse_coord_ref",
    "parse_point_ref",
    "region_centroid",
    "sweep_input_dim",
    "sweep_metric_dim",
    "sweep_range",
    "variable_resolution",
]
