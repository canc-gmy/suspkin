"""Study specification: design variables, sweeps and constraints (YAML, schema 1).

Frozen pydantic models (``extra="forbid"``). Values in the YAML may carry units
(``"30 mm"``, ``"-1.5 deg"``, ``"0.5 deg/deg"``); everything is converted to SI on load and
the unit dimension is checked against the metric registry. Regions are in the vehicle frame
on the left side (y > 0); for an asymmetric right point they are on the right side.

Hexahedron vertex order: bottom face 0-1-2-3 counter-clockwise seen from +z, then the top
face 4-5-6-7 above 0-1-2-3 (see :mod:`suspkin.optimize.regions`).

Submodules: ``variables`` (regions, variables, sweeps), ``constraints``, ``objectives``
(objectives, budget, optimizer) and ``study`` (the :class:`Study` model and the loader). The
surrogate block is in ``surrogate``. Only the common names are re-exported here.
"""

from __future__ import annotations

from suspkin.optimize.spec.objectives import Optimizer
from suspkin.optimize.spec.study import Study, check_against_vehicle, load_study
from suspkin.optimize.spec.variables import StudyConfigError

__all__ = ["Optimizer", "Study", "StudyConfigError", "check_against_vehicle", "load_study"]
