"""Hardpoint edits: apply a list of point moves to a vehicle (pure, no I/O).

Same mechanism as ``suspkin.optimize.design``: dump the vehicle, change the left
``hardpoints`` and the ``right_overrides``, validate again with ``Vehicle.model_validate``.

Sides
-----
``both``
    Moves the left point; the right side mirrors it. Any right override of that point is
    dropped (the edit says both sides follow).
``left``
    Moves the left point only. An existing right override is kept, so the right side does
    not move. Without an override the right side mirrors the new left point.
``right``
    Writes ``right_overrides`` only, in right-side coordinates (``y <= 0``). Missing
    coordinates keep the current right value (the override, or the mirrored left point).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from suspkin.app.models import HardpointEdit
from suspkin.geometry import Vehicle

__all__ = ["HardpointEditError", "apply_hardpoint_edits"]


class HardpointEditError(ValueError):
    """An edit that names an unknown point or produces an invalid vehicle."""


def _merge(cur: Sequence[float], edit: HardpointEdit) -> tuple[float, float, float]:
    """``cur`` with the coordinates given in ``edit`` replaced."""
    x, y, z = (float(c) for c in cur)
    return (
        float(edit.x) if edit.x is not None else x,
        float(edit.y) if edit.y is not None else y,
        float(edit.z) if edit.z is not None else z,
    )


def _ref(e: HardpointEdit) -> str:
    return f"{e.axle}.{e.point} ({e.side})"


def apply_hardpoint_edits(vehicle: Vehicle, edits: Sequence[HardpointEdit]) -> Vehicle:
    """Return a new vehicle with ``edits`` applied in order (SI, vehicle frame).

    Args:
        vehicle: Source vehicle (not modified).
        edits: Hardpoint coordinate overrides in m, applied in order.

    Raises:
        HardpointEditError: Unknown point, an edit without coordinates, or a geometry that fails
            validation. The message names the edited points.
    """
    if not edits:
        raise HardpointEditError("no edits given")
    d: dict[str, Any] = vehicle.model_dump()
    for e in edits:
        if e.x is None and e.y is None and e.z is None:
            raise HardpointEditError(f"{_ref(e)}: no coordinate to change")
        ax = d[e.axle]
        left = ax["hardpoints"].get(e.point)
        if left is None:
            raise HardpointEditError(
                f"{e.axle}.{e.point} does not exist in this vehicle (points cannot be added)"
            )
        overrides: dict[str, Any] = ax["right_overrides"]
        if e.side == "right":
            cur = overrides.get(e.point) or (left[0], -left[1], left[2])
            overrides[e.point] = _merge(cur, e)
        else:
            ax["hardpoints"][e.point] = _merge(left, e)
            if e.side == "both":
                overrides.pop(e.point, None)
    try:
        return Vehicle.model_validate(d)
    except ValidationError as exc:
        names = ", ".join(sorted({f"{e.axle}.{e.point}" for e in edits}))
        reasons = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:3]
        )
        raise HardpointEditError(f"invalid geometry after editing {names}: {reasons}") from exc
