"""Surrogate block of a study (``surrogate:`` in the study YAML)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    Field,
    StrictBool,
    model_validator,
)

from suspkin._spec_base import _Frozen, _scalar


class AxisGrid(_Frozen):
    """Uniform 1-D grid ``[min, max]`` with ``points`` nodes (SI), containing 0 (static pose)."""

    min: float
    max: float
    points: int = Field(ge=3)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        out = dict(raw)
        for k in ("min", "max"):
            if k in out:
                out[k] = _scalar(out[k], "length")
        return out

    @model_validator(mode="after")
    def _check(self) -> AxisGrid:
        if not self.min < self.max:
            raise ValueError("grid min must be below max")
        if not self.min <= 0.0 <= self.max:
            raise ValueError("grid must contain 0 (the static pose)")
        return self


class GridSpec(_Frozen):
    """Output grid of the surrogate: wheel travel and (front axle only) rack."""

    travel: AxisGrid
    rack: AxisGrid | None = None


class DatasetSpec(_Frozen):
    """Initial Latin-hypercube dataset: samples per axle, seed and rows per shard file."""

    n_initial: int = Field(default=20000, ge=1)
    seed: int = 1
    shard_size: int = Field(default=500, ge=1)
    lhs_optimization: Literal["auto", "random-cd", "none"] = "auto"


class TrainingSpec(_Frozen):
    """Network training settings (the ``surrogate.training`` block)."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_epochs: int = Field(default=2000, ge=1)
    patience: int = Field(default=100, ge=1)
    num_threads: int | None = Field(default=None, ge=1)
    finetune_lr_factor: float = Field(default=0.1, gt=0.0)


class HpoSpec(_Frozen):
    """Architecture search budget (the ``surrogate.hpo`` block); ``timeout`` in seconds (SI).

    ``max_epochs`` caps each search trial (``None``: ``training.max_epochs``; the effective cap
    is the smaller of the two) and ``subsample`` is the number of training rows each trial
    uses (``None``: all). The final retrain of the best configuration ignores both.
    """

    n_trials: int = Field(default=100, ge=1)
    timeout: float | None = Field(default=4 * 3600.0, gt=0.0)
    parallel: Literal["auto"] | StrictBool = "auto"
    max_epochs: int | None = Field(default=300, ge=1)
    subsample: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping) and raw.get("timeout") is not None:
            out = dict(raw)
            try:
                out["timeout"] = _scalar(out["timeout"], "time")
            except ValueError as exc:
                raise ValueError(f"hpo timeout: {exc}") from exc
            return out
        return raw


class InfillSpec(_Frozen):
    """Infill points per active-learning iteration: extra candidates and perturbations."""

    top: int = Field(default=0, ge=0)
    perturb: int = Field(default=4, ge=0)


class FinetuneSpec(_Frozen):
    """Fine-tuning of the axle networks between active-learning iterations.

    ``epochs`` caps one fine-tune, ``patience`` stops it after that many epochs without a
    new best score of the run. ``weight`` is the sample weight of the new rows, ``replay``
    the number of earlier train rows replayed (drawn once, then kept on the device).
    ``trigger`` is the residual (in tolerance units) of the new verified point above which an
    axle is fine-tuned at all. After ``skip_after`` consecutive fine-tunes that kept the
    previous weights an axle is skipped for ``skip_for`` iterations, unless its residual
    exceeds ``stall_override * trigger``. ``trigger=0`` and ``skip_for=0`` always fine-tune.
    ``on_cpu`` decides what happens when the networks run on the CPU: ``"skip"`` (default)
    never fine-tunes (it takes minutes per iteration), ``"run"`` applies the rules above.
    """

    epochs: int = Field(default=60, ge=1)
    weight: float = Field(default=5.0, gt=0.0)
    replay: int = Field(default=2048, ge=0)
    patience: int = Field(default=30, ge=1)
    trigger: float = Field(default=1.0, ge=0.0)
    skip_after: int = Field(default=2, ge=1)
    skip_for: int = Field(default=2, ge=0)
    stall_override: float = Field(default=3.0, ge=1.0)
    on_cpu: Literal["skip", "run"] = "skip"


class TrustRegionSpec(_Frozen):
    """Trust-region radius in the unit cube: initial, minimum and maximum."""

    initial: float = Field(default=0.2, gt=0.0, le=1.0)
    min: float = Field(default=0.01, gt=0.0, le=1.0)
    max: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> TrustRegionSpec:
        if not self.min <= self.initial <= self.max:
            raise ValueError("trust_region needs min <= initial <= max")
        return self


class StopSpec(_Frozen):
    """Active-learning stopping: relative agreement ``eps_f`` and ``patience`` iterations."""

    eps_f: float = Field(default=0.01, gt=0.0)
    patience: int = Field(default=5, ge=1)


class ActiveLearningSpec(_Frozen):
    """Active-learning optimizer (``surrogate.active_learning``); ``time`` in seconds (SI)."""

    max_iterations: int = Field(default=50, ge=1)
    fom_evaluations: int = Field(default=2000, ge=1)
    time: float | None = Field(default=None, gt=0.0)
    surrogate_evaluations: int = Field(default=20000, ge=1)
    infill: InfillSpec = Field(default_factory=InfillSpec)
    finetune: FinetuneSpec = Field(default_factory=FinetuneSpec)
    margin: float = Field(default=0.05, ge=0.0)
    trust_region: TrustRegionSpec = Field(default_factory=TrustRegionSpec)
    stop: StopSpec = Field(default_factory=StopSpec)

    @model_validator(mode="before")
    @classmethod
    def _units(cls, raw: Any) -> Any:
        if isinstance(raw, Mapping) and raw.get("time") is not None:
            out = dict(raw)
            try:
                out["time"] = _scalar(out["time"], "time")
            except ValueError as exc:
                raise ValueError(f"active_learning time: {exc}") from exc
            return out
        return raw


class SurrogateSpec(_Frozen):
    """Surrogate block of a study: grid, dataset, training, search and active learning."""

    grid: GridSpec
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    hpo: HpoSpec = Field(default_factory=HpoSpec)
    active_learning: ActiveLearningSpec = Field(default_factory=ActiveLearningSpec)
