"""Tests for the GIF/MP4 sweep animation."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from suspkin.geometry import load_vehicle
from suspkin.plotting.animation import animate_sweep
from suspkin.sweeps import sweep_vehicle

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minimal_vehicle.yaml"


def test_gif_written(tmp_path: Path) -> None:
    veh = load_vehicle(FIXTURE)
    states = sweep_vehicle(veh, roll=np.radians([-1.0, 0.0, 1.0]))
    out = animate_sweep(veh, states, tmp_path / "a.gif", fps=5)
    assert out.exists()
    assert out.read_bytes()[:3] == b"GIF"
    assert out.stat().st_size > 1000


def test_unsupported_suffix_and_empty(tmp_path: Path) -> None:
    veh = load_vehicle(FIXTURE)
    with pytest.raises(ValueError, match="suffix"):
        animate_sweep(veh, sweep_vehicle(veh, heave=[0.0]), tmp_path / "a.avi")
    with pytest.raises(ValueError, match="empty"):
        animate_sweep(veh, [], tmp_path / "a.gif")


def test_mp4_without_ffmpeg_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.animation import FFMpegWriter

    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
    veh = load_vehicle(FIXTURE)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        animate_sweep(veh, sweep_vehicle(veh, heave=[0.0]), tmp_path / "a.mp4")
