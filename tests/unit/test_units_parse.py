"""Quantity parsing, dimensions and display formatting."""

import math

import pytest

from suspkin.units import (
    display_unit,
    normalize_dim,
    parse_quantity,
    slope_dim,
)


@pytest.mark.parametrize(
    ("text", "dim", "expected"),
    [
        ("30 mm", "length", 0.03),
        ("2 cm", "length", 0.02),
        ("-1.5 deg", "angle", math.radians(-1.5)),
        ("0.5 deg/deg", "ratio", 0.5),
        ("2 mm/mm", "ratio", 2.0),
        ("1.2 deg/g", "angle/gload", math.radians(1.2)),
        ("34 N/mm", "stiffness", 34000.0),
        ("34 N/mm", "force/length", 34000.0),
        ("50 %", "percent", 50.0),
        ("0.1 deg/mm", "angle/length", math.radians(0.1) * 1000.0),
        ("1e-3 m", "length", 1e-3),
        ("2 kN", "force", 2000.0),
        ("9.81 m/s^2", "accel", 9.81),
        ("9.81 m/s2", "accel", 9.81),
        ("2 h", "time", 7200.0),
        ("90 min", "time", 5400.0),
        ("30 s", "time", 30.0),
    ],
)
def test_parse_units(text: str, dim: str, expected: float) -> None:
    assert parse_quantity(text, dim) == pytest.approx(expected)


def test_plain_numbers_are_si() -> None:
    assert parse_quantity(0.5, "length") == 0.5
    assert parse_quantity(3, "angle") == 3.0
    assert parse_quantity("0.25", "ratio") == 0.25


@pytest.mark.parametrize(
    ("text", "dim"), [("5 mm", "angle"), ("5 deg", "length"), ("1 N", "ratio")]
)
def test_wrong_dimension_names_expected_and_given(text: str, dim: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_quantity(text, dim)
    assert repr(normalize_dim(dim)) in str(e.value)
    assert text.split()[1] in str(e.value)


def test_unparsable() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_quantity("abc", "length")
    with pytest.raises(ValueError, match="unknown unit"):
        parse_quantity("3 furlong", "length")


def test_slope_dim() -> None:
    assert slope_dim("angle", "length") == "angle/length"
    assert slope_dim("angle", "angle") == "ratio"
    assert slope_dim("length", "length") == "ratio"
    assert slope_dim("length", "angle") == "length/angle"
    assert slope_dim("ratio", "length") == "1/length"
    assert normalize_dim("stiffness") == "force/length"


def test_display_roundtrip() -> None:
    assert display_unit("length") == "mm"
    assert display_unit("angle/length") == "deg/mm"
    assert display_unit("stiffness") == "N/mm"
    assert display_unit("ratio") == ""
    assert display_unit("percent") == "%"


def test_mm_per_s2_is_not_confused_with_m_per_s2() -> None:
    with pytest.raises(ValueError, match=r"unknown unit|expected .accel."):
        parse_quantity("5 mm/s^2", "accel")
