"""Deprecation helper for the stable public API."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    reason: str, *, since: str, removed_in: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function or method deprecated; calls emit a :class:`DeprecationWarning`.

    Works on plain functions and methods only, not on classes or properties.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        msg = (
            f"{func.__qualname__} is deprecated since {since} and will be removed in "
            f"{removed_in}: {reason}"
        )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
