"""Worker-count resolution and the spawn process pool shared by the evaluation loops."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

__all__ = ["make_pool", "resolve_workers"]


def resolve_workers(workers: int | str) -> int:
    """``auto`` -> ``max(1, cpu_count - 1)``; an int is used as given."""
    if workers == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return int(workers)


def make_pool(workers: int, init: Callable[..., None], initargs: tuple[Any, ...]) -> Executor:
    """Spawn-context process pool whose workers run ``init(*initargs)`` once."""
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init,
        initargs=initargs,
    )
