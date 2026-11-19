"""TAL calibration framework: calibration, temporal geometry, prediction I/O,
matching and evaluation shared by every experiment."""
from __future__ import annotations

__version__ = "0.0.0"

from .paths import (  # noqa: F401
    FRAMEWORK_ROOT,
    DATA_ROOT,
    RAW_DATA_ROOT,
    CONFIG_ROOT,
    RESULTS_ROOT,
    RUNS_ROOT,
    resolve,
)

__all__ = [
    "__version__",
    "FRAMEWORK_ROOT",
    "DATA_ROOT",
    "RAW_DATA_ROOT",
    "CONFIG_ROOT",
    "RESULTS_ROOT",
    "RUNS_ROOT",
    "resolve",
]
