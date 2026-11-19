"""Framework-relative path resolution.

``FRAMEWORK_ROOT`` is ``$TAL_CALIBRATION_ROOT`` if set, else the nearest
directory above this file that contains ``pyproject.toml``; every data path is
resolved relative to it, from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "FRAMEWORK_ROOT",
    "DATA_ROOT",
    "RAW_DATA_ROOT",
    "CONFIG_ROOT",
    "RESULTS_ROOT",
    "RUNS_ROOT",
    "assert_not_in_results",
    "resolve",
]


def _find_root() -> Path:
    override = os.environ.get("TAL_CALIBRATION_ROOT")
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(
                f"TAL_CALIBRATION_ROOT={p} does not exist or is not a directory"
            )
        return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Cannot locate the framework root: no pyproject.toml found above "
        f"{here} and TAL_CALIBRATION_ROOT is not set. Set TAL_CALIBRATION_ROOT "
        "to the framework directory (the one containing pyproject.toml)."
    )


FRAMEWORK_ROOT: Path = _find_root()
DATA_ROOT: Path = FRAMEWORK_ROOT / "data"
RAW_DATA_ROOT: Path = DATA_ROOT / "raw"
CONFIG_ROOT: Path = FRAMEWORK_ROOT / "configs"
#: the frozen results the paper is built from (tracked; see results/README.md)
RESULTS_ROOT: Path = FRAMEWORK_ROOT / "results"
#: scratch output of runs (gitignored)
RUNS_ROOT: Path = FRAMEWORK_ROOT / "runs"


def assert_not_in_results(path: str | os.PathLike[str]) -> Path:
    """Refuse a run output path inside ``results/``; runs write to ``runs/``."""
    p = Path(path).resolve()
    root = RESULTS_ROOT.resolve()
    if p == root or root in p.parents:
        raise ValueError(f"refusing to write run output into {p}: it is inside {root}")
    return p


def resolve(rel: str | os.PathLike[str]) -> Path:
    """Resolve a framework-relative path to an absolute ``Path``; absolute
    paths and paths that ``..`` out of ``FRAMEWORK_ROOT`` are rejected."""
    p = Path(rel)
    if p.is_absolute():
        raise ValueError(
            f"resolve() takes framework-relative paths only; got absolute {p!r}"
        )
    resolved = (FRAMEWORK_ROOT / p).resolve()
    # compares fully resolved paths, so "data/../../etc" is caught
    if not resolved.is_relative_to(FRAMEWORK_ROOT):
        raise ValueError(
            f"resolve() path escapes framework root: {rel!r} -> {resolved}"
        )
    return resolved
