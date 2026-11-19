"""Annotation audit on ActivityNet-1.3 (synthetic boundary worlds): entry point.

Writes the chain's result CSVs to --out-dir (default runs/annotation/). The
frozen copies live in results/annotation/; compare a run against them with
``python -m tal_calibration.results compare annotation runs/annotation``.

Run from any cwd:
    python experiments/annotation/run_annotation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tal_calibration.paths import RUNS_ROOT, assert_not_in_results
from tal_calibration.protocols.annotation import lib as L
from tal_calibration.protocols.annotation import pipeline

OUT_DIR = RUNS_ROOT / "annotation"


def write_outputs(res, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for csv, df in pipeline.results_to_csv_frames(res).items():
        path = out_dir / csv
        L.atomic_write_text(path, df.to_csv(index=False))
        written[csv] = path
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Annotation audit (ActivityNet-1.3)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    try:
        assert_not_in_results(args.out_dir)
    except ValueError as e:
        ap.error(str(e))

    res = pipeline.compute()
    written = write_outputs(res, args.out_dir)
    print("\nwrote:")
    for name, path in sorted(written.items()):
        print(f"  {name}  ({len(pd.read_csv(path))} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
