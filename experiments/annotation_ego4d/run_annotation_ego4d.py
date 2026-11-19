"""Annotation audit on Ego4D-MQ with real annotators: entry point.

Released world = the union of every annotator's primary moments (ego4d.json,
the benchmark labels and the ActionFormer training annotation). Alternative
world = one annotator per clip, drawn uniformly. The chain lives in
``protocols.annotation.ego4d`` and calls the ActivityNet chain's matcher,
calibrator, metrics, population, floor, ICC and target-shift functions.
Its invariants (ego4d.validate) must pass before anything is written.

Writes the result CSVs to --out-dir (default runs/annotation_ego4d/); the
frozen copies live in results/annotation_ego4d/.

Run from any cwd:
    python experiments/annotation_ego4d/run_annotation_ego4d.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tal_calibration.paths import RUNS_ROOT, assert_not_in_results
from tal_calibration.protocols.annotation import lib as L
from tal_calibration.protocols.annotation import ego4d as E

OUT_DIR = RUNS_ROOT / "annotation_ego4d"
#: result key -> CSV basename. 04/05/10 reuse the ActivityNet chain's
#: functions and so carry its schema and file names.
OFFICIAL_CSV = {
    "parity": "00_parity.csv",
    "claim_a": "04_claimA_floor.csv",
    "claim_b": "05_claimB_icc.csv",
    "multiseed": "09_multiseed_ci.csv",
    "target_shift": "10_target_shift.csv",
}


def write_outputs(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for key, csv in OFFICIAL_CSV.items():
        path = out_dir / csv
        L.atomic_write_text(path, pd.DataFrame(list(res[key])).to_csv(index=False))
        written[csv] = path
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Annotation audit (Ego4D-MQ, real annotators)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    try:
        assert_not_in_results(args.out_dir)
    except ValueError as e:
        ap.error(str(e))
    res = E.compute()
    E.validate(res)
    written = write_outputs(res, args.out_dir)
    print("\nwrote:")
    for name, path in sorted(written.items()):
        print(f"  {name}  ({len(pd.read_csv(path))} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
