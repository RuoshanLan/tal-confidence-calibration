"""Table 1 score-only localization-quality estimation runner.

Writes the R^2 cells of Table 1 for its 16 pairs into a new
directory ``<runs-root>/<timestamp>/12_table1_quality_estimation.csv``. Nothing
is written into ``data/raw`` or ``results/``.

Run from any cwd:

    python experiments/benchmark/run_benchmark_quality.py [--runs-root DIR]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tal_calibration import paths
from tal_calibration.paths import assert_not_in_results
from tal_calibration.protocols.benchmark import benchmark_tasks, protocol
from tal_calibration.protocols.benchmark import table1_quality_estimation as T1

RUNS_ROOT = paths.RUNS_ROOT / "benchmark_quality"
SUMMARY_CSV = "12_table1_quality_estimation.csv"


def compute(log=print):
    """Every (evaluation population, conditioning) cell of each Table 1 pair,
    in ``TABLE1_PAIRS`` order."""
    pairs = list(T1.TABLE1_PAIRS)
    cfg_ds, pairs_cfg = protocol.benchmark_matrix()
    rows = []
    for i, (ds, mo) in enumerate(pairs, 1):
        log(f"\n=== [{i}/{len(pairs)}] {ds}:{mo} ===")
        ctx = benchmark_tasks.build_pair_context(ds, mo, cfg_ds, pairs_cfg[(ds, mo)],
                                                 log=log)
        rows.extend(T1.run_pair(ctx, log=log))
    summary = T1.summary_frame(rows)
    n_expected = len(T1.EVAL_POPULATIONS) * len(T1.POPULATIONS) * len(pairs)
    if len(summary) != n_expected:
        raise RuntimeError(f"summary has {len(summary)} rows, expected {n_expected}")
    numeric = summary.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        bad = [c for c in numeric.columns
               if not np.isfinite(numeric[c].to_numpy(dtype=np.float64)).all()]
        raise RuntimeError(f"non-finite values in summary columns {bad}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Table 1 score-only localization-quality estimation")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    args = ap.parse_args(argv)
    try:
        assert_not_in_results(args.runs_root)
    except ValueError as e:
        ap.error(str(e))
    summary = compute()
    run_dir = Path(args.runs_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol.atomic_write_text(run_dir / SUMMARY_CSV, summary.to_csv(index=False))
    print(f"\nwrote {run_dir / SUMMARY_CSV} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
