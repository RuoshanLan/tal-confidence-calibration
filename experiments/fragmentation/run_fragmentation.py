"""Fragmentation entry point: write the fragmentation chain's result CSVs.

Labels every pool with the official and the fragment-tolerant matcher and fits
both calibrator families from the raw predictions on each run (nothing is
cached), for the official top-K pool and the LRP operating point. Never reads
the benchmark chain's outputs.

Outputs:
  * 03_matching_summary.csv       -- transfer / refit ECE against the
    fragment-tolerant labels, delta_match_ece and its paired video-bootstrap CI
    (Table 3, supp. Table S5);
  * fragment_counts.csv           -- predictions, official TPs and rescued
    fragments per (pair, population, tau) (Table 3 "Rescued", Fig. 4);
  * fig1b_fragment_example.csv    -- the real instance drawn in Fig. 1(b).

The framework produces numeric CSVs only; tables and figures are drawn from
them elsewhere.

Run from any cwd:
    python experiments/fragmentation/run_fragmentation.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tal_calibration.paths import RUNS_ROOT, assert_not_in_results
from tal_calibration.protocols.benchmark import fragmentation_tasks as F
from tal_calibration.protocols.benchmark import protocol

# result name -> CSV basename
FRAGMENTATION_CSV = {
    "matching_summary": "03_matching_summary.csv",
    "fragment_counts": "fragment_counts.csv",
    "fragment_example": "fig1b_fragment_example.csv",
}

OUT_DIR = RUNS_ROOT / "fragmentation"

CI_COLS = ["delta_match_ece_ci_lo", "delta_match_ece_ci_hi"]


def compute(log=print):
    """matching_audit + paired bootstrap for every (pair, population), in
    ``TABLE1_PAIRS`` (Table 1) then ``POPULATIONS`` order, all in memory."""
    cfg_ds, pairs_cfg = protocol.benchmark_matrix()
    pair_list = list(protocol.TABLE1_PAIRS)

    count_rows, summary_rows, example = [], [], None
    for i, (ds, mo) in enumerate(pair_list, 1):
        log(f"\n=== [{i}/{len(pair_list)}] {ds}:{mo} ===")
        for ctx in F.build_pair_contexts(ds, mo, cfg_ds, pairs_cfg[(ds, mo)], log=log):
            cr, sr = F.matching_audit(ctx, log=log)
            ci_map = F.bootstrap_audit(ctx)
            for r in sr:
                r.update(ci_map[protocol.tau_r(r["tau"])])
            count_rows.extend(cr)
            summary_rows.extend(sr)
            if (ds, mo, ctx["population"]) == (F.EXAMPLE["dataset"], F.EXAMPLE["model"],
                                               F.EXAMPLE["population"]):
                example = F.fragment_example(ctx)

    results = {"matching_summary": pd.DataFrame(summary_rows),
               "fragment_counts": pd.DataFrame(count_rows)}
    if example is not None:
        results["fragment_example"] = example
    return results


def validate(results):
    """Exact (pair, population, tau) grid, finite cells, ordered CIs and the
    one-directional relaxation (rescued fragments are never negative)."""
    pair_list = list(protocol.TABLE1_PAIRS)
    ms = results["matching_summary"]
    protocol.validate_exact_keys(
        ms, "03_matching_summary",
        ["dataset", "model", "population", "tau", "arm"],
        F.keys_matching_summary(set(pair_list)))
    num = ["ece_cfit_yfrag", "ece_ffit_yfrag", "delta_match_ece"] + CI_COLS
    if not np.isfinite(ms[num].to_numpy(dtype=np.float64)).all():
        raise SystemExit("03_matching_summary: non-finite ECE or CI cell")
    if (ms["delta_match_ece_ci_lo"] > ms["delta_match_ece_ci_hi"]).any():
        raise SystemExit("03_matching_summary: CI lower bound above upper bound")
    fc = results["fragment_counts"]
    key = ["dataset", "model", "population", "tau"]
    if not fc.set_index(key).index.equals(ms.set_index(key).index):
        raise SystemExit("fragment_counts: rows differ from 03_matching_summary")
    if (fc["n_rescued"] < 0).any() or (fc["n_predictions"] != ms["population_n"].to_numpy()).any():
        raise SystemExit("fragment_counts: negative rescue count or wrong population size")


def main(argv=None):
    ap = argparse.ArgumentParser(description="fragmentation: write the result CSVs")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    try:
        assert_not_in_results(args.out_dir)
    except ValueError as e:
        ap.error(str(e))
    results = compute()
    validate(results)  # before anything hits the target dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nwrote:")
    for name, df in results.items():
        path = out_dir / FRAGMENTATION_CSV[name]
        protocol.atomic_write_text(path, df.to_csv(index=False))
        print(f"  {path.name}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
