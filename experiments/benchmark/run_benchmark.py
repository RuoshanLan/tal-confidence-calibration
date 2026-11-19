"""Benchmark entry point: write the benchmark's result CSVs (credit target only).

Labels every pool with the credit-only matcher and fits every calibrator from
the raw predictions on each run, and never reads the fragmentation chain's
outputs. Before anything is written, every output is verified to carry no frag
rows or columns.

Run from any cwd:
    python experiments/benchmark/run_benchmark.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tal_calibration.paths import RUNS_ROOT, assert_not_in_results
from tal_calibration.protocols.benchmark import benchmark_tasks, protocol

# result name -> CSV basename
BENCHMARK_CSV = {
    "benchmark_main": "01_benchmark_main.csv",
    "pair_summary": "01_pair_summary.csv",
    "dilution": "02_population_dilution.csv",
    "bootstrap_ci": "05_bootstrap_ci.csv",
    "laece_reveal": "08_laece_reveal.csv",
    "lrp": "10_lrp_operating_point.csv",
    "reliability_bins": "13_reliability_bins.csv",
}

OUT_DIR = RUNS_ROOT / "benchmark"


def compute(log=print):
    """Run every benchmark analysis for every pair in one pass, in memory.

    Pair order follows ``BENCHMARK_PAIRS`` (``OFFICIAL_POPULATION`` order),
    which fixes the CSV row order.
    """
    cfg_ds, pairs_cfg = protocol.benchmark_matrix()
    pair_list = list(benchmark_tasks.BENCHMARK_PAIRS)

    bench_rows, pair_rows, ci_rows = [], [], []
    dilution_rows, laece_rows, lrp_rows, bin_rows = [], [], [], []

    for i, (ds, mo) in enumerate(pair_list, 1):
        log(f"\n=== [{i}/{len(pair_list)}] {ds}:{mo} ===")
        ctx = benchmark_tasks.build_pair_context(ds, mo, cfg_ds, pairs_cfg[(ds, mo)],
                                                 log=log)
        pr, mr = benchmark_tasks.benchmark(ctx, log=log)
        pair_rows.append(pr)
        bench_rows.extend(mr)
        ci_rows.extend(benchmark_tasks.bootstrap_ci(ctx, log=log))
        dilution_rows.extend(benchmark_tasks.population_dilution(ctx, log=log))
        laece_rows.extend(benchmark_tasks.laece_reveal(ctx, log=log))
        lrp_rows.append(benchmark_tasks.lrp_operating_point(ctx, log=log))
        bin_rows.extend(benchmark_tasks.reliability_bins(ctx, log=log))

    return {
        "benchmark_main": pd.DataFrame(bench_rows),
        "pair_summary": pd.DataFrame(pair_rows),
        "bootstrap_ci": pd.DataFrame(ci_rows),
        "dilution": pd.DataFrame(dilution_rows),
        "laece_reveal": pd.DataFrame(laece_rows),
        "lrp": pd.DataFrame(lrp_rows),
        "reliability_bins": pd.DataFrame(bin_rows),
    }


def validate(results):
    """Exact credit-only key grid, zero frag semantics."""
    pair_list = list(benchmark_tasks.BENCHMARK_PAIRS)
    bm = results["benchmark_main"]
    protocol.validate_exact_keys(
        bm, "01_benchmark_main",
        ["dataset", "model", "population", "tau", "arm", "target"],
        benchmark_tasks.keys_benchmark_main(set(pair_list)))
    for name, df in results.items():
        bad = [c for c in df.columns if "frag" in c.lower()]
        if bad:
            raise SystemExit(f"{name}: frag columns {bad} in a benchmark output")
        if "target" in df.columns and set(df["target"].unique()) != {"credit"}:
            raise SystemExit(f"{name}: non-credit target rows present")
    benchmark_tasks.validate_dilution_grid(results["dilution"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="benchmark: write the result CSVs")
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
        path = out_dir / BENCHMARK_CSV[name]
        protocol.atomic_write_text(path, df.to_csv(index=False))
        print(f"  {path.name}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
