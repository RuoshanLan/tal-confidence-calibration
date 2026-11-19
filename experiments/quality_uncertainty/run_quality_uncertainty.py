"""Run the quality_uncertainty chain (Sec. 5 and supp. Sec. E).

Computes its prepared data in memory from the raw inputs and writes the
summary CSVs into ``<runs-root>/<run_id>/summary/`` (default
``runs/quality_uncertainty/``) and nowhere else.

    # the main-paper run: five ActionFormer datasets, seeds 0-4
    python experiments/quality_uncertainty/run_quality_uncertainty.py \
        --config src/tal_calibration/protocols/quality_uncertainty/configs/formal_all_datasets.yaml
    # the supplementary ten-arm detector matrix
    python experiments/quality_uncertainty/run_quality_uncertainty.py \
        --config src/tal_calibration/protocols/quality_uncertainty/configs/formal_multi_detector.yaml
"""
from __future__ import annotations

import argparse

from tal_calibration.protocols.quality_uncertainty.driver import driver, tasks


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="quality_uncertainty chain: point audit, mechanism "
                    "diagnostic, interval and floor on the LRP head")
    ap.add_argument("--config", default=str(tasks.CONFIG_PATH),
                    help="config yaml (default: formal_all_datasets.yaml)")
    ap.add_argument("--runs-root", default=str(driver.RUNS_ROOT),
                    help="directory the run is written under (default: "
                         "runs/quality_uncertainty)")
    args = ap.parse_args(argv)
    config = tasks.load_config(args.config)
    completed = driver.run(config, runs_root=args.runs_root)
    print(f"run complete: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
