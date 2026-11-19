"""Driver of the quality_uncertainty chain.

``run`` expands the configured tasks, runs one shard per (dataset, seed),
checks each shard's tables (``contracts``), builds the readout tables and,
when the config asks for it, the figure data, and writes every summary CSV
into a fresh run directory ``<runs_root>/<run_id>/summary/``. Prepared data,
cloud features and video partitions are recomputed in memory from the raw
inputs on every run; nothing is cached on disk.
"""
import datetime
import shutil
import uuid
from pathlib import Path

import pandas as pd
import yaml

from ....paths import RUNS_ROOT as _RUNS_ROOT
from . import contracts
from . import figure_data
from . import readout as READOUT
from . import shard as X
from .tasks import expand_tasks, registered_arms

#: default run root; ``run(runs_root=...)`` overrides it
RUNS_ROOT = _RUNS_ROOT / "quality_uncertainty"

#: per-task table -> summary CSV
_TABLE_CSVS = {
    "point": "10_point_seeds.csv",
    "mechanism": "10_mechanism_seeds.csv",
    "interval": "10_interval_seeds.csv",
    "floor": "10_floor_seeds.csv",
    "selection": "10_selection_ledger.csv",
}


def run(config, runs_root=None, log=print):
    """Execute one fresh run into ``runs_root`` (default ``RUNS_ROOT``) and
    return the completed run directory. A failed run is moved to
    ``<runs_root>/failed/<run_id>`` and the exception re-raised."""
    runs_root = Path(RUNS_ROOT if runs_root is None else runs_root)
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    tmp = runs_root / f".{run_id}.tmp"
    summ = tmp / "summary"
    summ.mkdir(parents=True)
    with open(tmp / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    try:
        _run_body(config, summ, log)
    except Exception as e:
        failed = runs_root / "failed" / run_id
        failed.parent.mkdir(parents=True, exist_ok=True)
        if failed.exists():
            shutil.rmtree(failed)
        tmp.replace(failed)
        log(f"[quality_uncertainty] FAILED {run_id}: {e} (kept at {failed})")
        raise

    completed = runs_root / run_id
    tmp.replace(completed)
    log(f"[quality_uncertainty] completed {run_id} -> {completed}")
    return completed


def _run_body(config, summ, log):
    arms = registered_arms()
    tasks = expand_tasks(config, arms=arms)
    n_boot = config["run"]["n_boot"]

    collected = {}
    for t in tasks:
        tables = X.chain_shard(
            t.dataset, t.seed, n_boot=n_boot,
            feature_interface=t.feature_interface, detector=t.detector,
            canonical_dataset=t.canonical_dataset, log=log)
        contracts.validate_output(t, tables)
        for name, df in tables.items():
            collected.setdefault(name, []).append(df)

    # The readout tables run after every task has been checked; their scope is
    # the registered readout arms.
    readout = (READOUT.build_readout(tasks, arms, log=log)
               if contracts.readout_in_scope(config) else {})

    for name, csv_name in _TABLE_CSVS.items():
        pd.concat(collected[name], ignore_index=True).to_csv(
            summ / csv_name, index=False)
    for name, csv_name in READOUT.READOUT_CSVS.items():
        df = readout.get(name)
        if df is not None and len(df):
            df.to_csv(summ / csv_name, index=False)
    if contracts.figure_data_in_scope(config):
        figure_data.write(summ, log=log)

    expected = contracts.expected_summary_csvs(config)
    present = {p.name for p in summ.glob("*.csv")}
    if present != expected:
        raise contracts.ContractViolation(
            f"summary CSV set {sorted(present)} != expected {sorted(expected)}")
