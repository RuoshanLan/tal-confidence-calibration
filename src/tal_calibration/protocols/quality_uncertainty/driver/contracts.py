"""Output checks for the quality_uncertainty chain.

A shard's tables are checked before they can reach a summary CSV. Each check
guards a released number: a violation means a value would be wrong, or would
describe a different population from the one the paper states.

  * every registered arm appears exactly once per table, no extras;
  * the unmatched target is exactly 0 (``unmatched_target_max_abs``);
  * the fits consumed exactly the selected rows: per partition the ledger's
    realized total equals the row count, and point / interval / floor carry
    the same counts;
  * ``trivial_zero`` is recorded as not fitted with a NaN correction, and every
    arm's head status agrees with its fitted flag;
  * paired CIs are finite and ordered, and the null's self-contrast is 0;
  * the mechanism table's matched count equals the interval table's hits
    (matched <=> q>0) and its gap equals oracle R^2 - obs R^2;
  * the ledger has one partition-level row per partition with a finite cutoff.
"""
import numpy as np

from ..estimation.arms import BASELINE_TAGS, NULL_ARM, bundle_for
from ..estimation.conformal import HEAD_NOT_FITTED, HEAD_STATUSES
from ..estimation.metrics import RATE_COLUMNS
from ..estimation.operating_point import PARTITION_NAMES
from .readout import READOUT_ARMS
from .readout import READOUT_CSVS as _READOUT_TABLE_CSVS

TABLES = ("point", "mechanism", "interval", "floor", "selection")
SUMMARY_CSVS = (
    "10_point_seeds.csv",
    "10_mechanism_seeds.csv",
    "10_interval_seeds.csv",
    "10_floor_seeds.csv",
    "10_selection_ledger.csv",
)
READOUT_CSVS = tuple(_READOUT_TABLE_CSVS.values())
#: the figure data a run writes when its config asks for it
FIGURE_CSVS = ("fig1c_hidden_extent.csv",)
_INTERVAL_PAIR_STATS = ("covered", "width", "iscore")
_FLOOR_PAIR_STATS = ("covered", "floor", "positive_floor")


class ContractViolation(RuntimeError):
    pass


def _fail(task, msg):
    raise ContractViolation(f"{task.task_id}: {msg}")


def _check_arms(task, df, registry, label):
    got = list(df["arm"])
    if len(got) != len(set(got)):
        _fail(task, f"{label}: duplicate arm rows")
    if set(got) != set(registry):
        missing = sorted(set(registry) - set(got))
        extra = sorted(set(got) - set(registry))
        _fail(task, f"{label}: arm set mismatch missing={missing} "
                    f"extra={extra}")


def _check_rates(task, df, label):
    for col in RATE_COLUMNS:
        if col not in df.columns:
            continue
        v = df[col].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            _fail(task, f"{label}.{col} has non-finite values")
        if not ((v >= 0).all() and (v <= 1).all()):
            _fail(task, f"{label}.{col} outside [0,1]")


def _check_uncertainty(task, df, pair_stats, label):
    unc = bundle_for(task.feature_interface).uncertainty
    _check_arms(task, df, unc, label)
    _check_rates(task, df, label)

    tgt = df["unmatched_target_max_abs"].to_numpy(dtype=float)
    if not (tgt == 0.0).all():
        _fail(task, f"{label}: unmatched target is not exactly 0 "
                    f"(max |q| = {tgt.max()})")
    for col in ("n_train", "n_cal", "n_test", "n_train_videos",
                "n_cal_videos", "n_test_videos"):
        if (df[col].to_numpy(dtype=float) <= 0).any():
            _fail(task, f"{label}.{col} must be positive on every arm")
    for pname, ncol in (("train", "n_train"), ("cal", "n_cal"),
                        ("test", "n_test")):
        real = df[f"sel_realized_{pname}"].to_numpy(dtype=float)
        if not (real == df[ncol].to_numpy(dtype=float)).all():
            _fail(task, f"{label}: {ncol} != ledger realized total ({pname}) "
                        "-- fits did not consume exactly the selected rows")

    fitted = dict(zip(df["arm"], df["fitted"].astype(bool)))
    for name, arm in unc.items():
        if fitted[name] != arm.fitted:
            _fail(task, f"{label}: arm {name} fitted={fitted[name]} != "
                        f"registry ({arm.fitted})")
    bad = set(df["head_status"].unique()) - set(HEAD_STATUSES)
    if bad:
        _fail(task, f"{label}: unknown head_status {sorted(bad)}")
    for _, r in df.iterrows():
        if (r["head_status"] == HEAD_NOT_FITTED) != (not bool(r["fitted"])):
            _fail(task, f"{label}: arm {r['arm']} head_status "
                        f"{r['head_status']!r} contradicts fitted={r['fitted']}")
    triv = df[df["arm"] == "trivial_zero"]
    if not np.isnan(triv["delta_alpha"].to_numpy(dtype=float)).all():
        _fail(task, f"{label}: trivial_zero must carry a NaN conformal "
                    "correction (it is not fitted)")

    for name in unc:
        row = df[df["arm"] == name].iloc[0]
        for tag in BASELINE_TAGS.values():
            for stat in pair_stats:
                lo = float(row[f"pair_{stat}_vs_{tag}_lo"])
                hi = float(row[f"pair_{stat}_vs_{tag}_hi"])
                if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
                    _fail(task, f"{label}: pair_{stat}_vs_{tag} malformed on "
                                f"arm {name}")
                if name == NULL_ARM and not (lo == 0.0 and hi == 0.0):
                    _fail(task, f"{label}: the null arm's self-contrast "
                                f"pair_{stat}_vs_{tag} must be exactly 0")


def _check_point(task, pt):
    _check_arms(task, pt, bundle_for(task.feature_interface).point, "point")
    for col in ("mae", "rmse"):
        v = pt[col].to_numpy(dtype=float)
        if not (np.isfinite(v).all() and (v >= 0).all()):
            _fail(task, f"point.{col} missing or negative")
    r2 = pt["r2"].to_numpy(dtype=float)
    if not np.isfinite(r2).all() or (r2 > 1.0).any():
        _fail(task, "point.r2 non-finite or > 1")
    if (pt["r2_ci_hi"].to_numpy(dtype=float)
            < pt["r2_ci_lo"].to_numpy(dtype=float)).any():
        _fail(task, "point r2 CI inverted")


def _check_mechanism(task, mech, itv):
    b = bundle_for(task.feature_interface)
    _check_arms(task, mech, b.mechanism, "mechanism")
    n_hits = int(itv["n_test_hits"].iloc[0])
    if set(mech["n_test_matched"].astype(int)) != {n_hits}:
        _fail(task, f"mechanism.n_test_matched != interval n_test_hits "
                    f"{n_hits} (matched <=> q>0 must agree)")
    gap = mech["oracle_gap"].to_numpy(dtype=float)
    if not np.isfinite(gap).all() or len(set(np.round(gap, 12))) != 1:
        _fail(task, "mechanism.oracle_gap missing or inconsistent")
    if (mech["oracle_gap_ci_hi"].to_numpy(dtype=float)
            < mech["oracle_gap_ci_lo"].to_numpy(dtype=float)).any():
        _fail(task, "mechanism oracle gap CI inverted")
    obs = mech[mech["arm"] == b.mech_obs].iloc[0]
    orc = mech[mech["arm"] == b.mech_oracle].iloc[0]
    if not np.isclose(float(orc["r2"]) - float(obs["r2"]),
                      float(obs["oracle_gap"])):
        _fail(task, "mechanism: gap != oracle R^2 - obs R^2")


def _check_selection(task, sel, itv):
    if set(sel["partition"]) != set(PARTITION_NAMES.values()) \
            or len(sel) != len(PARTITION_NAMES):
        _fail(task, "selection must carry exactly one row per partition")
    if (sel["label"].to_numpy() != -1).any():
        _fail(task, "selection: partition-level rows must carry label == -1")
    if not np.isfinite(sel["score_cutoff"].to_numpy(dtype=float)).all():
        _fail(task, "selection: missing score cutoff")
    for pname, ncol in (("train", "n_train"), ("cal", "n_cal"),
                        ("test", "n_test")):
        total = int(sel.loc[sel["partition"] == pname, "realized"].sum())
        if total != int(itv[ncol].iloc[0]):
            _fail(task, f"selection {pname} realized {total} != "
                        f"interval {ncol} {int(itv[ncol].iloc[0])}")


def validate_output(task, tables):
    """Raise ContractViolation unless the shard's tables are well formed."""
    if set(tables) != set(TABLES):
        _fail(task, f"tables {sorted(tables)} != required {sorted(TABLES)}")
    for name, df in tables.items():
        if len(df) == 0:
            _fail(task, f"table {name} is empty")
    itv, flr, pt = tables["interval"], tables["floor"], tables["point"]
    _check_uncertainty(task, itv, _INTERVAL_PAIR_STATS, "interval")
    _check_uncertainty(task, flr, _FLOOR_PAIR_STATS, "floor")
    for col in ("mean_width", "median_width", "mean_width_hit"):
        v = itv[col].to_numpy(dtype=float)
        if not (np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            _fail(task, f"interval.{col} outside [0,1]")
    for col in ("floor_mean", "floor_median"):
        v = flr[col].to_numpy(dtype=float)
        if not (np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            _fail(task, f"floor.{col} outside [0,1]")
    if (itv["interval_score"].to_numpy(dtype=float) < 0).any():
        _fail(task, "interval.interval_score negative")
    for col in ("n_train", "n_cal", "n_test", "n_test_hits"):
        vals = (set(pt[col].astype(int)) | set(itv[col].astype(int))
                | set(flr[col].astype(int)))
        if len(vals) != 1:
            _fail(task, f"count column {col} disagrees across "
                        f"point/interval/floor: {sorted(vals)}")
    _check_point(task, pt)
    _check_mechanism(task, tables["mechanism"], itv)
    _check_selection(task, tables["selection"], itv)


def readout_in_scope(config):
    """True when the run builds the readout tables: the config does not switch
    them off (``run.readout: false``) and its dataset axis intersects the
    registered readout scope."""
    if not config["run"].get("readout", True):
        return False
    datasets = set()
    for spec in (config.get("experiments") or {}).values():
        datasets |= set(spec.get("datasets") or ())
    return bool(datasets & set(READOUT_ARMS))


def figure_data_in_scope(config):
    """True when the config asks for the figure data (``run.figure_data``)."""
    return bool(config["run"].get("figure_data", False))


def expected_summary_csvs(config):
    """The exact CSV set a run must leave in summary/."""
    expected = set(SUMMARY_CSVS)
    if readout_in_scope(config):
        expected |= set(READOUT_CSVS)
    if figure_data_in_scope(config):
        expected |= set(FIGURE_CSVS)
    return expected
