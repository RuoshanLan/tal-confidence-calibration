"""The (dataset, seed) shard of the quality_uncertainty chain.

  0/1. build the head (``estimation.head.build_head``): universe, the seed's
       video partition, and the LRP selection made BEFORE any fitting;
  2. point regressors fitted on selected TRAIN rows, evaluated
     on selected TEST rows;
  3. the matched-subset obs-vs-oracle diagnostic on the same rows restricted
     to q > 0 (Table 4 "Matched R^2");
  4. every uncertainty arm (interval and floor) fitted on TRAIN, conformalized on
     CAL, evaluated on TEST.

Tables: point, mechanism, interval, floor (one row per arm) and selection
(one row per partition: realized size and score cutoff).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..estimation import population as POP
from ..estimation import operating_point as SEL
from ..estimation.estimators import GBR_PARAMS
from ..estimation.arms import (
    BASELINE_TAGS, ORACLE_FEATURE, PAIR_BASELINES, assert_registries_wellformed,
)
from ..estimation.conformal import (
    ALPHA, CONFORMAL_MODE, COVERAGE_ESTIMAND, one_sided_arm, two_sided_arm,
)
from ..estimation.head import HeadError, build_head
from ..estimation.metrics import (
    floor_metrics, floor_row_quantities, interval_metrics,
    interval_row_quantities, point_metrics,
)
from ..estimation.bootstrap import point_r2_bootstrap, weighted_mean_ci
from .tasks import EXPERIMENT

#: per-row quantities that get a bootstrap CI (and paired contrasts vs the null)
INTERVAL_CI_STATS = ("covered", "width", "iscore")
FLOOR_CI_STATS = ("covered", "floor", "positive_floor")


class ShardError(RuntimeError):
    pass


def _design(pop, feats):
    if not feats:
        return np.empty((len(pop), 0))
    return pop[list(feats)].to_numpy(dtype=np.float64)


def _fit_point_arm(pop, arm, split, y):
    """TRAIN-fit / TEST-eval point predictions for one arm."""
    tr, te = (split == 0), (split == 2)
    X = _design(pop, arm.check(pop))
    mdl = HistGradientBoostingRegressor(**GBR_PARAMS)
    mdl.fit(X[tr], y[tr])
    return mdl.predict(X[te])


def chain_shard(dataset, seed, n_boot, feature_interface="full_observable",
                detector="actionformer", canonical_dataset=None, log=print):
    """One (dataset, seed) shard. Returns ``{table name: DataFrame}``."""
    assert_registries_wellformed()
    # ---- 0/1. universe, partition, selection ----
    # The detector's registered feature interface picks the arm bundle; the
    # bundles have identical arm cardinality, so the bootstrap matrix width
    # is the same for every detector.
    try:
        ctx = build_head(dataset, seed,
                         feature_interface=feature_interface, log=log)
    except HeadError as e:
        raise ShardError(str(e)) from e
    bundle = ctx.bundle
    POINT_ARMS = bundle.point
    UNCERTAINTY_ARMS = bundle.uncertainty
    MECHANISM_ARMS = bundle.mechanism
    MECH_OBS_ARM, MECH_ORACLE_ARM = bundle.mech_obs, bundle.mech_oracle
    pop, split, y = ctx.pop, ctx.split, ctx.y
    ledgers = ctx.ledgers
    tr, ca, te = ctx.tr, ctx.ca, ctx.te
    y_te, is_hit_te, vids_test = ctx.y_te, ctx.is_hit_te, ctx.vids_test
    counts, shared = ctx.counts, ctx.shared

    # ---- 2. point estimation ----
    # `dataset` is the registration key; `detector` and `canonical_dataset`
    # let a table be grouped by (dataset, detector) without parsing it.
    arm_identity = {"detector": detector,
                    "canonical_dataset": canonical_dataset or dataset,
                    "feature_interface": feature_interface}
    point_identity = {"dataset": dataset, "seed": seed,
                      **arm_identity,
                      "population": POP.POPULATION_NAME,
                      "target": POP.TARGET_NAME, "n_boot": int(n_boot)}
    point_preds = {name: _fit_point_arm(pop, arm, split, y)
                   for name, arm in POINT_ARMS.items()}
    point_ci, _ = point_r2_bootstrap(
        vids_test, y_te, point_preds,
        np.random.default_rng(seed + 2000), n_boot)
    point_rows = []
    for name, arm in POINT_ARMS.items():
        m = point_metrics(y_te, point_preds[name])
        lo, hi = point_ci[name]
        point_rows.append({
            **point_identity, "arm": name, "subset": "full_head",
            "n_features": len(arm.features),
            "fold_clean_priors": bool(arm.fold_clean_priors), **m,
            "r2_ci_lo": lo, "r2_ci_hi": hi, **shared,
        })
        log(f"[{dataset}/{EXPERIMENT}/s{seed}] point {name:16s} "
            f"r2={m['r2']:+.3f} [{lo:+.3f},{hi:+.3f}]""")

    # ---- 3. matched-subset obs-vs-oracle diagnostic ----
    m_mask = pop["is_hit"].to_numpy(dtype=bool)
    sub = pop[m_mask].reset_index(drop=True)
    split_m = split[m_mask]
    y_m = y[m_mask]
    oracle_r = np.log(sub["matched_gt_dur"].to_numpy(dtype=np.float64)
                      / sub["pred_len"].to_numpy(dtype=np.float64))
    if not np.isfinite(oracle_r).all():
        raise ShardError(f"[{dataset}/s{seed}] non-finite oracle R on the "
                         "matched subset")
    sub[ORACLE_FEATURE] = oracle_r
    tr_m, te_m = (split_m == 0), (split_m == 2)
    if not tr_m.any() or not te_m.any():
        raise ShardError(f"[{dataset}/s{seed}] matched subset lacks TRAIN or "
                         "TEST rows; the mechanism diagnostic is undefined")
    y_m_te = y_m[te_m]
    sub_test = sub[te_m]
    mech_preds = {}
    for name, arm in MECHANISM_ARMS.items():
        X = _design(sub, arm.check(sub))
        mdl = HistGradientBoostingRegressor(**GBR_PARAMS)
        mdl.fit(X[tr_m], y_m[tr_m])
        mech_preds[name] = mdl.predict(X[te_m])
    mech_ci, mech_pair = point_r2_bootstrap(
        sub_test["video-id"].astype(str), y_m_te, mech_preds,
        np.random.default_rng(seed + 3000), n_boot,
        pairs=[(MECH_ORACLE_ARM, MECH_OBS_ARM)])
    mech_metrics = {n: point_metrics(y_m_te, p) for n, p in mech_preds.items()}
    gap = mech_metrics[MECH_ORACLE_ARM]["r2"] - mech_metrics[MECH_OBS_ARM]["r2"]
    gap_lo, gap_hi = mech_pair[(MECH_ORACLE_ARM, MECH_OBS_ARM)]
    mech_counts = {
        "n_train_matched": int(tr_m.sum()),
        "n_test_matched": int(te_m.sum()),
        "n_test_matched_videos": int(sub_test["video-id"].nunique()),
    }
    mech_rows = []
    for name, arm in MECHANISM_ARMS.items():
        lo, hi = mech_ci[name]
        mech_rows.append({
            **point_identity, "arm": name, "subset": "matched",
            "diagnostic": True, "n_features": len(arm.features),
            **mech_metrics[name], "r2_ci_lo": lo, "r2_ci_hi": hi,
            "oracle_gap": float(gap),
            "oracle_gap_ci_lo": gap_lo, "oracle_gap_ci_hi": gap_hi,
            **mech_counts, "hit_rate_test": counts["hit_rate_test"],
            "unmatched_target_max_abs": shared["unmatched_target_max_abs"],
        })
    log(f"[{dataset}/{EXPERIMENT}/s{seed}] mechanism obs="
        f"{mech_metrics[MECH_OBS_ARM]['r2']:+.3f} oracle="
        f"{mech_metrics[MECH_ORACLE_ARM]['r2']:+.3f} "
        f"gap={gap:+.3f} [{gap_lo:+.3f},{gap_hi:+.3f}]")

    # ---- 4. interval + floor on the identical rows ----
    identity = {
        "dataset": dataset, "seed": seed, **arm_identity,
        "conformal": CONFORMAL_MODE,
        "alpha": ALPHA, "population": POP.POPULATION_NAME,
        "target": POP.TARGET_NAME,
        "coverage_estimand": COVERAGE_ESTIMAND, "n_boot": int(n_boot),
    }
    interval_fits, floor_fits = {}, {}
    for name, arm in UNCERTAINTY_ARMS.items():
        arm.check(pop)
        interval_fits[name] = two_sided_arm(pop, arm, split, y, seed)
        floor_fits[name] = one_sided_arm(pop, arm, split, y, seed)

    interval_rq = {n: interval_row_quantities(f, y_te)
                   for n, f in interval_fits.items()}
    floor_rq = {n: floor_row_quantities(f, y_te) for n, f in floor_fits.items()}

    # one bootstrap pass over all arms/outputs, so pairing is exact
    columns, col_index = [], {}
    for name in UNCERTAINTY_ARMS:
        for stat in INTERVAL_CI_STATS:
            col_index[("interval", name, stat)] = len(columns)
            columns.append(interval_rq[name][stat])
        for stat in FLOOR_CI_STATS:
            col_index[("floor", name, stat)] = len(columns)
            columns.append(floor_rq[name][stat])
    M = np.column_stack(columns)

    pairs, pair_keys = [], []
    for output, stats in (("interval", INTERVAL_CI_STATS),
                          ("floor", FLOOR_CI_STATS)):
        for name in UNCERTAINTY_ARMS:
            for base in PAIR_BASELINES:
                if name == base:
                    continue
                for stat in stats:
                    pairs.append((col_index[(output, name, stat)],
                                  col_index[(output, base, stat)]))
                    pair_keys.append((output, name, base, stat))
    ci_lo, ci_hi, pair_lo, pair_hi = weighted_mean_ci(
        vids_test, M, np.random.default_rng(seed + 1000), n_boot, pairs)
    pair_ci = {k: (float(lo), float(hi))
               for k, lo, hi in zip(pair_keys, pair_lo, pair_hi)}

    # Video-macro coverage CI: collapse `covered` to per-video means, then
    # bootstrap those means over videos, matching the video-balanced estimand.
    macro_index, macro_names = {}, []
    macro_cols = []
    vids_arr = np.asarray(vids_test)
    for output, rqs in (("interval", interval_rq), ("floor", floor_rq)):
        for name in UNCERTAINTY_ARMS:
            macro_index[(output, name)] = len(macro_cols)
            macro_cols.append(rqs[name]["covered"])
            macro_names.append((output, name))
    Mv = (pd.DataFrame(np.column_stack(macro_cols))
          .groupby(vids_arr).mean())
    mac_lo, mac_hi, _, _ = weighted_mean_ci(
        Mv.index.to_numpy(), Mv.to_numpy(),
        np.random.default_rng(seed + 5000), n_boot)

    def _macro_ci(output, name):
        j = macro_index[(output, name)]
        return float(mac_lo[j]), float(mac_hi[j])

    def _ci(output, name, stat):
        j = col_index[(output, name, stat)]
        return float(ci_lo[j]), float(ci_hi[j])

    def _pair_cols(output, name, stats):
        out = {}
        for base in PAIR_BASELINES:
            tag = BASELINE_TAGS[base]
            for stat in stats:
                if name == base:
                    lo = hi = 0.0
                else:
                    lo, hi = pair_ci[(output, name, base, stat)]
                out[f"pair_{stat}_vs_{tag}_lo"] = lo
                out[f"pair_{stat}_vs_{tag}_hi"] = hi
        return out

    interval_rows, floor_rows = [], []
    for name, arm in UNCERTAINTY_ARMS.items():
        m = interval_metrics(interval_rq[name], is_hit_te,
                             interval_fits[name], vids_test)
        cov_lo, cov_hi = _ci("interval", name, "covered")
        w_lo, w_hi = _ci("interval", name, "width")
        s_lo, s_hi = _ci("interval", name, "iscore")
        interval_rows.append({
            **identity, "arm": name, "n_features": len(arm.features), **m,
            "cov_ci_lo": cov_lo, "cov_ci_hi": cov_hi,
            "coverage_video_macro_ci_lo": _macro_ci("interval", name)[0],
            "coverage_video_macro_ci_hi": _macro_ci("interval", name)[1],
            "mean_width_ci_lo": w_lo, "mean_width_ci_hi": w_hi,
            "interval_score_ci_lo": s_lo, "interval_score_ci_hi": s_hi,
            **_pair_cols("interval", name, INTERVAL_CI_STATS), **shared,
        })

        f = floor_metrics(floor_rq[name], is_hit_te, floor_fits[name],
                          vids_test)
        fcov_lo, fcov_hi = _ci("floor", name, "covered")
        fm_lo, fm_hi = _ci("floor", name, "floor")
        fp_lo, fp_hi = _ci("floor", name, "positive_floor")
        floor_rows.append({
            **identity, "arm": name, "n_features": len(arm.features), **f,
            "cov_ci_lo": fcov_lo, "cov_ci_hi": fcov_hi,
            "coverage_video_macro_ci_lo": _macro_ci("floor", name)[0],
            "coverage_video_macro_ci_hi": _macro_ci("floor", name)[1],
            "floor_mean_ci_lo": fm_lo, "floor_mean_ci_hi": fm_hi,
            "positive_floor_rate_ci_lo": fp_lo,
            "positive_floor_rate_ci_hi": fp_hi,
            **_pair_cols("floor", name, FLOOR_CI_STATS), **shared,
        })
        log(f"[{dataset}/{EXPERIMENT}/s{seed}] {name:19s} "
            f"cov={m['coverage']:.3f}/{m['coverage_video_macro']:.3f} "
            f"iscore={m['interval_score']:.4f} "
            f"floor_med={f['floor_median']:.3f} "
            f"pos_floor={f['positive_floor_rate']:.3f}")

    selection_table = pd.DataFrame(SEL.ledger_rows(
        dataset, seed, ledgers, head=POP.POPULATION_NAME))
    for col, val in arm_identity.items():
        selection_table[col] = val

    return {"point": pd.DataFrame(point_rows),
            "mechanism": pd.DataFrame(mech_rows),
            "interval": pd.DataFrame(interval_rows),
            "floor": pd.DataFrame(floor_rows),
            "selection": selection_table}
