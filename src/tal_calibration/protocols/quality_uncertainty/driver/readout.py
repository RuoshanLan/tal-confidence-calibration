"""Readout tables: the chain's numbers beyond the five per-task tables.

The per-task tables fix alpha at .20, the population at the LRP head and the
target at ``q_o2o_tau0``; each readout table varies one of the three. Every
table builds its population with ``estimation.head.build_head``, the same
function ``chain_shard`` calls. The readout runs on the five main-paper
ActionFormer arms (READOUT_ARMS) only.

  readout_attribution  A/B decomposition + hit AUC (Table 4 "Hit", "Full",
                       "A-B")
  readout_convention   one-to-one vs max-tIoU target on the same rows
                       (supp. Table S12)
  readout_coverage     median floor and positive-floor rate against the
                       demanded coverage (supp. Fig. S2)
  readout_population   the feature-free floor against a score cutoff that
                       targets a TRAIN hit rate, with the recall it costs
                       (supp. Table S11)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..estimation import population as POP
from ..estimation.arms import ORACLE_FEATURE
from ..estimation.conformal import (
    ALPHA, CONFORMAL_MODE, CONFORMAL_UNIT, COVERAGE_ESTIMAND, one_sided_arm,
)
from ..estimation.estimators import GBR_PARAMS, fold_clean_prior_features
from ..estimation.head import build_head
from ..estimation.metrics import (
    floor_metrics, floor_row_quantities, r2_score, video_macro_mean,
)

#: the five main-paper ActionFormer arms
READOUT_ARMS = ("thumos", "epic_verb", "epic_noun", "ego4d", "anet")

READOUT_TABLES = ("readout_attribution", "readout_convention",
                  "readout_coverage", "readout_population")
READOUT_CSVS = {
    "readout_attribution": "11_readout_attribution.csv",
    "readout_convention": "11_readout_convention.csv",
    "readout_coverage": "11_readout_coverage.csv",
    "readout_population": "11_readout_population.csv",
}

#: coverage grid; it starts at .30 so every dataset's floor (Ego4D hit rate
#: ~.36) crosses zero inside the plotted range
COVERAGE_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
#: target TRAIN hit rates of the population sweep
HIT_GRID = (0.70, 0.80, 0.90)
#: below this a partition cannot support the conformal step
MIN_ROWS = 30
#: coverage-sweep arms: the feature-free quantile and the full interface
SWEEP_ARMS = ("population_marginal", "full_observable")
#: the population sweep reports the feature-free floor only
POPULATION_ARMS = ("population_marginal",)
#: (name, target column, oracle GT-duration column). The oracle must reveal
#: the duration of the same instance the target scores.
CONVENTIONS = (("one_to_one", "q", "matched_gt_dur"),
               ("max", "max_tiou", "best_gt_dur"))


def _fit(X, y, tr, te):
    mdl = HistGradientBoostingRegressor(**GBR_PARAMS)
    mdl.fit(X[tr], y[tr])
    return mdl.predict(X[te])


def _design(frame, feats):
    if not feats:
        return np.empty((len(frame), 0))
    return frame[list(feats)].to_numpy(dtype=np.float64)


def _auc(labels, scores):
    """Rank AUC without a sklearn.metrics import; ties get mid-ranks."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# 1. attribution: A (full regressor) vs B (hit probability x constant)
# ---------------------------------------------------------------------------
def _attribution(ctx, identity):
    """A vs B (vs C, the two-part model) on the registered target.

    B is p_hat(x) * qbar_hit: the hit probability from the same features and
    the same squared-error loss, times a constant, so it carries no quality
    information among hits. A - B is what A gains from resolving quality.
    """
    pop, split, y = ctx.pop, ctx.split, ctx.y
    tr, te = ctx.tr, ctx.te
    arm = ctx.bundle.point["full_observable"]
    X = _design(pop, arm.check(pop))
    hit = y > 0
    qbar = float(y[tr & hit].mean())

    pred_a = _fit(X, y, tr, te)
    # squared-loss estimate of P(hit|x): same loss family as A
    p_hat = np.clip(_fit(X, hit.astype(np.float64), tr, te), 0.0, 1.0)
    m_hat = _fit(X, y, tr & hit, te)

    r2_a = r2_score(ctx.y_te, pred_a)
    r2_b = r2_score(ctx.y_te, p_hat * qbar)
    r2_c = r2_score(ctx.y_te, p_hat * m_hat)
    return [{
        **identity,
        "n_test": int(te.sum()), "n_test_hits": int(ctx.is_hit_te.sum()),
        "hit_rate_test": video_macro_mean(
            ctx.is_hit_te.astype(float), ctx.vids_test),
        "hit_rate_test_pooled": float(ctx.is_hit_te.mean()),
        "qbar_hit_train": qbar,
        "auc_hit": _auc(ctx.is_hit_te, p_hat),
        "r2_a_full_regressor": r2_a,
        "r2_b_weight_only": r2_b,
        "r2_c_two_part": r2_c,
        "gap_a_minus_b": r2_a - r2_b,
    }]


# ---------------------------------------------------------------------------
# 2. convention robustness: q_{1-to-1} versus q_{max}, fully paired
# ---------------------------------------------------------------------------
def _convention(ctx, identity):
    pop, split = ctx.pop, ctx.split
    tr, te = ctx.tr, ctx.te
    point_arm = ctx.bundle.point["full_observable"]
    mech_obs = ctx.bundle.mechanism[ctx.bundle.mech_obs]
    mech_orc = ctx.bundle.mechanism[ctx.bundle.mech_oracle]
    marginal = ctx.bundle.uncertainty["population_marginal"]
    X = _design(pop, point_arm.check(pop))
    pred_len = pop["pred_len"].to_numpy(dtype=np.float64)

    out = []
    for cname, ycol, durcol in CONVENTIONS:
        y = pop[ycol].to_numpy(dtype=np.float64)
        y_te = y[te]
        hit = y > 0
        qbar = float(y[tr & hit].mean())

        r2_all = r2_score(y_te, _fit(X, y, tr, te))
        p_hat = np.clip(_fit(X, hit.astype(np.float64), tr, te), 0.0, 1.0)
        r2_b = r2_score(y_te, p_hat * qbar)
        r2_c = r2_score(y_te, p_hat * _fit(X, y, tr & hit, te))

        # the mechanism pair on this convention's matched rows, with the
        # oracle naming this convention's instance
        sub = pop[hit].reset_index(drop=True)
        split_m = split[hit]
        y_m = y[hit]
        sub[ORACLE_FEATURE] = np.log(
            sub[durcol].to_numpy(dtype=np.float64) / pred_len[hit])
        tr_m, te_m = (split_m == 0), (split_m == 2)
        r2_m = r2_mo = np.nan
        if (np.isfinite(sub[ORACLE_FEATURE].to_numpy()).all()
                and tr_m.sum() > 10 and te_m.sum() > 10):
            y_m_te = y_m[te_m]
            r2_m = r2_score(y_m_te, _fit(
                _design(sub, mech_obs.check(sub)), y_m, tr_m, te_m))
            r2_mo = r2_score(y_m_te, _fit(
                _design(sub, mech_orc.check(sub)), y_m, tr_m, te_m))

        vids_te = pop.loc[te, "video-id"].astype(str).to_numpy()
        fit = one_sided_arm(pop, marginal, split, y, ctx.seed, alpha=ALPHA)
        fm = floor_metrics(floor_row_quantities(fit, y_te), y_te > 0, fit,
                           vids_te)
        out.append({
            **identity, "convention": cname, "target_column": ycol,
            "oracle_duration_column": durcol, "alpha": ALPHA,
            "n_test": int(te.sum()), "n_matched_test": int((y_te > 0).sum()),
            "hit_rate_test": video_macro_mean(
                (y_te > 0).astype(float), vids_te),
            "zero_mass_test": video_macro_mean(
                (y_te <= 0).astype(float), vids_te),
            "r2_all": r2_all, "r2_matched": r2_m,
            "r2_matched_oracle": r2_mo, "oracle_gap": r2_mo - r2_m,
            "r2_b_weight_only": r2_b, "r2_c_two_part": r2_c,
            "gap_a_minus_b": r2_all - r2_b,
            "floor_median": fm["floor_median"],
            "floor_coverage": fm["coverage"],
            "floor_coverage_video_macro": fm["coverage_video_macro"],
            "marginal_floor": fm["marginal_floor"],
            "positive_floor_rate": fm["positive_floor_rate"],
        })
    return out


# ---------------------------------------------------------------------------
# 3. coverage sweep: alpha moves, the population is held fixed
# ---------------------------------------------------------------------------
def _coverage(ctx, identity):
    pop, split, y, y_te = ctx.pop, ctx.split, ctx.y, ctx.y_te
    out = []
    for cov in COVERAGE_GRID:
        alpha = round(1.0 - cov, 10)
        for name in SWEEP_ARMS:
            arm = ctx.bundle.uncertainty[name]
            arm.check(pop)
            fit = one_sided_arm(pop, arm, split, y, ctx.seed, alpha=alpha)
            m = floor_metrics(floor_row_quantities(fit, y_te),
                              ctx.is_hit_te, fit, ctx.vids_test)
            out.append({
                **identity, "target_coverage": cov, "alpha": alpha,
                "is_registered_operating_point": bool(np.isclose(alpha, ALPHA)),
                "arm": name, "n_features": len(arm.features),
                "n_test": int(ctx.te.sum()),
                "hit_rate_test": video_macro_mean(
                    ctx.is_hit_te.astype(float), ctx.vids_test),
                **{k: m[k] for k in (
                    "coverage", "coverage_video_macro", "floor_mean",
                    "floor_median", "zero_floor_rate", "positive_floor_rate",
                    "floor_ge_half_rate", "marginal_floor", "delta_alpha",
                    "conformal_level", "n_cal_units", "head_status")},
            })
    return out


# ---------------------------------------------------------------------------
# 4. population sweep: the cutoff moves, coverage is held at the registered
#    operating point
# ---------------------------------------------------------------------------
def _threshold_for_hit(scores_tr, q_tr, target):
    """Deepest TRAIN prefix whose hit rate is >= target; None if unreachable."""
    order = np.argsort(-scores_tr, kind="stable")
    s, qq = scores_tr[order], q_tr[order]
    rate = np.cumsum(qq > 0) / np.arange(1, len(s) + 1)
    ends = np.flatnonzero(np.r_[s[:-1] > s[1:], True])
    ok = ends[rate[ends] >= target]
    return float(s[ok[-1]]) if len(ok) else None


def _population(ctx, identity):
    universe, gt = ctx.universe, ctx.gt
    split_u, train_vids = ctx.split_u, ctx.train_vids
    scores_u = universe["score"].to_numpy(np.float64)
    q_u = universe["q"].to_numpy(np.float64)
    tr_u = np.asarray(split_u == 0)
    te_u = np.asarray(split_u == 2)
    gt_videos = gt["video-id"].astype(str)
    uni_videos = universe["video-id"].astype(str)
    n_gt_test = int(gt_videos.isin(set(uni_videos[te_u])).sum())
    lrp_thr = ctx.ledgers["train"]["threshold"]

    plan = [("lrp", np.nan, lrp_thr)]
    plan += [("hit", h, _threshold_for_hit(scores_u[tr_u], q_u[tr_u], h))
             for h in HIT_GRID]

    out = []
    for kind, knob, thr in plan:
        base = {**identity, "rule": kind, "knob": knob, "threshold": thr,
                "alpha": ALPHA, "lrp_threshold": lrp_thr,
                "n_gt_test": n_gt_test}
        if thr is None:
            out.append({**base, "status": "unreachable_on_train", "arm": None})
            continue
        mask = scores_u >= thr
        split = split_u[mask]
        counts = {p: int((split == p).sum()) for p in (0, 1, 2)}
        if min(counts.values()) < MIN_ROWS:
            out.append({**base, "status": "too_few_rows", "arm": None,
                        "n_train": counts[0], "n_cal": counts[1],
                        "n_test": counts[2]})
            continue
        pop = universe[mask].reset_index(drop=True)
        pop["prior_ratio"], pop["prior_cdf"] = fold_clean_prior_features(
            pop, gt, train_vids)
        y = pop["q"].to_numpy(np.float64)
        te = (split == 2)
        y_te = y[te]
        is_hit_te = y_te > 0
        vids_te = pop.loc[te, "video-id"].astype(str).to_numpy()
        shared = {
            **base, "status": "ok",
            "n_train": counts[0], "n_cal": counts[1], "n_test": counts[2],
            "hit_rate_train": float((y[split == 0] > 0).mean()),
            # video-balanced: the mean over test videos of the within-video
            # matched share, the footing of the per-video conformal unit
            "hit_rate_test": video_macro_mean(is_hit_te.astype(float), vids_te),
            "hit_rate_test_pooled": float(is_hit_te.mean()),
            # matched test rows over GT instances in the test videos
            "recall_test": float(is_hit_te.sum() / max(n_gt_test, 1)),
        }
        for name in POPULATION_ARMS:
            arm = ctx.bundle.uncertainty[name]
            arm.check(pop)
            fit = one_sided_arm(pop, arm, split, y, ctx.seed, alpha=ALPHA)
            m = floor_metrics(floor_row_quantities(fit, y_te), is_hit_te, fit,
                              vids_te)
            out.append({**shared, "arm": name,
                        **{k: m[k] for k in (
                            "coverage", "coverage_video_macro", "floor_mean",
                            "floor_median", "positive_floor_rate",
                            "marginal_floor", "delta_alpha", "n_cal_units",
                            "head_status")}})
    return out


_BUILDERS = {
    "readout_attribution": _attribution,
    "readout_convention": _convention,
    "readout_coverage": _coverage,
    "readout_population": _population,
}


def build_readout(tasks, arms, log=print):
    """Every readout table over the readout arms among `tasks`.

    Returns {table name: DataFrame}; a table with no contributing task is
    absent rather than empty.
    """
    collected = {name: [] for name in READOUT_TABLES}
    todo = [t for t in tasks if t.dataset in READOUT_ARMS]
    if not todo:
        log("[quality_uncertainty/readout] no registered readout arm in this run")
        return {}
    for t in todo:
        spec = arms[t.dataset]
        ctx = build_head(t.dataset, t.seed,
                         feature_interface=t.feature_interface, log=lambda *a: None)
        identity = {
            "dataset": t.dataset, "seed": t.seed,
            "detector": t.detector,
            "canonical_dataset": t.canonical_dataset or t.dataset,
            "feature_interface": t.feature_interface,
            "population": POP.POPULATION_NAME, "target": POP.TARGET_NAME,
            "conformal": CONFORMAL_MODE,
            "conformal_unit": CONFORMAL_UNIT,
            "coverage_estimand": COVERAGE_ESTIMAND,
        }
        for name, fn in _BUILDERS.items():
            collected[name].extend(fn(ctx, identity))
        log(f"[{t.dataset}/readout/s{t.seed}] readout tables built "
            f"(interface {t.feature_interface}, detector {spec['detector']})")
    return {name: pd.DataFrame(rows)
            for name, rows in collected.items() if rows}
