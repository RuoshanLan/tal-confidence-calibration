"""Conformalization of the direct-q target (Sec. 5, Eq. 4; supp. Sec. E).

  * two-sided additive CQR       -> [L, U] at nominal 1 - alpha = .80, from the
    alpha/2 and 1 - alpha/2 quantile heads;
  * one-sided additive conformal -> floor L_q, the full alpha in the lower
    tail (the alpha quantile head, not the pair's alpha/2).

Both endpoints are clipped to the target support [0, 1]. Every function
returns the raw endpoints, the clipped endpoints and q_hat separately, so
degeneracy can be told apart from clipping.

``trivial_zero`` is not fitted: it returns [0, 0] and floor 0 with
q_hat = NaN and goes through the same metric code (the "[0,0]" control).

``head_status`` records what the quantile head delivered:

  not_fitted            the arm has no head (trivial_zero);
  marginal_constant     the head fell back to the constant TRAIN quantiles
                        (the "F" model in supp. Table S13);
  constant_non_marginal constant but not the marginal value;
  conditional           the head varies across rows (the "C" model).
"""
import numpy as np

from .estimators import fit_quantile_pair, fit_quantile_single

ALPHA = 0.20                       # nominal 80% (alpha = beta = .20)
TARGET_SUPPORT = (0.0, 1.0)        # q is a tIoU
CLIP_RULE = "full_support_both_endpoints"
CONFORMAL_MODE = "additive"
COVERAGE_ESTIMAND = "prediction_weighted_empirical"

# --------------------------------------------------------------------------
# Calibration unit
# --------------------------------------------------------------------------
# Split conformal needs exchangeable conformity scores, and predictions
# cluster by video (1.3 calibration rows per video on ActivityNet to 65 on
# EPIC verb), so the rank is taken over calibration VIDEOS: one uniformly
# drawn row per calibration video (single subsampling; Lee, Barber & Willett,
# Sec. 2.3.3). The guarantee is for a uniformly drawn row of a new video,
# reported as `coverage_video_macro` alongside the pooled rate.
CONFORMAL_UNIT = "video"
#: RNG stream for the one-per-video draw, derived from the split seed alone,
#: so a fixed population draws the same calibration rows at every alpha. The
#: value is an arbitrary fixed stream id.
CONFORMAL_SUBSAMPLE_STREAM = 20260824
#: the clustering column
GROUP_COLUMN = "video-id"


def finite_sample_level(n, alpha=ALPHA):
    return min(1.0, np.ceil((1 - alpha) * (n + 1)) / n)


def calibration_units(scores, groups, seed=0):
    """Reduce calibration conformity scores to one per calibration video.

    Returns ``(unit_scores, n_units, n_rows)``. The first occurrence of each
    group in a seeded permutation is kept, which is a uniform draw within
    every group.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_rows = int(scores.size)
    groups = np.asarray(groups)
    if groups.shape[0] != n_rows:
        raise ValueError(f"conformal grouping has {groups.shape[0]} labels "
                         f"for {n_rows} calibration rows")
    if n_rows == 0:
        return scores, 0, 0
    order = np.random.default_rng(
        [int(seed), CONFORMAL_SUBSAMPLE_STREAM]).permutation(n_rows)
    _, first = np.unique(groups[order], return_index=True)
    picked = np.sort(order[first])
    return scores[picked], int(picked.size), n_rows


def _design(pop, feats):
    if not feats:
        return np.empty((len(pop), 0))
    return pop[list(feats)].to_numpy(dtype=np.float64)


def _clip(values):
    lo, hi = TARGET_SUPPORT
    return np.clip(values, lo, hi)


def _violations(raw):
    lo, hi = TARGET_SUPPORT
    return float((raw < lo).mean()), float((raw > hi).mean())


HEAD_NOT_FITTED = "not_fitted"
HEAD_MARGINAL = "marginal_constant"
HEAD_CONSTANT_OTHER = "constant_non_marginal"
HEAD_CONDITIONAL = "conditional"
HEAD_STATUSES = (HEAD_NOT_FITTED, HEAD_MARGINAL, HEAD_CONSTANT_OTHER,
                 HEAD_CONDITIONAL)


def _head_status(heads, marginals):
    """Classify the fitted quantile head's TEST output (see module docstring)."""
    constant = all(len(h) and h.min() == h.max() for h in heads)
    if not constant:
        return HEAD_CONDITIONAL
    if all(np.isclose(h[0], m) for h, m in zip(heads, marginals)):
        return HEAD_MARGINAL
    return HEAD_CONSTANT_OTHER


def two_sided_arm(pop, arm, split, y, seed, alpha=ALPHA):
    """Fit on TRAIN, conformalize on CAL, evaluate TEST rows."""
    tr, ca, te = (split == 0), (split == 1), (split == 2)
    n_te = int(te.sum())
    if not arm.fitted:
        zeros = np.zeros(n_te)
        return {
            "raw_lo": zeros.copy(), "raw_hi": zeros.copy(),
            "lo": zeros.copy(), "hi": zeros.copy(),
            "qhat": np.nan, "level": np.nan, "n_cal": int(ca.sum()),
            "n_cal_units": int(ca.sum()), "conformal_unit": CONFORMAL_UNIT,
            "fitted": False, "head_status": HEAD_NOT_FITTED,
            "marginal_lo": np.nan, "marginal_hi": np.nan,
            "raw_lo_violation_rate": 0.0, "raw_hi_violation_rate": 0.0,
            "clip_rate": 0.0,
            "conformal": CONFORMAL_MODE, "coverage_estimand": COVERAGE_ESTIMAND,
        }

    m_lo = float(np.quantile(y[tr], alpha / 2))
    m_hi = float(np.quantile(y[tr], 1 - alpha / 2))
    if arm.estimator == "marginal":
        # the null: constant TRAIN quantiles, still conformalized on CAL
        lo_ca, hi_ca = np.full(int(ca.sum()), m_lo), np.full(int(ca.sum()), m_hi)
        lo_te, hi_te = np.full(n_te, m_lo), np.full(n_te, m_hi)
    else:
        X = _design(pop, arm.features)
        groups_tr = pop.loc[tr, "video-id"].to_numpy()
        lo_ca, hi_ca = fit_quantile_pair(X[tr], y[tr], X[ca], seed, groups_tr,
                                         alpha)
        lo_te, hi_te = fit_quantile_pair(X[tr], y[tr], X[te], seed, groups_tr,
                                         alpha)

    E = np.maximum(lo_ca - y[ca], y[ca] - hi_ca)
    E_u, n_units, n_rows = calibration_units(
        E, pop.loc[ca, GROUP_COLUMN].to_numpy(), seed)
    level = finite_sample_level(n_units, alpha)
    qhat = float(np.quantile(E_u, level, method="higher"))
    raw_lo, raw_hi = lo_te - qhat, hi_te + qhat
    lo_viol, _ = _violations(raw_lo)
    _, hi_viol = _violations(raw_hi)
    lo, hi = _clip(raw_lo), _clip(raw_hi)
    clipped = (raw_lo < TARGET_SUPPORT[0]) | (raw_hi > TARGET_SUPPORT[1])
    return {
        "raw_lo": raw_lo, "raw_hi": raw_hi, "lo": lo, "hi": hi,
        "qhat": qhat, "level": float(level), "n_cal": n_rows,
        "n_cal_units": n_units, "conformal_unit": CONFORMAL_UNIT,
        "fitted": True,
        "head_status": _head_status((lo_te, hi_te), (m_lo, m_hi)),
        "marginal_lo": m_lo, "marginal_hi": m_hi,
        "raw_lo_violation_rate": lo_viol, "raw_hi_violation_rate": hi_viol,
        "clip_rate": float(clipped.mean()),
        "conformal": CONFORMAL_MODE, "coverage_estimand": COVERAGE_ESTIMAND,
    }


def one_sided_arm(pop, arm, split, y, seed, alpha=ALPHA):
    """One-sided lower prediction bound on the same target/split as above."""
    tr, ca, te = (split == 0), (split == 1), (split == 2)
    n_te = int(te.sum())
    if not arm.fitted:
        zeros = np.zeros(n_te)
        return {
            "raw_floor": zeros.copy(), "floor": zeros.copy(),
            "qhat": np.nan, "level": np.nan, "n_cal": int(ca.sum()),
            "n_cal_units": int(ca.sum()), "conformal_unit": CONFORMAL_UNIT,
            "fitted": False, "head_status": HEAD_NOT_FITTED,
            "marginal_floor": np.nan,
            "raw_lo_violation_rate": 0.0, "raw_hi_violation_rate": 0.0,
            "clip_rate": 0.0,
            "conformal": CONFORMAL_MODE, "coverage_estimand": COVERAGE_ESTIMAND,
        }

    m_floor = float(np.quantile(y[tr], alpha))
    if arm.estimator == "marginal":
        lo_ca = np.full(int(ca.sum()), m_floor)
        lo_te = np.full(n_te, m_floor)
    else:
        X = _design(pop, arm.features)
        groups_tr = pop.loc[tr, "video-id"].to_numpy()
        lo_ca = fit_quantile_single(X[tr], y[tr], X[ca], seed, groups_tr, alpha)
        lo_te = fit_quantile_single(X[tr], y[tr], X[te], seed, groups_tr, alpha)

    E = lo_ca - y[ca]                       # one-sided conformity score
    E_u, n_units, n_rows = calibration_units(
        E, pop.loc[ca, GROUP_COLUMN].to_numpy(), seed)
    level = finite_sample_level(n_units, alpha)
    qhat = float(np.quantile(E_u, level, method="higher"))
    raw = lo_te - qhat
    lo_viol, hi_viol = _violations(raw)
    floor = _clip(raw)
    clipped = (raw < TARGET_SUPPORT[0]) | (raw > TARGET_SUPPORT[1])
    return {
        "raw_floor": raw, "floor": floor,
        "qhat": qhat, "level": float(level), "n_cal": n_rows,
        "n_cal_units": n_units, "conformal_unit": CONFORMAL_UNIT,
        "fitted": True,
        "head_status": _head_status((lo_te,), (m_floor,)),
        "marginal_floor": m_floor,
        "raw_lo_violation_rate": lo_viol, "raw_hi_violation_rate": hi_viol,
        "clip_rate": float(clipped.mean()),
        "conformal": CONFORMAL_MODE, "coverage_estimand": COVERAGE_ESTIMAND,
    }
