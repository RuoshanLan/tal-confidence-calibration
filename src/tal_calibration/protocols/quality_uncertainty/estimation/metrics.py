"""Metric families of the chain.

Point: TEST R^2 (primary), MAE, RMSE.
Interval: coverage (pooled and video-balanced), delta alpha, widths, the
interval (Winkler) score, degeneracy rates, raw endpoint violations, clipping.
Floor: coverage, mean/median floor, zero/positive floor rates.

Matched/unmatched subgroup quantities are descriptive only: conditioning on
hit status is conditioning on ground truth, so no subgroup guarantee is
claimed.
"""
import numpy as np

from .conformal import ALPHA, TARGET_SUPPORT

EXACT_ZERO_TOL = 1e-12
NEAR_ZERO_LO, NEAR_ZERO_HI = 0.01, 0.05
NEAR_FULL_LO, NEAR_FULL_HI = 0.05, 0.95
FLOOR_HIGH = 0.5
#: fixed degeneracy thresholds
DEGENERACY_THRESHOLDS = {
    "exact_zero_tol": EXACT_ZERO_TOL,
    "near_zero": [NEAR_ZERO_LO, NEAR_ZERO_HI],
    "near_full_support": [NEAR_FULL_LO, NEAR_FULL_HI],
    "floor_high": FLOOR_HIGH,
}


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def r2_score(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return np.nan
    return float(1.0 - ((y - pred) ** 2).sum() / ss_tot)


def point_metrics(y, pred):
    """R^2 (primary) plus MAE and RMSE in tIoU units."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    resid = y - pred
    return {
        "r2": r2_score(y, pred),
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
    }


# --------------------------------------------------------------------------
# interval metrics
# --------------------------------------------------------------------------
def interval_score(lo, hi, y, alpha=ALPHA):
    """Winkler / interval score, per row. Lower is better.

        S_alpha(L,U;y) = (U-L) + (2/alpha)(L-y)1{y<L} + (2/alpha)(y-U)1{y>U}
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return ((hi - lo)
            + (2.0 / alpha) * (lo - y) * (y < lo)
            + (2.0 / alpha) * (y - hi) * (y > hi))


def _mean(v, mask=None):
    v = np.asarray(v, dtype=np.float64)
    if mask is not None:
        v = v[mask]
    return float(v.mean()) if v.size else np.nan


def _median(v, mask=None):
    v = np.asarray(v, dtype=np.float64)
    if mask is not None:
        v = v[mask]
    return float(np.median(v)) if v.size else np.nan


def video_macro_mean(values, vids):
    """Mean over TEST videos of the within-video mean: the video-balanced
    rate the per-video conformal guarantee targets (``coverage`` is the
    pooled readout)."""
    values = np.asarray(values, dtype=np.float64)
    if vids is None or values.size == 0:
        return np.nan
    vids = np.asarray(vids)
    if vids.shape[0] != values.size:
        raise ValueError(f"video-macro mean got {vids.shape[0]} ids for "
                         f"{values.size} rows")
    order = np.argsort(vids, kind="stable")
    _, start = np.unique(vids[order], return_index=True)
    return float(np.mean([seg.mean()
                          for seg in np.split(values[order], start[1:])]))


def interval_row_quantities(fit, y_test, alpha=ALPHA):
    """Per-row arrays every interval metric and bootstrap is derived from."""
    lo, hi = fit["lo"], fit["hi"]
    covered = (y_test >= lo) & (y_test <= hi)
    width = hi - lo
    return {
        "covered": covered.astype(np.float64),
        "width": width,
        "iscore": interval_score(lo, hi, y_test, alpha),
        "exact_zero": ((np.abs(lo) <= EXACT_ZERO_TOL)
                       & (np.abs(hi) <= EXACT_ZERO_TOL)).astype(np.float64),
        "near_zero": ((lo <= NEAR_ZERO_LO)
                      & (hi <= NEAR_ZERO_HI)).astype(np.float64),
        "near_full": ((lo <= NEAR_FULL_LO)
                      & (hi >= NEAR_FULL_HI)).astype(np.float64),
    }


def floor_row_quantities(fit, y_test):
    """Per-row arrays every floor metric and bootstrap is derived from."""
    floor = fit["floor"]
    return {
        "covered": (y_test >= floor).astype(np.float64),
        "floor": floor,
        "zero_floor": (floor <= TARGET_SUPPORT[0] + EXACT_ZERO_TOL
                       ).astype(np.float64),
        "positive_floor": (floor > TARGET_SUPPORT[0] + EXACT_ZERO_TOL
                           ).astype(np.float64),
        "floor_ge_high": (floor >= FLOOR_HIGH).astype(np.float64),
    }


def interval_metrics(rq, is_hit, fit, vids=None):
    """Interval metrics for one arm on one (dataset, seed) test split."""
    hit = np.asarray(is_hit, dtype=bool)
    unm = ~hit
    qhat = fit["qhat"]
    return {
        "coverage": _mean(rq["covered"]),
        "coverage_video_macro": video_macro_mean(rq["covered"], vids),
        "coverage_hit": _mean(rq["covered"], hit),
        "coverage_unmatched": _mean(rq["covered"], unm),
        "delta_alpha": float(qhat) if np.isfinite(qhat) else np.nan,
        "delta_alpha_is_zero": bool(np.isfinite(qhat) and qhat == 0.0),
        "conformal_level": fit["level"],
        "conformal_unit": fit.get("conformal_unit"),
        "n_cal_units": fit.get("n_cal_units"),
        "fitted": bool(fit["fitted"]),
        "head_status": fit["head_status"],
        "marginal_lo": fit["marginal_lo"], "marginal_hi": fit["marginal_hi"],
        "mean_width": _mean(rq["width"]),
        "median_width": _median(rq["width"]),
        "mean_width_hit": _mean(rq["width"], hit),
        "median_width_hit": _median(rq["width"], hit),
        "mean_width_unmatched": _mean(rq["width"], unm),
        "exact_zero_rate": _mean(rq["exact_zero"]),
        "near_zero_rate": _mean(rq["near_zero"]),
        "near_full_support_rate": _mean(rq["near_full"]),
        "raw_lo_violation_rate": fit["raw_lo_violation_rate"],
        "raw_hi_violation_rate": fit["raw_hi_violation_rate"],
        "clip_rate": fit["clip_rate"],
        "interval_score": _mean(rq["iscore"]),
        "interval_score_hit": _mean(rq["iscore"], hit),
        "interval_score_unmatched": _mean(rq["iscore"], unm),
    }


def floor_metrics(rq, is_hit, fit, vids=None):
    """Floor metrics for one arm on one (dataset, seed) test split."""
    hit = np.asarray(is_hit, dtype=bool)
    unm = ~hit
    qhat = fit["qhat"]
    return {
        "coverage": _mean(rq["covered"]),
        "coverage_video_macro": video_macro_mean(rq["covered"], vids),
        "coverage_hit": _mean(rq["covered"], hit),
        "coverage_unmatched": _mean(rq["covered"], unm),
        "delta_alpha": float(qhat) if np.isfinite(qhat) else np.nan,
        "delta_alpha_is_zero": bool(np.isfinite(qhat) and qhat == 0.0),
        "conformal_level": fit["level"],
        "conformal_unit": fit.get("conformal_unit"),
        "n_cal_units": fit.get("n_cal_units"),
        "fitted": bool(fit["fitted"]),
        "head_status": fit["head_status"],
        "marginal_floor": fit["marginal_floor"],
        "floor_mean": _mean(rq["floor"]),
        "floor_median": _median(rq["floor"]),
        "floor_mean_hit": _mean(rq["floor"], hit),
        "floor_median_hit": _median(rq["floor"], hit),
        "zero_floor_rate": _mean(rq["zero_floor"]),
        "positive_floor_rate": _mean(rq["positive_floor"]),
        "floor_ge_half_rate": _mean(rq["floor_ge_high"]),
        "positive_floor_rate_hit": _mean(rq["positive_floor"], hit),
        "raw_lo_violation_rate": fit["raw_lo_violation_rate"],
        "raw_hi_violation_rate": fit["raw_hi_violation_rate"],
        "clip_rate": fit["clip_rate"],
    }


#: rate-valued columns that must lie in [0,1] (contract check)
RATE_COLUMNS = (
    "coverage", "coverage_video_macro", "coverage_hit", "coverage_unmatched",
    "exact_zero_rate", "near_zero_rate", "near_full_support_rate",
    "raw_lo_violation_rate", "raw_hi_violation_rate", "clip_rate",
    "zero_floor_rate", "positive_floor_rate", "floor_ge_half_rate",
    "positive_floor_rate_hit",
)
