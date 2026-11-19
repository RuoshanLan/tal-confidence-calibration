"""Benchmark tasks: official one-to-one credit and localization quality.

Produces Table 1 and its supplementary tables and figure data:
mAP / pair summary, credit-target calibration metrics for every arm on both
populations, the bootstrap intervals, the per-video budget sweep, LaECE_0, the
LRP operating point and the THUMOS14 reliability bins (Fig. 2).

Pools are labelled with the official matcher only (``include_frag=False``),
and every run labels and calibrates from the raw predictions.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from ... import populations
from ...evaluation import evaluate_predictions
from ...matching import add_tiou_columns
from . import common, protocol
from .protocol import (
    ARMS, CALIB_TAUS, EPS, MAP_TAUS, OFFICIAL_POPULATION,
    calib_metrics, cap_per_video, safe_auroc, tau_key, tau_r, video_bootstrap,
)

#: The benchmark's pairs: the sixteen of Table 1 plus the 10-query
#: ActivityNet-1.3 TadTR (the Table 1 footnote), i.e. every registered pair,
#: in OFFICIAL_POPULATION order (which fixes the CSV row order).
BENCHMARK_PAIRS = list(OFFICIAL_POPULATION)

N_BOOT = 200
#: Arms receiving bootstrap CIs, at each dataset's primary tau. Confidences are
#: held fixed across resamples, so near perfect calibration the absolute value
#: in ECE puts every replicate above the estimate (supp. Sec. B, bootstrap table).
CI_ARMS = ["raw", "iso"]
CI_KEYS = ["ECE", "NLL"]
N_BINS = 25                       # LaECE_0 bins (08)
DILUTION_ARMS = ["raw"]
DILUTION_K_GRID = [5, 10, 25, 50, 100, 200, 500, 1000, 2000]
#: Fig. 2: the THUMOS14 reliability diagram at tau = 0.5, ten equal-width bins
#: as in ``calibration.ece``.
RELIABILITY_DATASET = "thumos"
RELIABILITY_TAU = 0.5
RELIABILITY_ARMS = ["raw", "iso"]
RELIABILITY_N_BINS = 10


# ==========================================================================
# expected-key grids (credit target only)
# ==========================================================================
def keys_benchmark_main(pairs=None, populations=None):
    pairs = pairs if pairs is not None else set(BENCHMARK_PAIRS)
    populations = populations if populations is not None else REPORTED_POPULATIONS
    return {(ds, mo, pop, tau_r(tau), arm, "credit")
            for (ds, mo) in pairs
            for pop in populations
            for tau in CALIB_TAUS[ds]
            for arm in ARMS}


def validate_dilution_grid(df, k_grid=None, expected_pairs=None):
    k_grid = k_grid or DILUTION_K_GRID
    expected_pairs = expected_pairs or set(BENCHMARK_PAIRS)
    pairs = set(map(tuple, df[["dataset", "model"]].drop_duplicates().to_numpy()))
    if pairs != expected_pairs:
        raise RuntimeError(f"dilution sweep pairs != benchmark pairs "
                           f"(missing {sorted(expected_pairs - pairs)})")
    if set(df["arm"]) != set(DILUTION_ARMS):
        raise RuntimeError(f"dilution sweep arms {sorted(set(df['arm']))} != "
                           f"{DILUTION_ARMS}")
    for (ds, mo), g in df.groupby(["dataset", "model"]):
        full = g[g["is_full_pool"].astype(bool)]
        full_ks = sorted(full["topk_per_video"].unique())
        if len(full_ks) != 1:
            raise RuntimeError(f"{ds}:{mo}: expected exactly one full-pool K, "
                               f"got {full_ks}")
        full_k = int(full_ks[0])
        if not (full["rows_per_video_max"] == full_k).all():
            raise RuntimeError(
                f"{ds}:{mo}: full-pool K={full_k} != native max rows/video "
                f"{sorted(full['rows_per_video_max'].unique())}")
        expected_ks = [k for k in k_grid if k < full_k] + [full_k]
        if sorted(g["topk_per_video"]) != expected_ks:
            raise RuntimeError(f"{ds}:{mo}: K set {sorted(g['topk_per_video'])} "
                               f"!= expected {expected_ks} (grid points below "
                               f"the native max + the exact full-pool point)")


# ==========================================================================
# pair context
# ==========================================================================
def build_pair_context(dataset, model, cfg_ds, pair_cfg, log=print):
    """Credit-labeled native + official pools for one pair, with the OOF
    confidence grid (every arm x tau) fitted on the official pool."""
    entry = OFFICIAL_POPULATION[(dataset, model)]

    loaded = common.load_pair(dataset, model, cfg_ds, pair_cfg)
    gt, pred = loaded["gt"], loaded["pred"]

    # The native pool is the detector's own export (capped at EVAL_TOPK for the
    # post-hoc-truncated pairs); only the budget sweep reads it, on raw scores.
    native_budget = protocol.EVAL_TOPK if entry["suffix"] else entry["budget_k"]
    native_pool = cap_per_video(pred, k=native_budget)
    native_df = common.labeled_pool(native_pool, gt, dataset, model,
                                    native_budget,
                                    include_frag=False, log=log)

    if entry["suffix"]:
        common.ensure_conf_columns(native_df, dataset, model,
                                   arms=DILUTION_ARMS, log=log)
        k = entry["budget_k"]
        official_pool = cap_per_video(pred, k=k)
        official_df = common.labeled_pool(official_pool, gt, dataset, model, k,
                                          include_frag=False, log=log)
    else:
        official_df = native_df
    common.ensure_conf_columns(official_df, dataset, model,
                               arms=ARMS, log=log)

    max_per_video = int(official_df.groupby("video-id").size().max())
    if max_per_video > entry["budget_k"]:
        raise RuntimeError(
            f"{dataset}:{model}: official pool has {max_per_video} rows/video, "
            f"declared budget_k={entry['budget_k']}")

    ctx = {
        "dataset": dataset, "model": model,
        "gt": gt, "native_df": native_df, "official_df": official_df,
        "coverage": loaded["coverage"],
        "official_pool": entry["pool"],
        "budget_k": entry["budget_k"], "source": entry["source"],
        "native_max_per_video": int(native_df.groupby("video-id").size().max()),
    }
    # Populations resolve lazily (`lrp_crossfit` needs an anchor-tau rematch);
    # mAP always uses `official_df`.
    ctx["population"] = populations.DEFAULT_POPULATION
    ctx["population_view"] = lambda name=None, **kw: populations.resolve(
        ctx, name or ctx["population"], log=log, **kw)
    return ctx


# ==========================================================================
# population views
# ==========================================================================
#: Populations calibration is reported on, in CSV row order: the per-video
#: top-K pool and the LRP operating point. mAP is computed on the official pool
#: only, since it integrates the whole PR curve.
REPORTED_POPULATIONS = ("official_topk", "lrp_crossfit")


def population_frame(ctx, name, log=print):
    """The pair's rows for population ``name``, with the arm x tau confidence
    grid refit on that population (a calibrator fitted on the whole pool would
    measure the deployed map, not the operating point). Memoized on ``ctx``."""
    ds, mo = ctx["dataset"], ctx["model"]
    mask, prov = ctx["population_view"](name)
    if name == "official_topk":
        return ctx["official_df"], prov

    frames = ctx.setdefault("population_frames", {})
    if name not in frames:
        # drop the confidences fitted on the official pool, then refit
        df = ctx["official_df"][mask].reset_index(drop=True)
        df = df.drop(columns=[c for c in df.columns if c.startswith("conf_")])
        common.ensure_conf_columns(df, ds, mo, arms=ARMS, log=log)
        frames[name] = df
    df = frames[name]
    log(f"  [{ds}:{mo}] population {name}: {len(df)} rows "
        f"(thr {prov.get('threshold_min', float('nan')):.3f}-"
        f"{prov.get('threshold_max', float('nan')):.3f})")
    return df, prov


def benchmark(ctx, log=print):
    """Calibration rows per population and the pair summary on the official
    pool (01)."""
    ds, mo = ctx["dataset"], ctx["model"]
    df, gt, pool = ctx["official_df"], ctx["gt"], ctx["official_pool"]

    map_row = evaluate_predictions(df, gt, MAP_TAUS[ds])
    pair_row = {
        "dataset": ds, "model": mo, "pool": pool,
        "eval_split": protocol.EVAL_SPLIT[ds], **ctx["coverage"],
        "n_gt_instances": int(len(gt)),
        "n_preds": int(len(df)), "n_videos": int(df["video-id"].nunique()),
        "n_classes": int(gt["label"].nunique()),
        **map_row,
    }
    log(f"[{ds}:{mo}] avg mAP={map_row['average_mAP']:.2f}")

    metric_rows = []
    for popname in REPORTED_POPULATIONS:
        pdf, prov = population_frame(ctx, popname, log=log)
        plabels = pdf["label"].to_numpy()
        for tau in CALIB_TAUS[ds]:
            y_credit = pdf[f"tp_credit_{tau_key(tau)}"].to_numpy(dtype=np.int64)
            for arm in ARMS:
                conf = pdf[f"conf_{arm}_{tau_key(tau)}"].to_numpy(dtype=np.float64)
                row = {"dataset": ds, "model": mo, "pool": pool,
                       "population": popname,
                       "population_n": int(prov["population_n"]),
                       "tau": tau, "arm": arm, "target": "credit",
                       **calib_metrics(y_credit, conf, plabels)}
                if arm == "raw":
                    row["auroc"] = safe_auroc(y_credit, conf)
                metric_rows.append(row)
    return pair_row, metric_rows


def bootstrap_ci(ctx, log=print):
    """Video cluster-bootstrap CIs at the dataset's primary tau, per reported
    population (05).

    Videos are resampled within each population, which is held fixed (the LRP
    threshold is not refit inside the bootstrap). Every call reseeds.
    """
    ds, mo = ctx["dataset"], ctx["model"]
    pool = ctx["official_pool"]

    rows = []
    for popname in REPORTED_POPULATIONS:
        pdf, prov = population_frame(ctx, popname, log=log)
        videos = pdf["video-id"].to_numpy()
        labels = pdf["label"].to_numpy()
        for tau in [protocol.HEADLINE_TAU[ds]]:
            y = pdf[f"tp_credit_{tau_key(tau)}"].to_numpy(dtype=np.int64)
            for arm in CI_ARMS:
                conf = pdf[f"conf_{arm}_{tau_key(tau)}"].to_numpy(dtype=np.float64)
                point = calib_metrics(y, conf, labels)

                def stat(idx, _y=y, _conf=conf, _labels=labels):
                    m = calib_metrics(_y[idx], _conf[idx], _labels[idx])
                    return {k: m[k] for k in CI_KEYS}

                t0 = time.time()
                reps = video_bootstrap(videos, stat, n_boot=N_BOOT)
                row = {"dataset": ds, "model": mo, "pool": pool,
                       "population": popname,
                       "population_n": int(prov["population_n"]),
                       "tau": tau, "arm": arm, "n_boot": N_BOOT,
                       "mask_fixed_within_bootstrap": True}
                for k in CI_KEYS:
                    row[k] = point[k]
                    row[f"{k}_lo"] = float(np.percentile(reps[k], 2.5))
                    row[f"{k}_hi"] = float(np.percentile(reps[k], 97.5))
                rows.append(row)
                log(f"[{ds}:{mo}] {popname} tau={tau:g} {arm:<9s} "
                    f"ECE={row['ECE']:.3f} "
                    f"[{row['ECE_lo']:.3f},{row['ECE_hi']:.3f}] "
                    f"({time.time()-t0:.0f}s)")
    return rows


def population_dilution(ctx, k_grid=None, log=print):
    """Per-video budget sweep on the NATIVE pool at the primary tau (02).

    The full-pool row (``is_full_pool``) records each detector's native
    per-video output cap.
    """
    k_grid = k_grid or DILUTION_K_GRID
    ds, mo = ctx["dataset"], ctx["model"]
    df = ctx["native_df"]
    tau = protocol.HEADLINE_TAU[ds]
    ycol = f"tp_credit_{tau_key(tau)}"

    max_per_video = ctx["native_max_per_video"]
    ks = [k for k in k_grid if k < max_per_video] + [max_per_video]

    rows = []
    for k in ks:
        sub = cap_per_video(df, k=k)
        per_video = sub.groupby("video-id").size()
        shared = {
            "dataset": ds, "model": mo, "tau": tau,
            "pool": "native", "topk_per_video": k,
            "is_full_pool": bool(k == max_per_video),
            "rows_per_video_mean": float(per_video.mean()),
            "rows_per_video_max": int(per_video.max()),
        }
        y = sub[ycol].to_numpy()
        labels = sub["label"].to_numpy()
        for arm in DILUTION_ARMS:
            m = calib_metrics(y, sub[f"conf_{arm}_{tau_key(tau)}"].to_numpy(), labels)
            rows.append({**shared, "arm": arm, **m})
            log(f"[{ds}:{mo}] {arm:<9s} K={k:<5d} n={m['n']:>8d} "
                f"ECE={m['ECE']:.3f} tp_rate={m['tp_rate']:.4f}")
    return rows


def _laece0(conf, q, labels, n_bins=N_BINS):
    """(class-macro, pooled) LaECE_0 of ``conf`` against the quality target
    ``q`` over ``n_bins`` equal-width bins, as fractions."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    errs = []
    for cls in np.unique(labels):
        m = labels == cls
        nc = m.sum()
        err = 0.0
        for b in np.unique(bin_id[m]):
            mb = m & (bin_id == b)
            err += (mb.sum() / nc) * abs(conf[mb].mean() - q[mb].mean())
        errs.append(err)
    pooled = 0.0
    for b in np.unique(bin_id):
        mb = bin_id == b
        pooled += (mb.sum() / len(conf)) * abs(conf[mb].mean() - q[mb].mean())
    return float(np.mean(errs)), float(pooled)


def _oof_iso_q(scores, q, videos):
    """Out-of-fold isotonic regression of ``q`` on the score, on the same
    video-grouped folds as ``protocol.video_folds``."""
    out = np.full(len(scores), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(scores, q, videos):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(scores[tr], q[tr])
        out[te] = iso.predict(scores[te])
    return np.clip(out, 0.0, 1.0)


def laece_reveal(ctx, log=print):
    """LaECE_0 raw vs OOF isotonic, per reported population (08).

    The pooled value is the headline. The class-macro value is written too, with
    the per-class row counts that show why it is unreliable: on TAL's long-tail
    label space many classes hold a handful of rows, and a one-row class's error
    is |conf - q|, noise biased high.
    """
    ds, mo = ctx["dataset"], ctx["model"]
    pool = ctx["official_pool"]
    rows = []
    for popname in REPORTED_POPULATIONS:
        df, prov = population_frame(ctx, popname, log=log)
        scores = np.clip(df["score"].to_numpy(np.float64), EPS, 1 - EPS)
        q = df["matched_tiou0"].to_numpy(np.float64)
        labels = df["label"].to_numpy()
        videos = df["video-id"].to_numpy()
        matched = q > 0

        qhat = _oof_iso_q(scores, q, videos)
        raw_m, raw_p = _laece0(scores, q, labels)
        iso_m, iso_p = _laece0(qhat, q, labels)
        per_class = pd.Series(labels).value_counts()
        rows.append({
            "dataset": ds, "model": mo, "pool": pool,
            "population": popname, "n": len(df),
            "match_rate": float(matched.mean()),
            "pooled_LaECE0_raw": raw_p, "pooled_LaECE0_iso": iso_p,
            "LaECE0_macro_raw": raw_m, "LaECE0_macro_iso": iso_m,
            "n_classes_present": int(per_class.size),
            "rows_per_class_median": float(per_class.median()),
            "rows_per_class_min": int(per_class.min()),
            "n_classes_under_50_rows": int((per_class < 50).sum()),
        })
        log(f"[{ds}:{mo}] {popname} pooled LaECE0 {raw_p:.4f} -> {iso_p:.4f} "
            f"(macro {raw_m:.4f} -> {iso_m:.4f}; "
            f"{rows[-1]['n_classes_under_50_rows']}/{per_class.size} classes <50 rows)")
    return rows


def lrp_sweep(scores, matched_tiou, n_gt, tau):
    """oLRP over every score cutoff; returns (oLRP, threshold, TP, FP, FN) at
    the minimum. A row is a TP when ``matched_tiou > 0`` (matching locked at
    ``tau``), with localisation error (1 - tIoU) / (1 - tau). Cutoffs fall only
    between distinct scores, so tied rows enter together."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    q = matched_tiou[order]
    is_tp = (q > 0).astype(np.float64)
    loc_err = np.where(is_tp > 0, (1.0 - q) / (1.0 - tau), 0.0)
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.arange(1, len(s) + 1) - cum_tp
    cum_loc = np.cumsum(loc_err)
    ends = np.flatnonzero(np.r_[s[1:] < s[:-1], True])
    tp, fp = cum_tp[ends], cum_fp[ends]
    fn = float(n_gt) - tp
    lrp = (cum_loc[ends] + fp + fn) / (tp + fp + fn)
    i = int(np.argmin(lrp))
    e = ends[i]
    return (float(lrp[i]), float(s[e]),
            int(cum_tp[e]), int(cum_fp[e]), int(n_gt - cum_tp[e]))


def lrp_operating_point(ctx, log=print):
    """The cross-fitted LRP operating point on the official pool (10): one
    threshold per video fold, each fitted on the other folds at the dataset's
    primary tau and held fixed across the tau grid."""
    ds, mo = ctx["dataset"], ctx["model"]
    tau = protocol.HEADLINE_TAU[ds]
    df, gt, pool = ctx["official_df"], ctx["gt"], ctx["official_pool"]

    # The threshold is fitted against the anchor-tau matching; it must agree
    # with the credit labels every metric is computed on.
    dfm = add_tiou_columns(df, gt, lock_tau=tau)
    ycredit = dfm[f"tp_credit_{tau_key(tau)}"].to_numpy(np.int64)
    ymatch = (dfm["matched_tiou"].to_numpy(np.float64) > 0).astype(np.int64)
    if not (ycredit == ymatch).all():
        raise RuntimeError(
            f"{ds}:{mo}: {(ycredit != ymatch).sum()} rows where the LRP matching "
            f"disagrees with tp_credit_{tau_key(tau)}")
    row = {"dataset": ds, "model": mo, "tau": tau, "pool": pool}
    _, xprov = ctx["population_view"]("lrp_crossfit")
    fold_thr = xprov.get("fold_thresholds", {})
    row.update({
        "crossfit_threshold_mean": xprov.get("threshold_mean"),
        "crossfit_threshold_min": xprov.get("threshold_min"),
        "crossfit_threshold_max": xprov.get("threshold_max"),
        "crossfit_n_folds": xprov.get("n_splits"),
        "crossfit_n_keep": xprov.get("population_n"),
        "crossfit_anchor_tau": xprov.get("anchor_tau"),
        "crossfit_thresholds": ";".join(
            f"{k}:{v['threshold']:.6f}" for k, v in sorted(fold_thr.items())),
    })
    log(f"[{ds}:{mo}] tau={tau:g} LRP thresholds "
        f"{row['crossfit_threshold_min']:.4f}-{row['crossfit_threshold_max']:.4f} "
        f"keep={row['crossfit_n_keep']}")
    return row


def reliability_bins(ctx, log=print):
    """Fig. 2 data: per-bin mean confidence, TP rate and mass for the THUMOS14
    reliability diagram, on both reported populations and the raw / isotonic
    arms, at tau = 0.5 (13).

    The bins are ``calibration.ece``'s ten equal-width bins (left-closed, the
    last one closed at 1), so sum(share * |mean_conf - tp_rate|) over a panel's
    bins is that panel's ECE in 01. Empty bins keep a row with n = 0.
    """
    ds, mo = ctx["dataset"], ctx["model"]
    if ds != RELIABILITY_DATASET:
        return []
    tk = tau_key(RELIABILITY_TAU)
    edges = np.arange(0, 1, 1.0 / RELIABILITY_N_BINS)
    rows = []
    for popname in REPORTED_POPULATIONS:
        df, _ = population_frame(ctx, popname, log=log)
        y = df[f"tp_credit_{tk}"].to_numpy(dtype=np.float64)
        for arm in RELIABILITY_ARMS:
            conf = np.clip(df[f"conf_{arm}_{tk}"].to_numpy(dtype=np.float64),
                           EPS, 1.0 - EPS)
            bin_id = np.searchsorted(edges, conf, side="right") - 1
            for b in range(RELIABILITY_N_BINS):
                m = bin_id == b
                n = int(m.sum())
                rows.append({
                    "dataset": ds, "model": mo, "population": popname,
                    "tau": RELIABILITY_TAU, "arm": arm, "bin": b,
                    "bin_lo": b / RELIABILITY_N_BINS,
                    "bin_hi": (b + 1) / RELIABILITY_N_BINS,
                    "n": n, "share": n / len(conf),
                    "mean_conf": float(conf[m].mean()) if n else np.nan,
                    "tp_rate": float(y[m].mean()) if n else np.nan,
                })
    return rows
