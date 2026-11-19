"""Fragmentation-chain tasks: official vs fragment-tolerant matching.

Every pool is labeled twice by ``protocol.label_multi_tau``: the official
one-to-one credit target ``tp_credit`` and the fragment-tolerant target
``tp_frag`` (strict two-pass, supp. Algorithm S1). Two pooled-isotonic
calibrator families are fitted out of fold on the same video folds: one on
``tp_credit`` (``conf_iso_{tau}``) and one on ``tp_frag``
(``conf_ffit_iso_{tau}``). Both are evaluated against ``tp_frag``:

    ece_cfit_yfrag   credit-fitted confidences vs y_frag
    ece_ffit_yfrag   frag-fitted confidences vs y_frag
    delta_match_ece  = ece_cfit_yfrag - ece_ffit_yfrag   (positive = refit helps)

with a paired video-clustered bootstrap CI on ``delta_match_ece``. The chain
also counts the rescued fragments per (pair, population, tau) and extracts the
real instance drawn in Fig. 1(b).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# aliased: `build_pair_contexts` takes a parameter literally named
# `populations` (the tuple of names to build), which would shadow the
# module inside that function.
from ... import populations as pop_registry
from ...calibration import ece
from ...geometry import temporal_iou_with_group
from ...matching import add_tiou_columns, load_label_map
from . import common, protocol
from .protocol import (
    CALIB_TAUS, EPS, OFFICIAL_POPULATION, TABLE1_PAIRS, SEED, cap_per_video, tau_key, tau_r,
)

#: The only calibrator: pooled isotonic, the arm the paper reports.
ARM = "iso"
#: Reported populations, in CSV row order: the official top-K pool and the
#: LRP operating point.
POPULATIONS = ("official", "lrp_crossfit")

#: Paired video-clustered bootstrap on delta_match_ece.
N_BOOT = 2000
BOOT_SEED = SEED

#: The real instance of Fig. 1(b): Ego4D-MQ, ActionFormer, top-K, tau = .1.
EXAMPLE = {"dataset": "ego4d", "model": "actionformer", "population": "official",
           "video": "088ae5ff-ae50-40bd-bea9-7bbdaab0c29b",
           "label": "peel_a_fruit_or_vegetable", "tau": 0.1}


def keys_matching_summary(pairs=None, populations=POPULATIONS):
    pairs = pairs if pairs is not None else set(TABLE1_PAIRS)
    return {(ds, mo, pop, tau_r(tau), ARM)
            for (ds, mo) in pairs
            for pop in populations
            for tau in CALIB_TAUS[ds]}


# ==========================================================================
# pair contexts (one per population)
# ==========================================================================
def build_pair_contexts(dataset, model, cfg_ds, pair_cfg,
                        populations=POPULATIONS, log=print):
    """One labeled and calibrated context per population. The LRP population
    truncates the raw official pool before matching; matching and OOF
    calibration then run on the truncated pool against the full GT."""
    entry = OFFICIAL_POPULATION[(dataset, model)]

    loaded = common.load_pair(dataset, model, cfg_ds, pair_cfg)
    gt, pred = loaded["gt"], loaded["pred"]
    k = entry["budget_k"]
    official_raw = cap_per_video(pred, k=k)

    ctxs = []
    for pop in populations:
        if pop == "official":
            pool = official_raw
        elif pop == "lrp_crossfit":
            # Rows above the cross-fitted oLRP threshold at the anchor tau (the
            # middle of the dataset's tau grid); the same rows are used at every
            # tau, and `labeled_pool` below relabels the truncated pool.
            anchor = protocol.HEADLINE_TAU[dataset]
            # add_tiou_columns REORDERS rows (it groups by (video, label)), so
            # the mask it produces is aligned to `dfm`, not to `official_raw`.
            # Map back by source_row_id -- never by position.
            dfm = add_tiou_columns(official_raw, gt, lock_tau=anchor)
            mask, _ = pop_registry.select("lrp_crossfit", dfm, gt, tau=anchor)
            keep_ids = set(dfm.loc[mask, "source_row_id"])
            pool = official_raw[
                official_raw["source_row_id"].isin(keep_ids)
            ].reset_index(drop=True)
            if len(pool) != int(mask.sum()):
                raise RuntimeError(
                    f"{dataset}:{model}: lrp_crossfit mapped {len(pool)} rows "
                    f"back from a {int(mask.sum())}-row selection")
        else:
            raise ValueError(f"unknown population {pop!r}")

        df = common.labeled_pool(pool, gt, dataset, model, k,
                                 include_frag=True, log=log)
        common.ensure_conf_columns(df, dataset, model, arms=[ARM], log=log)
        # frag-fitted family: same rows, same video folds (protocol.video_folds
        # is a pure function of the video column), target = tp_frag.
        common.ensure_conf_columns(df, dataset, model, arms=[ARM], log=log,
                                   target_prefix="tp_frag", conf_prefix="conf_ffit")
        ctxs.append({
            "dataset": dataset, "model": model, "population": pop,
            "gt": gt, "df": df, "pool": entry["pool"], "anno": loaded["anno"],
            "population_n": int(len(df)),
        })
    return ctxs


# ==========================================================================
# analyses
# ==========================================================================
def _ece_pct(y, conf):
    """Pooled ECE in percent (10 equal-width bins), the benchmark's ECE."""
    return 100.0 * ece(y, np.clip(conf, EPS, 1.0 - EPS))


def matching_audit(ctx, log=print):
    """Rescued-fragment counts and the transfer-refit ECE gap for one
    (pair, population). Returns (count_rows, summary_rows), one row per tau."""
    ds, mo, pop = ctx["dataset"], ctx["model"], ctx["population"]
    df, pool = ctx["df"], ctx["pool"]

    count_rows, summary_rows = [], []
    for tau in CALIB_TAUS[ds]:
        tk = tau_key(tau)
        y_credit = df[f"tp_credit_{tk}"].to_numpy(dtype=np.int64)
        y_frag = df[f"tp_frag_{tk}"].to_numpy(dtype=np.int64)
        count_rows.append({
            "dataset": ds, "model": mo, "pool": pool, "population": pop, "tau": tau,
            "n_predictions": int(len(df)),
            "n_tp_official": int(y_credit.sum()),
            "n_rescued": int(((y_frag == 1) & (y_credit == 0)).sum()),
        })
        ece_c = _ece_pct(y_frag, df[f"conf_{ARM}_{tk}"].to_numpy(dtype=np.float64))
        ece_f = _ece_pct(y_frag, df[f"conf_ffit_{ARM}_{tk}"].to_numpy(dtype=np.float64))
        summary_rows.append({
            "dataset": ds, "model": mo, "pool": pool, "tau": tau, "arm": ARM,
            "population": pop,
            "population_n": ctx["population_n"],
            "ece_cfit_yfrag": ece_c,
            "ece_ffit_yfrag": ece_f,
            "delta_match_ece": ece_c - ece_f,
        })
        log(f"[{ds}:{mo}:{pop}] tau={tau:g} rescued={count_rows[-1]['n_rescued']} "
            f"dMatch={ece_c - ece_f:+.3f}")
    return count_rows, summary_rows


def _video_groups(videos):
    """Row-index arrays per unique video, O(n log n)."""
    videos = np.asarray(videos)
    order = np.argsort(videos, kind="mergesort")
    uniq, starts = np.unique(videos[order], return_index=True)
    bounds = np.append(starts, len(order))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def bootstrap_audit(ctx, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% paired video-clustered percentile bootstrap of delta_match_ece.

    Videos are resampled with replacement; confidences and the population stay
    fixed (no refit), and one replicate index feeds every tau. Returns
    ``{tau_r(tau): {delta_match_ece_ci_lo, delta_match_ece_ci_hi}}``.
    """
    df = ctx["df"]
    taus = CALIB_TAUS[ctx["dataset"]]
    groups = _video_groups(df["video-id"].to_numpy())
    n_videos = len(groups)

    per_tau = []
    for tau in taus:
        tk = tau_key(tau)
        per_tau.append((df[f"tp_frag_{tk}"].to_numpy(dtype=np.float64),
                        df[f"conf_{ARM}_{tk}"].to_numpy(dtype=np.float64),
                        df[f"conf_ffit_{ARM}_{tk}"].to_numpy(dtype=np.float64)))

    d_match = np.empty((len(taus), n_boot))
    rng = np.random.RandomState(seed)
    for b in range(n_boot):
        # one draw of n_videos per replicate: the frozen CIs depend on this
        # exact RandomState call sequence
        sample = rng.randint(0, n_videos, size=n_videos)
        idx = np.concatenate([groups[i] for i in sample])
        for ti, (yf, cc, cf) in enumerate(per_tau):
            d_match[ti, b] = _ece_pct(yf[idx], cc[idx]) - _ece_pct(yf[idx], cf[idx])

    ci_map = {}
    for ti, tau in enumerate(taus):
        lo, hi = np.percentile(d_match[ti], [2.5, 97.5])
        ci_map[tau_r(tau)] = {"delta_match_ece_ci_lo": float(lo),
                              "delta_match_ece_ci_hi": float(hi)}
    return ci_map


def fragment_example(ctx):
    """The Fig. 1(b) instance: every GT of ``EXAMPLE``'s (video, class) and
    every prediction reaching tIoU >= tau with one of them, with both labels
    and its tIoU with each GT. GT rows give the tIoU between GTs (the released
    Ego4D annotation is an annotator union, so near-duplicates appear)."""
    tau, tk = EXAMPLE["tau"], tau_key(EXAMPLE["tau"])
    label = load_label_map(Path(ctx["anno"]))[EXAMPLE["label"]]
    gt, df = ctx["gt"], ctx["df"]
    g = gt[(gt["video-id"] == EXAMPLE["video"]) & (gt["label"] == label)]
    gs = g["t-start"].to_numpy(dtype=np.float64)
    ge = g["t-end"].to_numpy(dtype=np.float64)
    p = df[(df["video-id"] == EXAMPLE["video"]) & (df["label"] == label)]
    p = p.sort_values(["score", "t-start", "t-end", "source_row_id"],
                      ascending=[False, True, True, True], kind="mergesort")

    rows = []
    for i, (s, e) in enumerate(zip(gs, ge)):
        rows.append({"role": "gt", "id": i, "t_start": s, "t_end": e,
                     "score": np.nan, f"tp_credit_{tk}": np.nan, f"tp_frag_{tk}": np.nan,
                     **{f"tiou_gt{j}": t for j, t in
                        enumerate(temporal_iou_with_group(s, e, gs, ge))}})
    for rid, s, e, score, yc, yf in zip(
            p["source_row_id"].to_numpy(), p["t-start"].to_numpy(dtype=np.float64),
            p["t-end"].to_numpy(dtype=np.float64), p["score"].to_numpy(dtype=np.float64),
            p[f"tp_credit_{tk}"].to_numpy(), p[f"tp_frag_{tk}"].to_numpy()):
        tious = temporal_iou_with_group(s, e, gs, ge)
        if tious.max() < tau:
            continue
        rows.append({"role": "pred", "id": int(rid), "t_start": s, "t_end": e,
                     "score": score, f"tp_credit_{tk}": int(yc), f"tp_frag_{tk}": int(yf),
                     **{f"tiou_gt{j}": t for j, t in enumerate(tious)}})
    # labels are integers on prediction rows and empty on GT rows
    return pd.DataFrame(rows).astype({f"tp_credit_{tk}": "Int64", f"tp_frag_{tk}": "Int64"})
