"""Evaluation-population registry shared by every experiment.

A selector takes the pair's candidate universe (the official pool, capped per
video at ``budget_k``) and the evaluation-split GT, and returns
``(mask, provenance)``: a boolean mask in ``df`` row order, and a dict with the
rule, its parameters, fitted thresholds and the population identity sha.

``DEFAULT_POPULATION`` is the population calibration is reported on; mAP is
always computed on ``official_topk``.
"""
from __future__ import annotations

import numpy as np

from .protocols.benchmark import common

#: Reported calibration population.
DEFAULT_POPULATION = "lrp_crossfit"

#: Tie policy shared by every rank-based selector.
TIE_POLICY = "score desc, source_row_id asc (mergesort)"

_REGISTRY: dict = {}


def register(name):
    def _wrap(fn):
        _REGISTRY[name] = fn
        return fn
    return _wrap


def available():
    return tuple(_REGISTRY)


def select(name, df, gt, **kw):
    """Resolve `name` against the registry. Returns (mask, provenance)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown population {name!r}; known: {available()}")
    mask, prov = _REGISTRY[name](df, gt, **kw)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(df):
        raise RuntimeError(f"{name}: mask length {len(mask)} != {len(df)} rows")
    prov = dict(prov)
    prov["population"] = name
    prov["population_n"] = int(mask.sum())
    prov["population_identity_sha256"] = common.population_identity_sha256(
        df.loc[mask, "source_row_id"])
    return mask, prov


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------
@register("official_topk")
def _official_topk(df, gt, **_):
    """The candidate universe itself (per-video top-K, applied when the pool is
    built): an all-true mask."""
    return np.ones(len(df), dtype=bool), {
        "rule": "per-video top-K (budget_k), applied at pool construction",
        "tie_policy": TIE_POLICY,
        "constructible": True,
        "gt_used": "none (budget is a declared constant)",
    }


@register("lrp_crossfit")
def _lrp_crossfit(df, gt, *, tau, folds=None, gt_folds=None, n_splits=5, **_):
    """Cross-fitted oLRP-optimal score threshold.

    For each video-grouped fold f the oLRP-minimising score cutoff is fitted on
    the other folds' rows and GT and applied to fold f, so no row's membership
    depends on its own GT. Cross-fitting stands in for the held-out validation
    split object detection fits this threshold on: predictions exist only for
    the evaluation split.

    ``tau`` is the anchor: the population is fitted once at this TP definition
    and held fixed across the tau grid, so a tau sweep changes only a row's
    label, never which rows are present.
    """
    from .protocols.benchmark import benchmark_tasks, protocol

    if "matched_tiou" not in df.columns:
        raise KeyError("lrp_crossfit needs a tau-locked `matched_tiou` column; "
                       "call matching.add_tiou_columns(df, gt, lock_tau=tau)")
    if folds is None:
        folds, gt_folds = fold_maps(df, gt, n_splits=n_splits)
    elif gt_folds is None:
        raise ValueError("pass gt_folds together with folds")

    scores = df["score"].to_numpy(np.float64)
    mtiou = df["matched_tiou"].to_numpy(np.float64)
    mask = np.zeros(len(df), dtype=bool)
    per_fold = {}
    for f in range(n_splits):
        te = folds == f
        tr = ~te
        n_gt_tr = int((gt_folds != f).sum())
        if not tr.any() or n_gt_tr == 0:
            continue
        lrp, thr, tp, fp, fn = benchmark_tasks.lrp_sweep(scores[tr], mtiou[tr],
                                                  n_gt_tr, tau)
        mask[te] = scores[te] >= thr
        per_fold[int(f)] = {"threshold": float(thr), "train_lrp": float(lrp),
                            "n_gt_train": n_gt_tr,
                            "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    if not per_fold:
        raise RuntimeError(
            "lrp_crossfit: no fold had both training rows and training GT -- "
            "the frame is degenerate for video-grouped cross-fitting (are all "
            "rows on a single video?)")
    if not mask.any():
        raise RuntimeError(
            f"lrp_crossfit: threshold(s) {sorted(v['threshold'] for v in per_fold.values())} "
            "selected zero rows")
    thrs = [v["threshold"] for v in per_fold.values()]
    return mask, {
        "rule": "cross-fitted oLRP-optimal score threshold",
        "tie_policy": "score >= per-fold threshold",
        "constructible": True,
        "gt_used": "OTHER folds' GT only (one scalar per fold)",
        "anchor_tau": float(tau),
        "n_splits": int(n_splits),
        "fold_thresholds": per_fold,
        "threshold_mean": float(np.mean(thrs)) if thrs else float("nan"),
        "threshold_min": float(min(thrs)) if thrs else float("nan"),
        "threshold_max": float(max(thrs)) if thrs else float("nan"),
        "fold_rule": protocol.video_folds.__doc__.splitlines()[0],
    }


def resolve(ctx, name, *, tau=None, log=print):
    """Population mask + provenance for a ``build_pair_context`` dict, selected
    on the official pool and memoized on ``ctx`` so every task of a pair sees
    the same rows. ``lrp_crossfit`` defaults to the dataset's headline tau."""
    from .matching import add_tiou_columns
    from .protocols.benchmark import protocol

    ds = ctx["dataset"]
    df, gt = ctx["official_df"], ctx["gt"]
    if name == "lrp_crossfit" and tau is None:
        tau = protocol.HEADLINE_TAU[ds]
    memo = ctx.setdefault("population_masks", {})
    key = (name, tau)
    if key not in memo:
        if name == "lrp_crossfit":
            # the anchor-tau matching is what the threshold is fitted against
            dfm = add_tiou_columns(df, gt, lock_tau=tau)
            memo[key] = select(name, dfm, gt, tau=tau)
        else:
            memo[key] = select(name, df, gt)
    mask, prov = memo[key]
    return mask.copy(), prov


# --------------------------------------------------------------------------
# shared fold mapping
# --------------------------------------------------------------------------
def fold_maps(df, gt, n_splits=5):
    """One video->fold map for predictions and GT.

    Predicted videos take ``protocol.video_folds`` (the OOF calibration folds);
    GT-only videos, which carry FN mass but no rows, are assigned by sorted
    round-robin.
    """
    from .protocols.benchmark import protocol

    vids = df["video-id"].to_numpy()
    row_folds = np.full(len(vids), -1, dtype=np.int64)
    for f, (_, te) in enumerate(protocol.video_folds(vids, n_splits=n_splits)):
        row_folds[te] = f
    if (row_folds < 0).any():
        raise RuntimeError("video_folds left rows unassigned")
    vid2fold = dict(zip(vids, row_folds))
    for i, v in enumerate(sorted(set(gt["video-id"].to_numpy()) - set(vid2fold))):
        vid2fold[v] = i % n_splits
    gt_folds = np.array([vid2fold[v] for v in gt["video-id"].to_numpy()])
    return row_folds, gt_folds
