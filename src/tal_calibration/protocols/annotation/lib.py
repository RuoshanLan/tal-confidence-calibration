"""Shared layer of the annotation-ambiguity audit (Sec. 4.1, Table 2, supp. Sec. D).

ActivityNet-1.3 validation x ActionFormer, the fused post-NMS prediction pkl.
`experiments/annotation/run_annotation.py` is the entry point; the Ego4D-MQ
chain (ego4d.py) reuses the matcher, calibrator, metrics and populations here.

Conventions: all randomness is seeded (world seeds 43..47, bootstrap seed 7);
folds are `zlib.crc32(video_id) % 5`. `score` (proposal x external classifier)
is the only ranking field; calibrated confidences feed the metrics only.
"""
import json
import os
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ... import populations
from ...calibration import ece
from ...matching import add_tiou_columns
from ...geometry import temporal_iou_with_group
from ...prediction_io import load_pickle_predictions
from ...datasets import load_datasets


_ANET = load_datasets(["anet"])["anet"]
PRED_PATH = _ANET["pred"]
ANNO_PATH = _ANET["anno"]
EVAL_SPLIT = "validation"

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------
REFERENCE_SEED = 43                 # = WORLD_SEEDS[0]; seeds 02_noise_curve.csv
N_WORLDS = 10                       # alternative annotation worlds per seed
N_FOLDS = 5                         # video-grouped OOF folds (crc32, not GroupKFold)
EPS = 1.0e-6

# TP/FP tIoU thresholds; Table 2 reads the lowest
CREDIT_TAUS = (0.50, 0.75, 0.95)
LOCK_TAU_Q = 1.0e-9                 # quality target: a GT is matchable if tIoU > this

ECE_BINS = 10                       # credit ECE (percent)
LAECE_BINS = 25                     # LaECE_0

#: Reported populations: all predictions, and the LRP operating point
#: cross-fitted on this chain's crc32 folds (so its folds coincide with the
#: calibration folds).
POPULATIONS = ("full", "lrp_crossfit")

#: tIoU the LRP threshold is fitted against: ActivityNet's primary tau.
LRP_ANCHOR_TAU = 0.75

# Video-clustered percentile bootstrap (95%).
N_BOOT = 200
BOOT_SEED = 7
# World-generation seeds; the first is the reference seed.
WORLD_SEEDS = (43, 44, 45, 46, 47)

# --------------------------------------------------------------------------
# DETAD joint distribution of GT length x annotator agreement (DETAD
# supplementary; percent of GT). Rows: length XS..XL; cols: agreement
# XW, W, M, H, XH.
# --------------------------------------------------------------------------
DETAD_J = np.array([
    [1.8, 12.2, 20.9, 14.5, 5.0],
    [0.0,  0.6,  4.1,  7.6, 3.9],
    [0.1,  0.6,  3.3,  7.3, 4.6],
    [0.1,  0.5,  2.3,  3.7, 2.8],
    [0.1,  0.2,  0.9,  1.6, 1.2],
], dtype=np.float64)
# DETAD Fig. 3 agreement marginal (test subset, % of GT)
DETAD_AGR_MARGINAL = np.array([2.1, 14.1, 31.5, 34.8, 17.4])
# bucket edges: length (s) XS..XL, agreement XW..XH
LEN_EDGES = (0.0, 30.0, 60.0, 120.0, 180.0, np.inf)
AGR_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# noise-model simulation grid
NOISE_SIGMA_GRID = np.linspace(0.02, 0.8, 25)
NOISE_N_SIM = 3000
NOISE_REDRAW_ROUNDS = 6


# --------------------------------------------------------------------------
# Conditional noise structure P(agr | len)
# --------------------------------------------------------------------------
def conditional_agr_given_len(J=DETAD_J):
    """P(agr-bucket | len-bucket) = J / rowsum(J). Row i indexes a length
    bucket XS..XL; the returned row sums to 1 over the 5 agreement buckets."""
    return J / J.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_predictions():
    """The prediction pkl as a DataFrame (video-id, t-start, t-end, label,
    score), t-start clipped to >= 0; `pred_row_id` records the pkl order."""
    df = load_pickle_predictions(PRED_PATH)
    df["t-start"] = np.clip(df["t-start"].to_numpy(), 0.0, None)
    df = df.reset_index(drop=True)
    df["pred_row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def load_gt():
    """ActivityNet-1.3 validation GT as an instance table with per-instance
    duration. Sanitation: (s,e)=sorted(segment), clip to [0,duration]; if
    e-s < 1e-3 set e = min(s+1e-3, duration). Labels are the integer label_id
    (== sorted-name index).

    Columns: video-id, t-start, t-end, label, duration, instance_id (0..7653 in
    JSON traversal order), len (= t-end - t-start after sanitation)."""
    with open(ANNO_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)["database"]
    vids, starts, ends, labels, durs = [], [], [], [], []
    for vid, v in db.items():
        if v["subset"].lower() != EVAL_SPLIT:
            continue
        dur = float(v["duration"])
        for ann in v["annotations"]:
            s, e = sorted(float(x) for x in ann["segment"])
            s = min(max(s, 0.0), dur)
            e = min(max(e, 0.0), dur)
            if e - s < 1.0e-3:
                e = min(s + 1.0e-3, dur)
            vids.append(vid)
            starts.append(s)
            ends.append(e)
            labels.append(int(ann["label_id"]))
            durs.append(dur)
    gt = pd.DataFrame({
        "video-id": vids,
        "t-start": np.asarray(starts, dtype=np.float64),
        "t-end": np.asarray(ends, dtype=np.float64),
        "label": np.asarray(labels, dtype=np.int64),
        "duration": np.asarray(durs, dtype=np.float64),
    })
    gt["instance_id"] = np.arange(len(gt), dtype=np.int64)
    gt["len"] = gt["t-end"].to_numpy() - gt["t-start"].to_numpy()
    return gt


# --------------------------------------------------------------------------
# Bucketing helpers
# --------------------------------------------------------------------------
def bucket_index(values, edges):
    """Half-open (edges[i], edges[i+1]] bucketing, 0-indexed. A value exactly at
    the left edge of the first bucket (e.g. length 0) maps to bucket 0."""
    values = np.asarray(values, dtype=np.float64)
    idx = np.digitize(values, edges[1:-1], right=True)
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


# --------------------------------------------------------------------------
# Deterministic video-grouped folds
# --------------------------------------------------------------------------
def crc32_fold(video_ids, n_folds=N_FOLDS):
    """OOF fold per row: crc32(video_id) % n_folds. Same video -> same fold;
    reproducible across processes (Python's `hash()` is salted)."""
    return np.array([zlib.crc32(str(v).encode("utf-8")) % n_folds
                     for v in video_ids], dtype=np.int64)


# --------------------------------------------------------------------------
# Noise model: symmetric Gaussian boundary jitter, relative to instance length
# --------------------------------------------------------------------------
def _median_pairwise_tiou(starts, ends):
    """Median of the 6 pairwise tIoUs among 4 annotations, per simulation.
    starts/ends: (n_sim, 4) arrays. Returns (n_sim,)."""
    n_sim = starts.shape[0]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ious = np.empty((n_sim, len(pairs)), dtype=np.float64)
    for k, (a, b) in enumerate(pairs):
        inter = np.minimum(ends[:, a], ends[:, b]) - np.maximum(starts[:, a], starts[:, b])
        inter = np.clip(inter, 0.0, None)
        union = (np.maximum(ends[:, a], ends[:, b]) - np.minimum(starts[:, a], starts[:, b]))
        ious[:, k] = np.where(union > 0, inter / union, 0.0)
    return np.median(ious, axis=1)


def simulate_agreement(sigma_rel, rng, n_sim=NOISE_N_SIM):
    """Mean median-pairwise-tIoU over n_sim simulations for one sigma_rel.

    Canonical unit segment [0,1]; original annotator + 3 jittered ones. Each
    jittered annotator's start and end are independently N(0, sigma_rel).
    Invalid (end <= start) draws are redrawn up to NOISE_REDRAW_ROUNDS times
    (resetting them to the original would inflate agreement at high sigma);
    residual invalids fall back to zero jitter."""
    s = np.zeros((n_sim, 4), dtype=np.float64)
    e = np.ones((n_sim, 4), dtype=np.float64)
    for a in range(1, 4):
        js = rng.normal(0.0, sigma_rel, n_sim)
        je = rng.normal(0.0, sigma_rel, n_sim)
        cs = 0.0 + js
        ce = 1.0 + je
        for _ in range(NOISE_REDRAW_ROUNDS):
            bad = ce <= cs
            if not bad.any():
                break
            cs[bad] = 0.0 + rng.normal(0.0, sigma_rel, int(bad.sum()))
            ce[bad] = 1.0 + rng.normal(0.0, sigma_rel, int(bad.sum()))
        bad = ce <= cs
        cs[bad] = 0.0
        ce[bad] = 1.0
        s[:, a] = cs
        e[:, a] = ce
    return float(np.mean(_median_pairwise_tiou(s, e)))


def build_noise_curve(seed, sigma_grid=NOISE_SIGMA_GRID, n_sim=NOISE_N_SIM):
    """Agreement curve a(sigma_rel) over sigma_grid (monotone decreasing to a
    floor A_MIN). Returns (sigma_grid, agreements, A_MIN)."""
    rng = np.random.default_rng(seed)
    agreements = np.array([simulate_agreement(sg, rng, n_sim) for sg in sigma_grid])
    # invert_agreement interpolates on this curve, so it must be monotone
    if not np.all(np.diff(agreements) <= 1.0e-9):
        raise ValueError(f"noise curve for seed {seed} is not monotone decreasing")
    a_min = float(agreements.min())
    return sigma_grid, agreements, a_min


def invert_agreement(target_a, sigma_grid, agreements, a_min):
    """sigma_rel(a) by linear interpolation on the (monotone decreasing) curve.
    Targets below A_MIN + 0.01 are clamped to the max-noise sigma and flagged.
    Returns (sigma_rel, clamped_bool). Vectorized over target_a."""
    target_a = np.asarray(target_a, dtype=np.float64)
    clamp_below = a_min + 0.01
    clamped = target_a < clamp_below
    a_eff = np.where(clamped, clamp_below, target_a)
    # curve is decreasing in sigma; flip both axes for np.interp (needs ascending x)
    xs = agreements[::-1]
    ys = sigma_grid[::-1]
    sig = np.interp(a_eff, xs, ys)
    # any target below the reachable floor still maps to the max sigma
    sig = np.where(target_a < clamp_below, sigma_grid[-1], sig)
    return sig, clamped


# --------------------------------------------------------------------------
# Per-instance annotation sampling
# --------------------------------------------------------------------------
def sample_instance_sigma(gt, curve, rng):
    """Per-instance sigma (absolute time) for boundary jitter.

    length bucket <- instance length; agreement bucket ~ P(agr | len);
    target a ~ Uniform(bucket agreement range); sigma_rel = sigma_rel(max(a,
    A_MIN+.01)); sigma = sigma_rel * instance length. Returns
    (sigma_abs, clamped)."""
    sigma_grid, agreements, a_min = curve
    lens = gt["len"].to_numpy()
    len_bucket = bucket_index(lens, LEN_EDGES)
    pcond = conditional_agr_given_len()          # (5 len, 5 agr)
    n = len(gt)
    agr_bucket = np.empty(n, dtype=np.int64)
    for lb in range(5):
        m = len_bucket == lb
        if m.any():
            agr_bucket[m] = rng.choice(5, size=int(m.sum()), p=pcond[lb])
    lo = np.array(AGR_EDGES[:-1])[agr_bucket]
    hi = np.array(AGR_EDGES[1:])[agr_bucket]
    sampled_a = lo + rng.random(n) * (hi - lo)
    # invert_agreement flags targets below A_MIN+0.01 and clamps their sigma to
    # the max-noise grid point; the flag must come from the raw sampled_a (the
    # sigma formula's max(a, A_MIN+.01) flooring is applied inside).
    sigma_rel, clamped = invert_agreement(sampled_a, sigma_grid, agreements, a_min)
    sigma_abs = sigma_rel * lens
    return sigma_abs, clamped


def sample_single_swap(gt, sigma_abs, rng, retries=20):
    """One alternative annotation per instance.
    Per boundary: jitter N(0, sigma), clip start >= 0 and end <= duration, retry
    up to `retries` for end > start + 1e-3, else fall back to the original
    boundary pair. Returns (starts, ends) arrays of shape (n,)."""
    s0 = gt["t-start"].to_numpy()
    e0 = gt["t-end"].to_numpy()
    dur = gt["duration"].to_numpy()
    n = len(gt)
    out_s = s0.copy()
    out_e = e0.copy()
    pending = np.ones(n, dtype=bool)
    for _ in range(retries):
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        cs = np.clip(s0[idx] + rng.normal(0.0, sigma_abs[idx]), 0.0, dur[idx])
        ce = np.clip(e0[idx] + rng.normal(0.0, sigma_abs[idx]), 0.0, dur[idx])
        good = ce > cs + 1.0e-3
        gi = idx[good]
        out_s[gi] = cs[good]
        out_e[gi] = ce[good]
        pending[gi] = False
    # residual pending keep the original (s0, e0) -- already in out_s/out_e
    return out_s, out_e


def build_match_plan(pred_df, gt_video, gt_label):
    """Precompute the world-independent matching plan reused across all worlds.

    Returns a list of (pred_rows, gt_rows) per (video, label) group that HAS GT,
    where `pred_rows` are original pred_df row positions in score-desc order
    (score desc, then t-start, t-end, pred_row_id ascending) and
    `gt_rows` are the row positions into the gt geometry arrays. Groups without
    GT are dropped (no positive target possible). Also returns the pred start/end
    arrays in pred_df row order for the tIoU calls."""
    pdf = pred_df.reset_index(drop=True)
    starts = pdf["t-start"].to_numpy(np.float64)
    ends = pdf["t-end"].to_numpy(np.float64)
    gt_key = pd.DataFrame({"video-id": gt_video, "label": gt_label})
    gt_groups = {k: idx.to_numpy()
                 for k, idx in gt_key.groupby(["video-id", "label"]).groups.items()}
    order = pdf.sort_values(["score", "t-start", "t-end", "pred_row_id"],
                            ascending=[False, True, True, True], kind="mergesort")
    plan = []
    for (v, lab), grp in order.groupby(["video-id", "label"], sort=False):
        gidx = gt_groups.get((v, lab))
        if gidx is None or len(gidx) == 0:
            continue
        plan.append((grp.index.to_numpy(), gidx))
    return plan, starts, ends


def compute_quality_targets(pred_df, gt_starts, gt_ends, gt_video, gt_label,
                            gt_instance, plan=None, return_match_id=False):
    """One-to-one greedy quality target q^{o2o} per prediction against a given
    world's GT geometry. Predictions are processed score-desc within each
    (video,label) group; eligibility tIoU > LOCK_TAU_Q; matched -> its tIoU,
    unmatched -> 0. Returns a float32 array in pred_df row order.

    return_match_id: also return the matched GT instance_id per prediction
    (-1 if unmatched), which separates the same instance moving from a
    reassignment to another GT of the same class (the ICC readout).
    Pass a `plan` from build_match_plan to reuse the grouping across worlds."""
    n = len(pred_df)
    q = np.zeros(n, dtype=np.float32)
    mid = np.full(n, -1, dtype=np.int64) if return_match_id else None
    if plan is None:
        plan, starts, ends = build_match_plan(pred_df, gt_video, gt_label)
    else:
        plan, starts, ends = plan
    for pred_rows, gidx in plan:
        gs = gt_starts[gidx]
        ge = gt_ends[gidx]
        locked = np.zeros(len(gidx), dtype=bool)
        for r in pred_rows:  # already score-desc
            tiou = temporal_iou_with_group(starts[r], ends[r], gs, ge)
            ranked = np.argsort(tiou)[::-1]
            for g in ranked:
                if tiou[g] <= LOCK_TAU_Q:
                    break
                if not locked[g]:
                    locked[g] = True
                    q[r] = tiou[g]
                    if mid is not None:
                        mid[r] = gt_instance[gidx[g]]
                    break
    return (q, mid) if return_match_id else q


def compute_credit_targets(pred_df, gt_starts, gt_ends, gt_video, gt_label, taus,
                           plan=None):
    """Per-tau greedy one-to-one credit y(tau): eligible = unmatched GT with
    tIoU >= tau, assign the max-tIoU eligible. Predictions processed
    score-desc within each group. Returns dict tau -> int8 array in pred_df row
    order. Pass a `plan` from build_match_plan to reuse the grouping."""
    taus = sorted(taus)
    n = len(pred_df)
    out = {t: np.zeros(n, dtype=np.int8) for t in taus}
    if plan is None:
        plan, starts, ends = build_match_plan(pred_df, gt_video, gt_label)
    else:
        plan, starts, ends = plan
    for pred_rows, gidx in plan:
        gs = gt_starts[gidx]
        ge = gt_ends[gidx]
        locks = {t: np.zeros(len(gidx), dtype=bool) for t in taus}
        for r in pred_rows:  # already score-desc
            tiou = temporal_iou_with_group(starts[r], ends[r], gs, ge)
            ranked = np.argsort(tiou)[::-1]
            for t in taus:
                if tiou[ranked[0]] < t:
                    continue
                for g in ranked:
                    if tiou[g] < t:
                        break
                    if not locks[t][g]:
                        locks[t][g] = True
                        out[t][r] = 1
                        break
    return out


# --------------------------------------------------------------------------
# Calibrators: pooled score-only isotonic, OOF crc32 folds.
# --------------------------------------------------------------------------
def oof_isotonic(scores, targets, folds):
    """Out-of-fold pooled score-only isotonic calibration under crc32
    video-grouped folds (out_of_bounds='clip'). Fits against the continuous
    quality target OR a binary credit target identically (isotonic regresses
    E[target | score]). Returns OOF predictions (float64)."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), EPS, 1.0 - EPS)
    targets = np.asarray(targets, dtype=np.float64)
    out = np.full(len(scores), np.nan)
    for f in range(int(folds.max()) + 1):
        te = folds == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        s_tr, y_tr, s_te = scores[tr], targets[tr], scores[te]
        if np.unique(y_tr).size < 2:      # degenerate: pass score through
            out[te] = s_te
            continue
        pooled = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        pooled.fit(s_tr, y_tr)
        out[te] = pooled.predict(s_te)
    return np.clip(np.nan_to_num(out, nan=float(np.nanmean(targets))), 0.0, 1.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def credit_ece(y, conf, n_bins=ECE_BINS):
    """Equal-width credit-ECE on confidence (percent), via calibration.ece."""
    y = np.asarray(y, dtype=bool)
    conf = np.clip(np.asarray(conf, dtype=np.float64), EPS, 1.0 - EPS)
    if y.size == 0:
        return float("nan")
    return 100.0 * float(ece(y, conf, n_bins=n_bins))


def laece0(conf, q, n_bins=LAECE_BINS):
    """LaECE_0 (POOLED): one set of fixed-width confidence bins over ALL rows,
    per-bin |mean confidence - mean q| weighted by bin mass. q is the continuous
    quality target in [0,1]; conf the calibrated quality confidence. Pooled
    rather than class-macro: at the LRP operating point most classes hold too
    few rows for a per-class estimate."""
    conf = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    q = np.asarray(q, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        bm = bins == b
        if bm.any():
            tot += bm.mean() * abs(conf[bm].mean() - q[bm].mean())
    return float(tot)


# --------------------------------------------------------------------------
# Populations: the full pool and the cross-fitted LRP operating point
# --------------------------------------------------------------------------
#: The LRP mask needs anchor-tau matching over the ~1M-row pool (a pure-Python
#: pass) and is requested several times per run, so it is memoized on a cheap
#: fingerprint of the inputs.
_MASK_MEMO = {}


def _mask_memo_key(pred_df, gt, tau):
    pid = pred_df["pred_row_id"].to_numpy()
    return (len(pred_df), len(gt),
            int(pid[0]) if len(pid) else -1, int(pid[-1]) if len(pid) else -1,
            float(pred_df["score"].to_numpy()[:1].sum()),
            int(gt["label"].sum()), round(float(tau), 6))


def population_masks(pred_df, gt, tau=None):
    """Boolean row masks for the reported populations: `full` and `lrp_crossfit`.

    `tau` is the LRP anchor; None keeps this chain's ActivityNet anchor
    (LRP_ANCHOR_TAU). The Ego4D chain passes its own dataset's anchor.

    `add_tiou_columns` returns its rows regrouped by (video, label) and
    score-sorted, not in `pred_df` order, so the mask is fitted on that frame
    with its own folds and ids and mapped back to `pred_df` via `pred_row_id`.
    """
    tau = LRP_ANCHOR_TAU if tau is None else float(tau)
    key = _mask_memo_key(pred_df, gt, tau)
    hit = _MASK_MEMO.get(key)
    if hit is not None:
        return {k: v.copy() for k, v in hit.items()}

    n = len(pred_df)
    masks = {"full": np.ones(n, dtype=bool)}

    # LRP-optimal score threshold, cross-fitted on this chain's own crc32
    # folds: each fold's cutoff is fitted on the OTHER folds' rows and GT, so
    # no row's membership depends on GT it participated in. The fit uses
    # anchor-tau matching, so the segment geometry is required.
    need = [c for c in ("t-start", "t-end", "video-id")
            if c not in pred_df.columns]
    if need:
        raise KeyError(
            f"population_masks: {need} missing; the lrp_crossfit head is "
            "fitted against anchor-tau matching and needs the segment "
            "geometry (a prediction frame from load_predictions always has it)")
    dfm = add_tiou_columns(pred_df, gt, lock_tau=tau)
    ids = dfm["pred_row_id"].to_numpy()
    folds = crc32_fold(dfm["video-id"].to_numpy())
    gt_folds = crc32_fold(gt["video-id"].to_numpy())
    sel, _ = populations.select(
        "lrp_crossfit", dfm.assign(source_row_id=ids),
        gt, tau=tau, folds=folds, gt_folds=gt_folds,
        n_splits=N_FOLDS)
    pos = pd.Series(np.arange(n), index=pred_df["pred_row_id"].to_numpy())
    lrp_mask = np.zeros(n, dtype=bool)
    lrp_mask[pos.loc[ids[sel]].to_numpy()] = True
    masks["lrp_crossfit"] = lrp_mask
    _MASK_MEMO[key] = {k: v.copy() for k, v in masks.items()}
    return masks


# --------------------------------------------------------------------------
# Atomic I/O
# --------------------------------------------------------------------------
def atomic_write_text(path, text):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------
# Video-clustered bootstrap
# --------------------------------------------------------------------------
def video_bootstrap_indices(video_ids, n_boot=N_BOOT, seed=BOOT_SEED):
    """Yield row-index arrays for B video-clustered bootstrap replicates.

    Videos are the resampling unit: each replicate samples len(unique videos)
    videos with replacement and concatenates ALL rows of the chosen videos.
    Deterministic given seed, so arms drawn with the same seed see the same
    videos per replicate."""
    video_ids = np.asarray(video_ids)
    unique = pd.unique(video_ids)
    groups = {v: np.flatnonzero(video_ids == v) for v in unique}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([groups[v] for v in chosen])


def percentile_ci(values, lo=2.5, hi=97.5):
    """(point-agnostic) percentile CI from bootstrap replicate values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    return (float(np.percentile(values, lo)),
            float(np.percentile(values, hi)), int(values.size))
