"""Protocol shared by the benchmark and fragmentation experiments: constants,
the registered pairs, the multi-tau matcher, the calibration arms, the metrics
and the key/grid validators.

``score`` (the raw detector score) is the only ranking field;
``conf_{arm}_{tau}`` are calibrated probabilities used only for ECE / ACE / SCE /
NLL. Evaluation GT is always the full split in ``EVAL_SPLIT``.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from ...calibration import ece, make_calibrator
from ...geometry import interval_overlap_with_union, temporal_iou_with_group
from ...datasets import load_benchmark_pairs, load_datasets

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
EPS = 1.0e-6
EVAL_TOPK = 2000          # per-video cap of the native pools
N_SPLITS = 5              # video-grouped CV folds
MIN_CLASS_FIT = 50        # iso_class fallback threshold
MAX_PARAM_FIT = 200_000   # subsample cap for netcal parametric fits (seeded)
SEED = 0

# Evaluation annotation split per dataset (fixed universe for mAP/LRP/GT stats).
EVAL_SPLIT = {
    "thumos": "test",
    "epic_verb": "validation",
    "epic_noun": "validation",
    "ego4d": "validation",
    "anet": "validation",
}

# Headline tau per dataset (single source).
HEADLINE_TAU = {"thumos": 0.5, "epic_verb": 0.3, "epic_noun": 0.3,
                "ego4d": 0.3, "anet": 0.75}


def _pop(suffix, pool, budget_k, source):
    return {"suffix": suffix, "pool": pool, "budget_k": budget_k, "source": source}


OFFICIAL_POPULATION = {
    ("thumos", "actionformer"): _pop("", "matched200", 200, "native"),
    ("thumos", "tridet"): _pop("_k200", "matched200", 200, "posthoc_truncation"),
    ("thumos", "temporalmax"): _pop("", "matched200", 200, "native"),
    ("thumos", "tadtr"): _pop("", "matched200", 200, "native"),
    ("thumos", "te-tad"): _pop("_k200", "matched200", 200, "posthoc_truncation"),
    ("epic_verb", "actionformer"): _pop("", "native", 2000, "native"),
    ("epic_verb", "tridet"): _pop("", "native", 2000, "native"),
    ("epic_verb", "temporalmax"): _pop("", "native", 2000, "native"),
    ("epic_noun", "actionformer"): _pop("", "native", 2000, "native"),
    ("epic_noun", "tridet"): _pop("", "native", 2000, "native"),
    ("ego4d", "actionformer"): _pop("", "native", 2000, "native"),
    ("anet", "actionformer"): _pop("", "native", 200, "native"),
    # ActivityNet: the official population is the fused-score top-200 (TadTR
    # exports <= 200 natively); the EPIC TE-TAD exports are <= 2000 natively.
    ("anet", "tadtr"): _pop("", "native", 200, "native"),
    ("anet", "tridet"): _pop("_k200", "matched200", 200, "posthoc_truncation"),
    ("epic_verb", "te-tad"): _pop("", "native", 2000, "native"),
    ("epic_noun", "te-tad"): _pop("", "native", 2000, "native"),
    # TadTR with num_queries=100 (Table 1's TadTR row): 100 proposals x CUHK
    # top-2 = 200 rows/video. Both TadTR arms declare budget 200, but
    # ("anet", "tadtr") (num_queries=10) realizes only 20 rows/video; read the
    # realized pool size from the data, not from budget_k.
    ("anet", "tadtr_nq100"): _pop("", "native", 200, "native"),
}

#: The 16 detector--dataset pairs of Table 1, in table order. The fragmentation
#: chain runs exactly these. The benchmark chain additionally keeps anet:tadtr
#: (num_queries=10) from OFFICIAL_POPULATION: Table 1's double-dagger footnote
#: prints its numbers.
TABLE1_PAIRS = (
    ("thumos", "actionformer"),
    ("thumos", "tridet"),
    ("thumos", "temporalmax"),
    ("thumos", "tadtr"),
    ("thumos", "te-tad"),
    ("epic_verb", "actionformer"),
    ("epic_verb", "tridet"),
    ("epic_verb", "temporalmax"),
    ("epic_verb", "te-tad"),
    ("epic_noun", "actionformer"),
    ("epic_noun", "tridet"),
    ("epic_noun", "te-tad"),
    ("ego4d", "actionformer"),
    ("anet", "actionformer"),
    ("anet", "tridet"),
    ("anet", "tadtr_nq100"),
)


CALIB_TAUS = {
    "thumos": [0.3, 0.4, 0.5, 0.6, 0.7],
    "epic_verb": [0.1, 0.2, 0.3, 0.4, 0.5],
    "epic_noun": [0.1, 0.2, 0.3, 0.4, 0.5],
    "ego4d": [0.1, 0.2, 0.3, 0.4, 0.5],
    "anet": [0.5, 0.75, 0.95],
}
MAP_TAUS = {
    "thumos": [0.3, 0.4, 0.5, 0.6, 0.7],
    "epic_verb": [0.1, 0.2, 0.3, 0.4, 0.5],
    "epic_noun": [0.1, 0.2, 0.3, 0.4, 0.5],
    "ego4d": [0.1, 0.2, 0.3, 0.4, 0.5],
    "anet": list(np.round(np.linspace(0.5, 0.95, 10), 2)),
}

ARMS = ["raw", "hist", "iso", "platt", "beta", "temp"]


def tau_key(tau):
    return f"{tau:g}"


def benchmark_matrix():
    """(dataset config, (dataset, model) -> {pred, source_type}) from
    ``benchmark_pairs`` in ``configs/datasets.yaml``; the pair set must equal
    ``OFFICIAL_POPULATION``, whose order fixes the CSV row order."""
    ds = load_datasets()
    entries = load_benchmark_pairs()
    lookup = {(p["dataset"], p["model"]): {"pred": str(p["pred"]),
                                           "source_type": p["source_type"]}
              for p in entries}
    if len(lookup) != len(entries):
        raise RuntimeError("benchmark_pairs (datasets.yaml) contains duplicate "
                           "(dataset, model) entries")
    if set(lookup) != set(OFFICIAL_POPULATION):
        raise RuntimeError(
            "benchmark_pairs (datasets.yaml) != OFFICIAL_POPULATION: "
            f"missing {sorted(set(OFFICIAL_POPULATION) - set(lookup))}, "
            f"unexpected {sorted(set(lookup) - set(OFFICIAL_POPULATION))}")
    return ds, {k: lookup[k] for k in OFFICIAL_POPULATION}


def eval_split_videos(dataset, cfg_ds):
    with open(cfg_ds[dataset]["anno"], "r", encoding="utf8") as f:
        db = json.load(f)["database"]
    split = EVAL_SPLIT[dataset]
    return {k for k, v in db.items() if v["subset"].lower() == split}


def check_pred_coverage(dataset, cfg_ds, pred_videos, gt):
    split_videos = eval_split_videos(dataset, cfg_ds)
    extra = set(pred_videos) - split_videos
    if extra:
        raise RuntimeError(
            f"{dataset}: {len(extra)} predicted videos outside the "
            f"{EVAL_SPLIT[dataset]} split, e.g. {sorted(extra)[:3]}")
    return {"n_eval_videos_total": len(split_videos),
            "n_eval_videos_with_gt": int(gt["video-id"].nunique()),
            "n_videos_with_predictions": len(set(pred_videos))}


def pool_tag(dataset, model):
    return OFFICIAL_POPULATION[(dataset, model)]["pool"]


# --------------------------------------------------------------------------
# Atomic text writes
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


# ---- exact-key grid validation ----
def tau_r(x):
    return round(float(x), 4)


def validate_exact_keys(df, name, key_cols, expected_keys, pool_fn=None):
    pool_fn = pool_fn or pool_tag
    if "pool" not in df.columns:
        raise SystemExit(f"{name}: no `pool` column")
    rows = df[list(key_cols)].copy()
    if "tau" in rows.columns:
        rows["tau"] = rows["tau"].map(tau_r)
    got = list(map(tuple, rows.to_numpy()))
    if len(set(got)) != len(got):
        dup = rows[rows.duplicated()]
        raise SystemExit(f"{name}: duplicate keys, e.g. {dup.head(3).to_numpy().tolist()}")
    got, expected_keys = set(got), set(expected_keys)
    if got != expected_keys:
        raise SystemExit(
            f"{name}: key set != expected grid "
            f"(missing {sorted(expected_keys - got)[:5]}"
            f"{'...' if len(expected_keys - got) > 5 else ''}, "
            f"unexpected {sorted(got - expected_keys)[:5]}"
            f"{'...' if len(got - expected_keys) > 5 else ''})")
    for (ds, model), sub in df.groupby(["dataset", "model"]):
        want = pool_fn(ds, model)
        pools = set(sub["pool"].unique())
        if pools != {want}:
            raise SystemExit(f"{name}: pool {pools} != expected '{want}' for {ds}:{model}")


def add_source_row_id(pred):
    if "source_row_id" in pred.columns:
        return pred
    pred = pred.reset_index(drop=True).copy()
    pred["source_row_id"] = np.arange(len(pred), dtype=np.int64)
    return pred


def cap_per_video(df, k=EVAL_TOPK, by="score"):
    if "source_row_id" not in df.columns:
        raise KeyError("cap_per_video needs source_row_id (add_source_row_id "
                       "at load time)")
    return (df.sort_values([by, "t-start", "t-end", "label", "source_row_id"],
                           ascending=[False, True, True, True, True],
                           kind="mergesort")
              .groupby("video-id", sort=False).head(k)
              .reset_index(drop=True))


def label_multi_tau(pred_df, gt_df, taus, include_frag=True):
    """All matching targets over (video, label) groups, at every tau.

    ``tp_credit_{tau}`` is the official score-ordered, class-consistent,
    one-to-one greedy matcher. ``tp_frag_{tau}`` is the strict two-pass
    procedure of supp. Sec. C: S_g is seeded with the official TP of g, the
    second pass revisits only official false positives in score order, a GT is
    eligible only once claimed, and the admissible GT with the highest tIoU is
    taken. ``matched_tiou0`` is the one-to-one match at any positive overlap
    (the quality target q). Predictions are ordered by score, then t-start,
    t-end and source_row_id.
    """
    taus = sorted(taus)
    gt_lookup = {}
    for (video_id, label), group in gt_df.groupby(["video-id", "label"], sort=False):
        gt_lookup[(video_id, int(label))] = (
            group["t-start"].to_numpy(dtype=np.float64),
            group["t-end"].to_numpy(dtype=np.float64),
        )

    if "source_row_id" not in pred_df.columns:
        raise KeyError("label_multi_tau needs source_row_id "
                       "(add_source_row_id at load time)")
    outputs = []
    for (video_id, label), group in pred_df.groupby(["video-id", "label"], sort=False, group_keys=False):
        ordered = group.sort_values(
            ["score", "t-start", "t-end", "source_row_id"],
            ascending=[False, True, True, True], kind="mergesort"
        ).copy()
        starts = ordered["t-start"].to_numpy(dtype=np.float64)
        ends = ordered["t-end"].to_numpy(dtype=np.float64)
        n = len(ordered)
        credit = np.zeros((len(taus), n), dtype=np.int8)
        frag = np.zeros((len(taus), n), dtype=np.int8)
        max_tiou = np.zeros(n)
        matched0 = np.zeros(n)

        gt_segments = gt_lookup.get((video_id, int(label)))
        if gt_segments is not None and len(gt_segments[0]) > 0:
            gt_starts, gt_ends = gt_segments
            n_gt = len(gt_starts)
            locks0 = np.zeros(n_gt, dtype=bool)
            tious, rankeds = [], []
            for i in range(n):
                tiou = temporal_iou_with_group(starts[i], ends[i], gt_starts, gt_ends)
                ranked = np.argsort(tiou)[::-1]
                tious.append(tiou)
                rankeds.append(ranked)
                max_tiou[i] = tiou[ranked[0]]
                for g in ranked:  # tau=0 one-to-one (any positive overlap)
                    if tiou[g] <= 0.0:
                        break
                    if not locks0[g]:
                        locks0[g] = True
                        matched0[i] = tiou[g]
                        break

            for t_idx, tau in enumerate(taus):
                # ---- pass 1: official greedy one-to-one ----
                locks = np.zeros(n_gt, dtype=bool)
                m_credit = np.full(n, -1, dtype=np.int64)
                for i in range(n):
                    if max_tiou[i] < tau:
                        continue
                    tiou, ranked = tious[i], rankeds[i]
                    for g in ranked:
                        if tiou[g] < tau:
                            break
                        if not locks[g]:
                            locks[g] = True
                            credit[t_idx, i] = 1
                            m_credit[i] = g
                            break
                if not include_frag:
                    continue
                # ---- S_g seeded with the official TP assigned to g ----
                claimed = [[] for _ in range(n_gt)]
                for i in range(n):
                    if credit[t_idx, i] == 1:
                        claimed[m_credit[i]].append((starts[i], ends[i]))
                # ---- pass 2: official false positives only, score order ----
                frag[t_idx] = credit[t_idx]
                for i in range(n):
                    if credit[t_idx, i] == 1 or max_tiou[i] < tau:
                        continue
                    tiou, ranked = tious[i], rankeds[i]
                    best = -1
                    for g in ranked:      # tIoU-descending == argmax order
                        if tiou[g] < tau:
                            break
                        if not claimed[g]:          # eligible only once claimed
                            continue
                        if interval_overlap_with_union(
                                starts[i], ends[i], claimed[g]) <= 0.0:
                            best = g
                            break
                    if best >= 0:
                        frag[t_idx, i] = 1
                        claimed[best].append((starts[i], ends[i]))

        for t_idx, tau in enumerate(taus):
            ordered[f"tp_credit_{tau_key(tau)}"] = credit[t_idx]
            if include_frag:
                ordered[f"tp_frag_{tau_key(tau)}"] = frag[t_idx]
        ordered["max_tiou"] = max_tiou
        ordered["matched_tiou0"] = matched0
        outputs.append(ordered)
    return pd.concat(outputs, ignore_index=True)


def _fit_netcal(arm, s_tr, y_tr, s_te, seed=SEED):
    if len(s_tr) > MAX_PARAM_FIT:
        idx = np.random.RandomState(seed).choice(len(s_tr), MAX_PARAM_FIT, replace=False)
        s_tr, y_tr = s_tr[idx], y_tr[idx]
    # histogram binning uses 20 bins; the other calibrators take netcal defaults
    model = make_calibrator(arm, bins=20) if arm == "hist" else make_calibrator(arm)
    model.fit(s_tr.astype(np.float64), y_tr.astype(np.int64))
    return np.asarray(model.transform(s_te.astype(np.float64)), dtype=np.float64).reshape(-1)


def video_folds(videos, n_splits=N_SPLITS):
    """Deterministic video-grouped K-fold assignment: list of (train, test)
    row-index arrays. GroupKFold uses only the group labels, so every
    calibrator fitted on the same rows gets the same folds."""
    videos = np.asarray(videos)
    dummy = np.zeros(len(videos))
    return list(GroupKFold(n_splits=n_splits).split(dummy, None, videos))


def oof_calibrate(scores, y, videos, labels, arm, n_splits=N_SPLITS, seed=SEED,
                  return_diag=False):
    """Out-of-fold calibrated confidences under video-grouped K-fold CV.

    With ``return_diag=True`` returns ``(conf, diag)``; for ``iso_class`` the
    diag records the share of test rows served by a per-class map.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), EPS, 1.0 - EPS)
    if arm == "raw":
        return (scores, {}) if return_diag else scores
    y = np.asarray(y, dtype=np.int64)
    out = np.full(len(scores), np.nan)
    n_percls_rows = 0
    refined_per_fold = []
    for tr, te in video_folds(videos, n_splits=n_splits):
        s_tr, y_tr, s_te = scores[tr], y[tr], scores[te]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            out[te] = s_te
            continue
        if arm == "iso":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(s_tr, y_tr)
            p = iso.predict(s_te)
        elif arm == "iso_class":
            pooled = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            pooled.fit(s_tr, y_tr)
            p = pooled.predict(s_te)
            lab_tr, lab_te = labels[tr], labels[te]
            n_refined = 0
            for cls in np.unique(lab_tr):
                m_tr = lab_tr == cls
                m_te = lab_te == cls
                if m_tr.sum() >= MIN_CLASS_FIT and m_te.any() and 0 < y_tr[m_tr].sum() < m_tr.sum():
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(s_tr[m_tr], y_tr[m_tr])
                    p[m_te] = iso.predict(s_te[m_te])
                    n_percls_rows += int(m_te.sum())
                    n_refined += 1
            refined_per_fold.append(n_refined)
        else:
            p = _fit_netcal(arm, s_tr, y_tr, s_te, seed=seed)
        out[te] = p
    conf = np.clip(np.nan_to_num(out, nan=0.5), EPS, 1.0 - EPS)
    if not return_diag:
        return conf
    diag = {}
    if arm == "iso_class":
        diag = {
            "per_class_row_fraction": round(n_percls_rows / max(len(scores), 1), 6),
            "mean_refined_classes_per_fold": (
                round(float(np.mean(refined_per_fold)), 2) if refined_per_fold else 0.0),
        }
    return conf, diag


def calib_metrics(y, conf, labels):
    """ECE (equal-width bins), ACE (equal-mass bins), class-macro SCE, all with
    10 bins and in percent, and NLL of ``conf`` against binary ``y``; ``conf``
    is clipped to [EPS, 1 - EPS] first."""
    y = np.asarray(y, dtype=bool)
    conf = np.clip(np.asarray(conf, dtype=np.float64), EPS, 1.0 - EPS)
    sce_values = []
    for cls in pd.unique(labels):
        m = labels == cls
        if m.sum() > 0:
            sce_values.append(ece(y[m], conf[m]))
    return {
        "n": int(len(y)),
        "ECE": 100.0 * ece(y, conf),
        "ACE": 100.0 * ece(y, conf, strategy="quantile"),
        "SCE": 100.0 * float(np.mean(sce_values)) if sce_values else 0.0,
        "NLL": float(log_loss(y_true=y, y_pred=conf, labels=[False, True])),
        "mean_conf": float(conf.mean()),
        "tp_rate": float(y.mean()),
    }


def safe_auroc(y, conf):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, conf))


def video_bootstrap(videos, stat_fn, n_boot=200, seed=SEED):
    """Cluster bootstrap over videos: ``stat_fn(row_idx)`` on each of ``n_boot``
    resamples of whole videos. Reseeds with ``seed`` on every call."""
    videos = np.asarray(videos)
    unique = pd.unique(videos)
    groups = {v: np.flatnonzero(videos == v) for v in unique}
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[v] for v in sample])
        rows.append(stat_fn(idx))
    return pd.DataFrame(rows)
