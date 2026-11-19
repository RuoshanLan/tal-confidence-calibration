"""Per-partition LRP selection (the Sec. 5 operating point).

For each seed:

    1. fit ONE oLRP-minimising score cutoff on the TRAIN partition, using
       TRAIN rows and the GT of TRAIN videos only, at the tau=0 one-to-one
       matching that also defines q (unlike the benchmark's cross-fitted
       cutoff at each dataset's primary tau);
    2. keep every row of every partition whose score is >= that cutoff.

Selection happens before any fitting or calibration, so ground truth enters
membership only through that one scalar, fitted on video-disjoint data. The
ledger records the cutoff and each partition's realized size and hit rate.
"""
import numpy as np

#: tie policy: a row is admitted iff its score reaches the cutoff
LRP_TIE_POLICY = "score >= threshold"
PARTITION_NAMES = {0: "train", 1: "cal", 2: "test"}


class SelectionError(RuntimeError):
    pass


def ledger_rows(dataset, seed, ledgers, head="lrp"):
    """Flatten the per-partition ledgers into table rows: one row per
    partition, with ``label = -1`` (the cutoff is not per class)."""
    rows = []
    for pname, ledger in ledgers.items():
        rows.append({
            "dataset": dataset, "seed": seed, "head": head,
            "partition": pname, "tie_policy": ledger["tie_policy"],
            "label": -1,
            "realized": int(ledger["realized_total"]),
            "score_cutoff": float(ledger["threshold"]),
        })
    return rows


def _lrp_sweep(scores, q, n_gt, tau=0.0):
    """oLRP-minimising score cutoff. `q` is the tau=0 one-to-one matched tIoU
    (0 for unmatched), the chain's target."""
    order = np.argsort(-scores, kind="stable")
    s, qq = scores[order], q[order]
    tp = np.cumsum(qq > 0)
    n = np.arange(1, len(s) + 1)
    fp = n - tp
    fn = n_gt - tp
    loc = np.cumsum(np.where(qq > 0, 1.0 - qq, 0.0)) / (1.0 - tau)
    denom = tp + fp + fn
    lrp = np.where(denom > 0, (loc + fp + fn) / np.maximum(denom, 1), np.inf)
    ends = np.flatnonzero(np.r_[s[:-1] > s[1:], True])   # last row per score
    i = ends[int(np.argmin(lrp[ends]))]
    return float(lrp[i]), float(s[i]), int(tp[i]), int(fp[i]), int(fn[i])


def per_partition_lrp(universe, gt, split, tau=0.0, log=print):
    """(mask over `universe`, {partition_name: ledger}): the cutoff is fitted
    on TRAIN rows and TRAIN-video GT only and applied unchanged to CAL and
    TEST."""
    videos = universe["video-id"].astype(str)
    gt_videos = gt["video-id"].astype(str)
    scores = universe["score"].to_numpy(np.float64)
    q = universe["q"].to_numpy(np.float64)

    tr = (split == 0)
    tr_arr = tr.to_numpy() if hasattr(tr, "to_numpy") else np.asarray(tr)
    if not tr_arr.any():
        raise SelectionError("train partition has no candidate rows")
    n_gt_tr = int(gt_videos.isin(set(videos[tr_arr])).sum())
    if n_gt_tr == 0:
        raise SelectionError("train partition has no GT instances")
    lrp, thr, tp, fp, fn = _lrp_sweep(scores[tr_arr], q[tr_arr], n_gt_tr, tau)

    mask = scores >= thr
    ledgers = {}
    for p, pname in PARTITION_NAMES.items():
        in_p = (split == p)
        in_p = in_p.to_numpy() if hasattr(in_p, "to_numpy") else np.asarray(in_p)
        m_p = mask & in_p
        if not m_p.any():
            raise SelectionError(f"partition {pname}: empty LRP selection "
                                 f"at threshold {thr:.6f}")
        ledgers[pname] = {
            "rule": "LRP-optimal score threshold fitted on TRAIN",
            "threshold": thr, "fitted_on": "train", "anchor_tau": float(tau),
            "train_lrp": lrp, "train_tp": tp, "train_fp": fp, "train_fn": fn,
            "n_gt_train": n_gt_tr,
            "realized_total": int(m_p.sum()),
            "hit_rate": float((q[m_p] > 0).mean()),
            "tie_policy": LRP_TIE_POLICY,
        }
        log(f"[lrp/{pname}] thr={thr:.4f} realized={int(m_p.sum())} "
            f"hit={ledgers[pname]['hit_rate']:.3f} "
            f"(fitted on train, n_gt_train={n_gt_tr})")
    return mask, ledgers

