"""Column semantics, stable row identity, validation.

Two GT identities per prediction (never conflate them):

  best_gt_idx / best_gt_dur       geometric argmax-tIoU GT; several
                                  predictions of one GT can share it (used
                                  by the max-tIoU convention, supp. Sec. E).
  matched_gt_idx / matched_gt_dur the one-to-one GT under LOCK_TAU; at most
                                  one prediction per GT, -1 / NaN otherwise.
                                  The target q and the oracle duration use
                                  this one.

gt_idx columns index into the (video-id, label) GT group; -1 / NaN mean
"none". `prediction_id` == `source_row_id` == the 0-based row number in the
prediction file as loaded; tables join on it, never on positional indices.
"""
import numpy as np

#: the one-to-one matcher's lock threshold: tau=0, any positive overlap can
#: lock (the LaECE0 convention). It defines the target q and the matched set.
LOCK_TAU = 1e-9

BASE_COLUMNS = [
    "prediction_id", "video-id", "label", "t-start", "t-end", "score",
    "pred_len", "log_len",
    "max_tiou", "ceiling", "best_gt_dur", "best_gt_idx",
    "matched_tiou", "matched_gt_dur", "matched_gt_idx",
]


def assign_prediction_ids(pred_df):
    """Stable identity from load order; also the matching tie-breaker."""
    pred_df = pred_df.reset_index(drop=True)
    pred_df["source_row_id"] = np.arange(len(pred_df), dtype=np.int64)
    pred_df["prediction_id"] = pred_df["source_row_id"]
    return pred_df


def validate_matched(df):
    """Invariant checks on a matched base table; raises ValueError on breach."""
    problems = []
    pred_len = (df["t-end"] - df["t-start"]).to_numpy()
    if not np.all(np.isfinite(df[["t-start", "t-end", "score"]].to_numpy())):
        problems.append("non-finite endpoints or scores")
    if (pred_len <= 0).any():
        problems.append(f"{int((pred_len <= 0).sum())} rows with non-positive length")
    unmatched = df["matched_gt_idx"].to_numpy() == -1
    if (df.loc[unmatched, "matched_tiou"] != 0).any():
        problems.append("unmatched rows with matched_tiou != 0")
    if not df.loc[unmatched, "matched_gt_dur"].isna().all():
        problems.append("unmatched rows with non-NaN matched_gt_dur")
    matched = ~unmatched
    if (df.loc[matched, "matched_tiou"] <= 0).any():
        problems.append("matched rows with matched_tiou <= 0")
    if (df.loc[matched, "matched_gt_dur"] <= 0).any():
        problems.append("matched rows with non-positive matched_gt_dur")
    if df["prediction_id"].duplicated().any():
        problems.append("duplicate prediction_id")
    # one-to-one: a locked GT appears at most once per (video, label)
    m = df[matched]
    dup = m.duplicated(subset=["video-id", "label", "matched_gt_idx"]).sum()
    if dup:
        problems.append(f"{int(dup)} duplicate locked GT assignments")
    if problems:
        raise ValueError("matched-table invariants violated: " + "; ".join(problems))
    return {"n_rows": int(len(df)), "n_matched": int(matched.sum())}


#: the registered split seeds
REGISTERED_SEEDS = (0, 1, 2, 3, 4)
