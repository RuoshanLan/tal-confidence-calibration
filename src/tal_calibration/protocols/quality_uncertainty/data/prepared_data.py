"""Prepared-data layer: match + featurize once per dataset.

Parts:

    gt         GT rows of the dataset's evaluation split
    base       predictions + matching columns (schemas.BASE_COLUMNS)
    neighbor   prediction_id + NBR_FEATS (all rows)

The pre-NMS cloud features are built by ``data.cloud`` over all post-NMS rows.

Nothing is written to disk: ``build`` is a pure function of the raw inputs,
memoised per process (``_PREPARED``) so a dataset's five seeds pay the cost
once. Frames handed out by ``build``/``load`` are the memoised objects; treat
them as read-only.
"""
from pathlib import Path

import numpy as np

from ....datasets import load_datasets
from ....matching import add_tiou_columns, load_label_map
from .observable_features import NBR_FEATS, add_neighbor_features
from . import schemas
from .splits import all_video_population

from ....prediction_io import load_predictions_from_path


class PreparedDataError(RuntimeError):
    """The prepared frames cannot be produced as requested."""


def load_metrics_module():
    """The vendored TAL evaluator module (MIT)."""
    from ....third_party import actionformer_metrics
    return actionformer_metrics


_PARTS = ("gt", "base", "neighbor")
DEFAULT_PARTS = _PARTS

#: dataset -> {"gt", "base", "neighbor", "provenance"}; process-local memo
_PREPARED = {}


def clear_memo():
    """Drop the in-process prepared-data memo (tests; long-lived processes)."""
    _PREPARED.clear()


#: benchmark's official-population rule (benchmark.protocol.cap_per_video), so
#: a truncated stream consumes exactly the rows Table 1 reports on. The keys
#: are copied rather than imported to keep this package independent of the
#: benchmark protocol. The two select the same set of rows, not in the same
#: order.
TRUNCATION_SORT_KEYS = ["score", "t-start", "t-end", "label", "source_row_id"]
TRUNCATION_ASCENDING = [False, True, True, True, True]


def truncate_per_video(pred, top_k):
    """Keep each video's top-``top_k`` rows. Returns (frame, ledger).

    ``top_k=None`` returns the caller's own object unmutated (only the per-video
    census of the ledger is computed), so an undeclared truncation cannot touch
    a native stream. ``max_rows_per_video_before`` tells a natively-200 stream
    apart from one cut to 200 from 2000.

    Otherwise it is a BOOLEAN MASK over the frame as loaded, never a
    sort/merge/reindex: ``prediction_id`` is the row's position in the
    prediction file and the final tiebreak of the matching sort, so surviving
    rows keep both their ids and their relative order.
    """
    n_before = int(len(pred))
    per_video_before = pred.groupby("video-id", sort=False).size()
    # .max() on an empty census is NaN; the None path must never raise
    max_before = int(per_video_before.max()) if n_before else 0
    ledger = {
        "top_k": None if top_k is None else int(top_k),
        "applied": False,
        "n_rows_before": n_before,
        "n_rows_after": n_before,
        "n_rows_dropped": 0,
        "max_rows_per_video_before": max_before,
        "max_rows_per_video_after": max_before,
        "n_videos_truncated": 0,
    }
    if top_k is None:
        return pred, ledger
    top_k = int(top_k)
    if top_k <= 0:
        raise PreparedDataError(f"top_k must be a positive int, got {top_k}")

    # positional, never index-label: `pred.index.isin(keep)` over-selects
    # when the index carries duplicate labels
    tmp = pred[["video-id"] + TRUNCATION_SORT_KEYS].copy()
    tmp["_pos"] = np.arange(len(pred), dtype=np.int64)
    keep = (tmp.sort_values(TRUNCATION_SORT_KEYS,
                            ascending=TRUNCATION_ASCENDING, kind="mergesort")
               .groupby("video-id", sort=False).head(top_k)["_pos"].to_numpy())
    mask = np.zeros(len(pred), dtype=bool)
    mask[keep] = True
    out = pred[mask].reset_index(drop=True)

    per_video_after = out.groupby("video-id", sort=False).size()
    ledger.update(
        applied=True,
        n_rows_after=int(len(out)),
        n_rows_dropped=n_before - int(len(out)),
        max_rows_per_video_after=(int(per_video_after.max())
                                  if len(out) else 0),
        n_videos_truncated=int((per_video_before > top_k).sum()),
    )
    return out, ledger


def load_raw(dataset, metrics_module=None, log=print):
    """Load GT + predictions for one dataset, restricted to the canonical
    evaluation population and truncated to the stream's declared per-video
    budget. Returns (gt, pred, cfg, truncation)."""
    cfg = load_datasets([dataset])[dataset]
    metrics_module = metrics_module or load_metrics_module()
    gt = metrics_module.load_gt_seg_from_json(cfg["anno"], split=None)
    # A json prediction source carries class NAMES; the annotation supplies the
    # name -> label_id map (the integer space the pickle sources and the GT
    # already use). Pickle streams never receive it.
    label_map = (load_label_map(Path(cfg["anno"]))
                 if cfg["source_type"] == "json" else None)
    pred = load_predictions_from_path(Path(cfg["pred"]), label_map,
                                      cfg["source_type"])
    pred = schemas.assign_prediction_ids(pred)
    # Restrict to the canonical evaluation split. This comes AFTER
    # assign_prediction_ids and is a boolean mask, so prediction ids (the
    # matching tiebreak) are unchanged. The partition resolves the CANONICAL
    # anchor's annotation, while the GT and label map above come from this
    # entry, so every detector stream of a dataset must name the same file.
    canon_cfg = load_datasets([cfg["canonical"]])[cfg["canonical"]]
    if Path(cfg["anno"]) != Path(canon_cfg["anno"]):
        raise PreparedDataError(
            f"[{dataset}] annotation {cfg['anno']} differs from its canonical "
            f"anchor {cfg['canonical']}'s {canon_cfg['anno']}; every detector "
            "of one dataset must share one annotation")
    canon_videos = all_video_population(dataset)
    extra = sorted(set(map(str, pred["video-id"].unique())) - canon_videos)
    if extra:
        # a raise, not a silent drop: no prediction may be excluded
        raise PreparedDataError(
            f"[{dataset}] {len(extra)} predicted videos lie outside the "
            f"declared evaluation split, e.g. {extra[:3]}")
    gt = gt[gt["video-id"].astype(str).isin(canon_videos)].copy()
    # Per-video top-K, declared per stream in datasets.yaml. Absent -> no-op.
    pred, truncation = truncate_per_video(pred, cfg["top_k"])
    if truncation["applied"]:
        log(f"[{dataset}] top-K truncation k={truncation['top_k']}: "
            f"{truncation['n_rows_before']} -> {truncation['n_rows_after']} "
            f"rows ({truncation['n_videos_truncated']} videos truncated)")
    return gt.reset_index(drop=True), pred, cfg, truncation


def build(dataset, parts=DEFAULT_PARTS, metrics_module=None, log=print):
    """Prepared frames for one dataset, computed IN MEMORY.

    Returns ``{part: DataFrame}`` for every requested part, plus
    ``"provenance"`` (counts and the top-K truncation ledger).

    Memoised per dataset for the life of the process, so five seeds
    of one dataset pay the matching + featurisation cost once.
    """
    unknown = [p for p in parts if p not in _PARTS]
    if unknown:
        raise PreparedDataError(f"unknown prepared part(s) {unknown}; "
                                f"known: {list(_PARTS)}")
    entry = _PREPARED.get(dataset)
    if entry is None:
        entry = _compute(dataset, metrics_module=metrics_module, log=log)
        _PREPARED[dataset] = entry
    out = {part: entry[part] for part in parts}
    out["provenance"] = entry["provenance"]
    return out


def _compute(dataset, metrics_module, log):
    """The actual match + featurize pass. One raw read, one matching kernel."""
    gt, pred, cfg, truncation = load_raw(dataset, metrics_module, log=log)

    log(f"[{dataset}] matching {len(pred)} rows at lock_tau={schemas.LOCK_TAU}...")
    df = add_tiou_columns(pred, gt, schemas.LOCK_TAU)
    df["pred_len"] = df["t-end"] - df["t-start"]
    df["log_len"] = np.log(df["pred_len"])
    schemas.validate_matched(df)
    base = df[schemas.BASE_COLUMNS]

    log(f"[{dataset}] neighbor features on {len(df)} rows...")
    nbr = add_neighbor_features(df.copy())[["prediction_id"] + NBR_FEATS]

    matched = df["matched_gt_idx"].to_numpy() != -1
    provenance = {
        # what the stream's own export was cut down to, if anything: a
        # natively-200-per-video stream and one truncated to 200 from 2000 are
        # different populations, and the released rows say which
        "truncation": truncation,
        "counts": {
            "n_pred": int(len(df)), "n_gt": int(len(gt)),
            "n_videos": int(df["video-id"].nunique()),
            "n_classes": int(gt["label"].nunique()),
            "n_matched": int(matched.sum()),
        },
    }
    log(f"[{dataset}] prepared: {provenance['counts']}")
    return {"gt": gt, "base": base, "neighbor": nbr, "provenance": provenance}


def load(dataset, parts=("gt", "base"), log=print):
    """Prepared tables: dict name -> DataFrame, plus "provenance"."""
    tables = build(dataset, parts=parts, log=log)
    n = tables["provenance"]["counts"]["n_pred"]
    for part in ("base", "neighbor"):
        if part in tables and len(tables[part]) != n:
            raise PreparedDataError(
                f"{part} row count {len(tables[part])} != prepared {n}")
    return tables


def load_joined(dataset, with_neighbor=False, log=print):
    """base (+gt) joined with the neighbor features on prediction_id."""
    parts = ["gt", "base"]
    if with_neighbor:
        parts.append("neighbor")
    t = load(dataset, parts, log=log)
    df = t["base"]
    n = len(df)
    if "neighbor" in t:
        df = df.merge(t["neighbor"], on="prediction_id", how="inner",
                      validate="1:1")
    if len(df) != n:
        raise PreparedDataError("prepared-data join changed the row count")
    return df, t["gt"], t["provenance"]
