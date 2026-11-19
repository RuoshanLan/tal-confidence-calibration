"""Prediction-to-GT matching and label-map helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import temporal_iou_with_group

__all__ = [
    "LOCK_TAU",
    "minmax_ratio",
    "add_tiou_columns",
    "load_label_map",
]

LOCK_TAU = 0.1  # uniform matching floor across datasets (not the eval tau)


def minmax_ratio(a, b):
    return np.minimum(a, b) / np.maximum(a, b)


def load_label_map(annotation_json: Path):
    annotation_json = Path(annotation_json)
    with annotation_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    label_map = {}
    for video in data["database"].values():
        for ann in video.get("annotations", []):
            label_map[ann["label"]] = int(ann["label_id"])
    return label_map


def add_tiou_columns(pred_df, gt_df, lock_tau=LOCK_TAU):
    """Attach matching columns to predictions.

    Geometric (argmax-tIoU) GT:  max_tiou, ceiling, best_gt_dur, best_gt_idx.
    Locked one-to-one GT:        matched_tiou, matched_gt_dur, matched_gt_idx.

    Within each (video, label) predictions are matched greedily in score order
    (ties: t-start, t-end, source_row_id); each takes the unclaimed GT of highest
    tIoU, if that tIoU is >= ``lock_tau``. The output is regrouped by
    (video, label), so it is NOT in ``pred_df`` row order. gt_idx columns index
    the (video, label) GT group in gt_df order (-1 = none).
    """
    gt_lookup = {}
    for (video_id, label), group in gt_df.groupby(["video-id", "label"], sort=False):
        gt_lookup[(video_id, int(label))] = (
            group["t-start"].to_numpy(dtype=np.float64),
            group["t-end"].to_numpy(dtype=np.float64),
        )

    sort_cols = ["score", "t-start", "t-end"]
    if "source_row_id" in pred_df.columns:
        sort_cols.append("source_row_id")
    outputs = []
    for (video_id, label), group in pred_df.groupby(
        ["video-id", "label"], sort=False, group_keys=False
    ):
        ordered = group.sort_values(
            sort_cols,
            ascending=[False] + [True] * (len(sort_cols) - 1),
            kind="mergesort",
        ).copy()
        starts = ordered["t-start"].to_numpy(dtype=np.float64)
        ends = ordered["t-end"].to_numpy(dtype=np.float64)
        n = len(ordered)
        max_tiou = np.zeros(n)
        matched_tiou = np.zeros(n)
        ceiling = np.full(n, np.nan)
        best_gt_dur = np.full(n, np.nan)
        best_gt_idx = np.full(n, -1, dtype=np.int64)
        matched_gt_dur = np.full(n, np.nan)
        matched_gt_idx = np.full(n, -1, dtype=np.int64)

        gt_segments = gt_lookup.get((video_id, int(label)))
        if gt_segments is not None and len(gt_segments[0]) > 0:
            gt_starts, gt_ends = gt_segments
            gt_lens = gt_ends - gt_starts
            locked = np.zeros(len(gt_starts), dtype=bool)
            for i in range(n):
                tiou = temporal_iou_with_group(starts[i], ends[i], gt_starts, gt_ends)
                best = int(np.argmax(tiou))
                max_tiou[i] = tiou[best]
                if tiou[best] > 0:
                    pred_len = ends[i] - starts[i]
                    ceiling[i] = min(pred_len, gt_lens[best]) / max(pred_len, gt_lens[best])
                    best_gt_dur[i] = gt_lens[best]
                    best_gt_idx[i] = best
                for gt_idx in np.argsort(tiou)[::-1]:
                    if tiou[gt_idx] < lock_tau:
                        break
                    if not locked[gt_idx]:
                        locked[gt_idx] = True
                        matched_tiou[i] = tiou[gt_idx]
                        matched_gt_dur[i] = gt_lens[gt_idx]
                        matched_gt_idx[i] = int(gt_idx)
                        break

        ordered["max_tiou"] = max_tiou
        ordered["matched_tiou"] = matched_tiou
        ordered["ceiling"] = ceiling
        ordered["best_gt_dur"] = best_gt_dur
        ordered["best_gt_idx"] = best_gt_idx
        ordered["matched_gt_dur"] = matched_gt_dur
        ordered["matched_gt_idx"] = matched_gt_idx
        outputs.append(ordered)
    return pd.concat(outputs, ignore_index=True)
