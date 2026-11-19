"""TAL mAP on the vendored ActionFormer evaluator
(``third_party.actionformer_metrics``, MIT), whose GT loader and AP routine are
re-exported here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .third_party.actionformer_metrics import (  # noqa: F401  (re-export)
    compute_average_precision_detection,
    load_gt_seg_from_json,
)

__all__ = [
    "evaluate_predictions",
    "compute_average_precision_detection",
    "load_gt_seg_from_json",
]


def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame, tiou_thresholds):
    """Per-class AP -> per-tIoU mAP -> average mAP (percent).

    Classes are grouped on their original label ids, so a non-contiguous label
    space (EPIC) needs no remapping; prediction classes without GT enter no AP.
    """
    gt_by_label = gt_df.groupby("label")
    pred_by_label = pred_df.groupby("label")
    classes = sorted(gt_df["label"].unique())
    ap = np.zeros((len(tiou_thresholds), len(classes)), dtype=np.float64)
    for i, class_id in enumerate(classes):
        gt_group = gt_by_label.get_group(class_id).reset_index(drop=True)
        try:
            pred_group = pred_by_label.get_group(class_id).reset_index(drop=True)
        except KeyError:
            pred_group = pd.DataFrame(
                columns=["video-id", "t-start", "t-end", "label", "score"]
            )
        ap[:, i] = compute_average_precision_detection(
            ground_truth=gt_group,
            prediction=pred_group,
            tiou_thresholds=np.asarray(tiou_thresholds, dtype=np.float64),
        )
    per_tiou = ap.mean(axis=1)
    result = {"average_mAP": float(per_tiou.mean() * 100.0)}
    for thr, value in zip(tiou_thresholds, per_tiou):
        result[f"mAP@{thr:.1f}"] = float(value * 100.0)
    return result
