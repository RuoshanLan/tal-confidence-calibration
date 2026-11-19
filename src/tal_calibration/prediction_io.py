"""Load post-NMS predictions (ActionFormer-style pickle or TadTR / TE-TAD json)
into a ``video-id, t-start, t-end, label, score`` frame."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "infer_source_type",
    "resolve_source_type",
    "load_pickle_predictions",
    "load_json_predictions",
    "load_predictions_from_path",
]

_COLUMNS = ["video-id", "t-start", "t-end", "label", "score"]


def infer_source_type(path: Path) -> str:
    path = Path(path)
    if path.suffix == ".pkl":
        return "pickle"
    if path.suffix == ".json":
        return "json"
    raise ValueError(f"Cannot infer prediction source type from path: {path}")


def resolve_source_type(path: Path, source_type=None) -> str:
    inferred_type = infer_source_type(path)
    if source_type is None:
        return inferred_type
    if source_type != inferred_type:
        raise ValueError(
            f"Configured source_type {source_type!r} does not match path suffix "
            f"for {path} (inferred {inferred_type!r})."
        )
    return source_type


def load_pickle_predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("rb") as f:
        results = pickle.load(f)
    return pd.DataFrame(
        {
            "video-id": results["video-id"],
            "t-start": np.asarray(results["t-start"], dtype=np.float64),
            "t-end": np.asarray(results["t-end"], dtype=np.float64),
            "label": np.asarray(results["label"], dtype=np.int64),
            "score": np.asarray(results["score"], dtype=np.float64),
        }
    )


def load_json_predictions(path: Path, label_map) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if label_map is None:
        raise ValueError(
            f"{path} is a json prediction source and needs a label map; pass "
            "matching.load_label_map(<annotation>) rather than None")
    unmapped = sorted({p["label"] for preds in payload["results"].values()
                       for p in preds} - set(label_map))
    if unmapped:
        raise ValueError(
            f"{path}: {len(unmapped)} label(s) absent from the annotation "
            f"label map, e.g. {unmapped[:3]}")
    rows = []
    for video_id, preds in payload["results"].items():
        for pred in preds:
            rows.append(
                {
                    "video-id": video_id,
                    "t-start": float(pred["segment"][0]),
                    "t-end": float(pred["segment"][1]),
                    "label": int(label_map[pred["label"]]),
                    "score": float(pred["score"]),
                }
            )
    return pd.DataFrame(rows, columns=_COLUMNS)


def load_predictions_from_path(path: Path, label_map, source_type=None) -> pd.DataFrame:
    resolved_source_type = resolve_source_type(path, source_type)
    if resolved_source_type == "pickle":
        return load_pickle_predictions(path)
    return load_json_predictions(path, label_map)
