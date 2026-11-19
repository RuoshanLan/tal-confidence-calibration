"""Build ``data/raw/annotations/ego4d/ego4d.json`` from the official Ego4D-MQ
annotations (Episodic Memory Moments Queries v1: ``moments_train.json`` and
``moments_val.json``).

The annotation rules are those of ActionFormer's
``tools/convert_ego4d_trainval.py``: per clip, every annotator's primary moments
of at least 0.25 s, times relative to the clip start and rounded to 2 decimals;
clips with fps outside [10, 100] or without such a moment are skipped. Unlike
that script, no feature-grid alignment is applied (the released file pairs with
EgoVLP's clip-level features), and no feature files are read.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import load_annotator_sources

DATASET = "ego4d"
#: ActionFormer's tools/ego4d_label_map.txt, verbatim (THIRD_PARTY_NOTICES.md)
LABEL_MAP = Path(__file__).resolve().parent / "third_party" / "ego4d_label_map.txt"
MIN_MOMENT_SEC = 0.25
MIN_FPS, MAX_FPS = 10, 100
SUBSETS = {"train": "training", "val": "validation"}


def load_label_map(path=LABEL_MAP):
    """{label: label_id} from the tab-separated map. Labels are kept exactly as
    the moments files spell them (those with commas carry literal quotes)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label_id, label = line.strip().split("\t")
            out[label] = int(label_id)
    return out


def build_database(videos, label_map):
    """ActionFormer-style ``database`` for Ego4D ``videos`` entries, clips in
    input order."""
    database = {}
    for video in videos:
        subset = SUBSETS.get(video["split"], video["split"])
        for clip in video["clips"]:
            ss = max(float(clip["video_start_sec"]), 0)
            sf = max(int(clip["video_start_frame"]), 0)
            duration = float(clip["video_end_sec"]) - ss
            fps = (int(clip["video_end_frame"]) - sf) / duration
            if fps < MIN_FPS or fps > MAX_FPS:
                continue
            annotations = []
            for annotator in clip["annotations"]:
                for item in annotator["labels"]:
                    if not item["primary"]:
                        continue
                    start = item["video_start_time"] - ss
                    end = item["video_end_time"] - ss
                    if end - start < MIN_MOMENT_SEC:
                        continue
                    annotations.append({
                        "label": item["label"],
                        "segment": [round(start, 2), round(end, 2)],
                        "segment(frames)": [item["video_start_frame"] - sf,
                                            item["video_end_frame"] - sf],
                        "label_id": label_map[item["label"]],
                    })
            if annotations:
                database[clip["clip_uid"]] = {
                    "subset": subset,
                    "duration": round(duration, 2),
                    "fps": round(fps, 2),
                    "annotations": annotations,
                }
    return database


def build_ego4d_json(training, validation, label_map=None):
    """The text of ego4d.json: training videos first, then validation.
    `label_map` defaults to LABEL_MAP."""
    videos = []
    for path in (training, validation):
        with open(path, "r", encoding="utf-8") as f:
            videos += json.load(f)["videos"]
    labels = load_label_map(LABEL_MAP if label_map is None else label_map)
    return json.dumps({"version": "v1", "database": build_database(videos, labels)})


def write_ego4d_json(out=None, log=print):
    """Rebuild ego4d.json and write it to `out` (default: the configured path).
    Returns 0."""
    src = load_annotator_sources()[DATASET]
    data = build_ego4d_json(src["training"], src["validation"]).encode("utf-8")
    out = Path(out) if out is not None else src["derived"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    log(f"wrote {out} ({len(data)} bytes)")
    return 0
