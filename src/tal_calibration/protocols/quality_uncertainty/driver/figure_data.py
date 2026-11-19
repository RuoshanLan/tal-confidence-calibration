"""Data behind the figure panels this chain owns.

``fig1c_hidden_extent.csv`` -- Fig. 1(c): two EPIC-Kitchens-100 verb
validation instances of the class "mix" whose very different durations cap
the tIoU that an ActionFormer prediction of the class's median exported
duration can attain, ``ceiling = min(L_p, L_g) / max(L_p, L_g)``. One row per
shown instance; the class-level columns repeat on both rows.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ....datasets import load_datasets
from ....prediction_io import load_predictions_from_path

FIGURE_CSV = "fig1c_hidden_extent.csv"

FIG1C_DATASET = "epic_verb"
#: verb class 10 of the official EPIC_100_verb_classes mapping
FIG1C_LABEL = 10
FIG1C_LABEL_NAME = "mix"
#: the two instances the panel draws, as (video id, annotated start in s);
#: both are isolated from any neighbouring same-class instance in their window
FIG1C_EXAMPLES = (("P15_06", 411.09), ("P32_06", 303.24))


def fig1c_hidden_extent():
    cfg = load_datasets([FIG1C_DATASET])[FIG1C_DATASET]
    with open(cfg["anno"], "r", encoding="utf8") as f:
        db = json.load(f)["database"]
    split = str(cfg["eval_split"]).lower()
    instances = [(str(vid), float(a["segment"][0]), float(a["segment"][1]))
                 for vid, v in db.items() if str(v["subset"]).lower() == split
                 for a in v["annotations"] if int(a["label_id"]) == FIG1C_LABEL]
    gt_dur = np.array([e - s for _, s, e in instances])

    pred = load_predictions_from_path(Path(cfg["pred"]), None, cfg["source_type"])
    pred_len = (pred["t-end"] - pred["t-start"])[pred["label"] == FIG1C_LABEL]
    pred_len = pred_len.to_numpy(dtype=np.float64)
    median_len = float(np.median(pred_len))

    rows = []
    for vid, start in FIG1C_EXAMPLES:
        hit = [(s, e) for v, s, e in instances
               if v == vid and abs(s - start) < 1e-6]
        if len(hit) != 1:
            raise ValueError(f"fig1c: expected one '{FIG1C_LABEL_NAME}' "
                             f"instance at {vid}@{start}, found {len(hit)}")
        s, e = hit[0]
        dur = e - s
        rows.append({
            "dataset": FIG1C_DATASET, "label_id": FIG1C_LABEL,
            "label_name": FIG1C_LABEL_NAME,
            "n_eval_instances": len(instances),
            "gt_dur_min": float(gt_dur.min()), "gt_dur_max": float(gt_dur.max()),
            "detector": "actionformer", "pred_n": int(len(pred_len)),
            "pred_len_median": median_len,
            "video_id": vid, "gt_start": s, "gt_end": e, "gt_dur": dur,
            "ceiling": min(dur, median_len) / max(dur, median_len),
        })
    return pd.DataFrame(rows)


def write(summary_dir, log=print):
    df = fig1c_hidden_extent()
    path = Path(summary_dir) / FIGURE_CSV
    df.to_csv(path, index=False)
    log(f"[quality_uncertainty] figure data -> {path.name}")
    return path
