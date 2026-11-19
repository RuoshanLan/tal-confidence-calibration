"""Video splits: the deterministic 50/25/25 partition of a dataset's
evaluation videos into train / calibration / test.

The population is the canonical dataset's official evaluation split
(``eval_split`` in ``configs/datasets.yaml``, read from the annotation's
``subset`` field), with no hit-status filtering, so every detector stream of
one dataset shares one partition. The partition is a pure function of the
video set and the seed and is recomputed on every run.
"""
import json

import numpy as np
import pandas as pd

from ....datasets import canonical_dataset, load_datasets
from .schemas import REGISTERED_SEEDS

SPLIT_FRACS = (0.5, 0.25, 0.25)
PARTITION_NAMES = {0: "train", 1: "cal", 2: "test"}


def three_way_video_partition(video_ids, seed, fracs=SPLIT_FRACS):
    """video -> 0/1/2: a seeded shuffle of the sorted ids, cut 50/25/25."""
    vids = np.array(sorted(pd.unique(pd.Series(video_ids))))
    np.random.default_rng(seed).shuffle(vids)
    n1 = int(len(vids) * fracs[0])
    n2 = int(len(vids) * (fracs[0] + fracs[1]))
    return {v: (0 if i < n1 else 1 if i < n2 else 2) for i, v in enumerate(vids)}


class SplitRegistryError(RuntimeError):
    pass


def _canonical_cfg(dataset):
    """(canonical name, canonical entry) for a dataset or detector-stream key.

    A stream resolves to its ANCHOR's annotation and eval_split -- never to
    its own -- so every stream of one dataset gets the same partition.
    """
    cfg = load_datasets()
    canon = canonical_dataset(dataset, cfg)
    return canon, cfg[canon]


def eval_split_videos(anno_path, eval_split):
    """Video ids of one annotation subset, read from the database KEYS.

    Not ``load_gt_seg_from_json``: that loader drops any video whose
    annotations all fail the (end - start) >= 1e-3 filter of
    ``remove_duplicate_annotations`` (2 of ActivityNet validation's 4,926
    videos), which would make membership depend on GT row content.
    """
    with open(anno_path, "r", encoding="utf8") as f:
        db = json.load(f)["database"]
    return {str(k) for k, v in db.items()
            if str(v["subset"]).lower() == str(eval_split).lower()}


def all_video_population(dataset):
    """Every video of the canonical dataset's official evaluation split."""
    canon, cfg = _canonical_cfg(dataset)
    vids = eval_split_videos(cfg["anno"], cfg["eval_split"])
    if not vids:
        raise SplitRegistryError(
            f"[{dataset}] empty {cfg['eval_split']!r} split in {cfg['anno']}")
    return vids


def build_table(videos, seeds=REGISTERED_SEEDS):
    """Pure builder: the registered partition of `videos` for every seed."""
    vids = sorted(map(str, videos))
    rows = []
    for seed in seeds:
        for v, p in three_way_video_partition(vids, int(seed)).items():
            rows.append({"video_id": v, "seed": int(seed),
                         "partition": int(p)})
    return pd.DataFrame(rows)
