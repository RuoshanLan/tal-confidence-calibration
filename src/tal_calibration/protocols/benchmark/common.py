"""Loading, row identity, pool labelling and OOF confidence fitting shared by
the benchmark and fragmentation experiments; each experiment chooses its
targets, arms and column families."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ...evaluation import load_gt_seg_from_json
from ...matching import load_label_map
from ...prediction_io import load_predictions_from_path
from . import protocol
from .protocol import CALIB_TAUS, label_multi_tau, oof_calibrate, tau_key


def population_identity_sha256(source_row_ids):
    """Population identity: sha256 of the ascending source_row_id list,
    JSON-encoded with compact separators."""
    ids = sorted(int(i) for i in source_row_ids)
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_pair(dataset, model, cfg_ds, pair_cfg):
    """Load GT + predictions for one pair; returns raw rows with source_row_id."""
    anno = str(cfg_ds[dataset]["anno"])
    gt = load_gt_seg_from_json(anno, split=protocol.EVAL_SPLIT[dataset])
    if len(gt) == 0:
        raise RuntimeError(
            f"empty GT for {dataset} split={protocol.EVAL_SPLIT[dataset]}")
    label_map = (load_label_map(Path(anno))
                 if pair_cfg["source_type"] == "json" else None)
    pred = load_predictions_from_path(Path(pair_cfg["pred"]), label_map,
                                      pair_cfg["source_type"])
    pred = protocol.add_source_row_id(pred)
    coverage = protocol.check_pred_coverage(dataset, cfg_ds,
                                            pred["video-id"].unique(), gt)
    return {"gt": gt, "pred": pred, "coverage": coverage, "anno": anno}


def labeled_pool(pool, gt, dataset, model, budget_k, include_frag=False, log=print):
    """Label an already-capped pool against GT at every tau of the dataset grid
    (``include_frag=False``: official matcher only)."""
    t0 = time.time()
    tag = f"{dataset}:{model}"
    df = label_multi_tau(pool, gt, CALIB_TAUS[dataset], include_frag=include_frag)
    if int(df.groupby("video-id").size().max()) > budget_k:
        raise RuntimeError(f"{tag}: labeled pool exceeds budget k={budget_k}")
    log(f"[{tag}] labeled {len(df)} rows "
        f"({df['video-id'].nunique()} videos) in {time.time()-t0:.0f}s")
    return df


def ensure_conf_columns(df, dataset, model, arms, log=print,
                        target_prefix="tp_credit", conf_prefix="conf",
                        diag_prefix=""):
    """Fit missing OOF confidence columns for the ``arms`` x tau grid, in place.

    The target is the per-tau label family ``target_prefix`` (default
    ``tp_credit``); ``conf_prefix`` and ``diag_prefix`` keep differently fitted
    families apart. Returns ``{f"{diag_prefix}{arm}_{tau}": diag}`` for the fits
    made by this call.
    """
    videos = df["video-id"].to_numpy()
    labels = df["label"].to_numpy()
    scores = df["score"].to_numpy(dtype=np.float64)
    diag_all = {}
    for tau in CALIB_TAUS[dataset]:
        y = df[f"{target_prefix}_{tau_key(tau)}"].to_numpy(dtype=np.int64)
        for arm in arms:
            col = f"{conf_prefix}_{arm}_{tau_key(tau)}"
            if col in df.columns:
                continue
            t0 = time.time()
            conf, diag = oof_calibrate(scores, y, videos, labels, arm,
                                       return_diag=True)
            df[col] = conf
            if diag:
                diag_all[f"{diag_prefix}{arm}_{tau_key(tau)}"] = diag
            if arm != "raw":
                log(f"  [{dataset}:{model}] tau={tau:g} arm={arm:<9s} "
                    f"target={target_prefix} fit {time.time()-t0:.0f}s")
    return diag_all
