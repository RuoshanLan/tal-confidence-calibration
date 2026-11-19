"""Load dataset and prediction paths from ``configs/datasets.yaml``; every path
there is relative to the framework root (``data/raw/...``)."""
from __future__ import annotations

from pathlib import Path

import yaml

from .paths import CONFIG_ROOT, resolve

_CFG_PATH = CONFIG_ROOT / "datasets.yaml"


def _load() -> dict:
    if not _CFG_PATH.is_file():
        raise FileNotFoundError(
            f"{_CFG_PATH} not found"
        )
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_datasets(names=None) -> dict:
    """dict name -> {anno, pred, source_type, ..., [pre_nms]} with resolved Paths."""
    cfg = _load()["datasets"]
    keys = list(cfg) if names is None else list(names)
    out = {}
    for k in keys:
        entry = {
            "anno": resolve(cfg[k]["anno"]),
            "pred": resolve(cfg[k]["pred"]),
            "source_type": cfg[k].get("source_type", "pickle"),
            # `canonical`: the dataset whose evaluation split this entry uses
            # (itself, or the anchor of a detector stream), so every stream of
            # one dataset shares one video partition; `eval_split`: the
            # annotation `subset` value of that split. Required keys.
            "canonical": cfg[k]["canonical"],
            "eval_split": str(cfg[k]["eval_split"]).lower(),
            # null: the export is already the official population; an int:
            # the per-video top-K truncation of the pair's entry in
            # protocol.OFFICIAL_POPULATION. Required, even when null.
            "top_k": (int(cfg[k]["top_k"]) if cfg[k]["top_k"] is not None
                      else None),
            # the detector that produced `pred` (benchmark spelling) and the
            # feature interface it supports
            "detector": cfg[k]["detector"],
            "feature_interface": cfg[k]["feature_interface"],
        }
        if "pre_nms" in cfg[k]:  # the pre-NMS candidate pool
            entry["pre_nms"] = resolve(cfg[k]["pre_nms"])
            # the pre-NMS export may use a different format from `pred`
            entry["pre_nms_source_type"] = cfg[k]["pre_nms_source_type"]
        out[k] = entry
    return out


def canonical_dataset(name, cfg=None) -> str:
    """The canonical dataset of an entry: itself, or the canonical entry a
    detector stream names. Exactly one hop; a chain is rejected, so all streams
    of a dataset resolve to the same anchor."""
    cfg = cfg if cfg is not None else load_datasets()
    canon = cfg[name]["canonical"]
    if canon not in cfg:
        raise KeyError(
            f"{name}: canonical {canon!r} is not a declared dataset")
    if cfg[canon]["canonical"] != canon:
        raise ValueError(
            f"{name}: canonical {canon!r} is itself a stream entry; the "
            "canonical reference must be exactly one hop")
    return canon


def load_benchmark_pairs() -> list[dict]:
    """The benchmark's (dataset, model) pairs: {dataset, model, pred, anno,
    source_type}, with the annotation taken from the dataset's ``datasets:``
    entry."""
    cfg = _load()
    ds_cfg = cfg["datasets"]
    out = []
    for item in cfg["benchmark_pairs"]:
        dataset = item["dataset"]
        out.append(
            {
                "dataset": dataset,
                "model": item["model"],
                "pred": resolve(item["pred"]),
                "anno": resolve(ds_cfg[dataset]["anno"]),
                "source_type": item.get("source_type", "pickle"),
            }
        )
    return out


def load_annotator_sources() -> dict:
    """dict dataset -> {"derived": released annotation, <subset>: raw file}:
    the raw per-annotator files a released annotation was built from (Ego4D)."""
    cfg = _load()["annotator_sources"]
    return {ds: {k: resolve(v) for k, v in entry.items()} for ds, entry in cfg.items()}


def all_loader_input_paths() -> set[Path]:
    """Every raw input path the loaders reference: ``datasets:`` (anno, pred,
    pre_nms), ``benchmark_pairs:`` (pred, anno) and ``annotator_sources:``."""
    paths_: set[Path] = set()
    for entry in load_datasets().values():
        for key in ("anno", "pred", "pre_nms"):
            if key in entry:
                paths_.add(entry[key])
    for pair in load_benchmark_pairs():
        paths_.add(pair["pred"])
        paths_.add(pair["anno"])
    for entry in load_annotator_sources().values():
        paths_.update(entry.values())
    return paths_
