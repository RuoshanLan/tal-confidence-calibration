"""The evaluated head: universe -> partition -> LRP selection -> shared frames.

Steps 0-1 of the chain. ``driver.shard.chain_shard`` and every readout table
build their population by calling ``build_head``, so no consumer transcribes
the population construction or drops one of its guards.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import operating_point as SEL
from . import population as POP
from .arms import bundle_for
from .estimators import fold_clean_prior_features


class HeadError(RuntimeError):
    """The head could not be built: the ledger disagrees with the selection."""


@dataclass
class HeadContext:
    """Everything the chain's steps 2-4 and the readout tables consume. Each
    caller gets its own copy of the universe frames."""
    dataset: str
    seed: int
    feature_interface: str
    bundle: Any
    # universe layer
    universe: pd.DataFrame
    gt: pd.DataFrame
    prov: dict
    split_u: Any
    train_vids: Any
    # selection layer
    sel_mask: Any
    ledgers: dict
    pop: pd.DataFrame
    split: Any
    # target and partition masks on the SELECTED rows
    y: np.ndarray
    tr: Any
    ca: Any
    te: Any
    # test-side conveniences
    test_df: pd.DataFrame
    y_te: np.ndarray
    is_hit_te: np.ndarray
    vids_test: pd.Series
    # released column blocks
    counts: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)


def resolve_feature_interface(dataset):
    """The registered feature interface for a detector arm."""
    from ....datasets import load_datasets
    return load_datasets([dataset])[dataset]["feature_interface"]


def build_head(dataset, seed, feature_interface=None, log=print):
    """Build the evaluated head (the LRP-selected population) for one
    (dataset, seed). ``feature_interface=None`` resolves it from the registry.
    """
    if feature_interface is None:
        feature_interface = resolve_feature_interface(dataset)
    bundle = bundle_for(feature_interface)
    universe, gt, prov = POP.build_universe(dataset, log=log)

    split_u, train_vids = POP.partition_for_seed(
        POP.split_table(dataset), seed, universe["video-id"])
    POP.assert_video_disjoint(universe["video-id"], split_u)

    # ---- 1. selection BEFORE any fitting ----
    sel_mask, ledgers = SEL.per_partition_lrp(universe, gt, split_u, log=log)
    pop = universe[sel_mask].reset_index(drop=True)
    split = split_u[sel_mask]
    POP.assert_video_disjoint(pop["video-id"], split)
    for p, pname in SEL.PARTITION_NAMES.items():
        if int((split == p).sum()) != ledgers[pname]["realized_total"]:
            raise HeadError(
                f"[{dataset}/s{seed}] {pname}: selected rows "
                f"{int((split == p).sum())} != ledger realized "
                f"{ledgers[pname]['realized_total']}")

    # class-duration priors from TRAIN-video GT only, shared by every arm
    pop["prior_ratio"], pop["prior_cdf"] = fold_clean_prior_features(
        pop, gt, train_vids)

    y = pop["q"].to_numpy(dtype=np.float64)
    tr, ca, te = (split == 0), (split == 1), (split == 2)
    test_df = pop[te].reset_index(drop=True)
    y_te = y[te]
    is_hit_te = test_df["is_hit"].to_numpy(dtype=bool)
    vids_test = test_df["video-id"].astype(str)

    counts = {
        "universe_n_rows": prov["counts"]["n_rows"],
        "universe_n_videos": prov["counts"]["n_videos"],
        "universe_hit_rate": prov["counts"]["hit_rate"],
        "n_train": int(tr.sum()), "n_cal": int(ca.sum()),
        "n_test": int(te.sum()),
        "n_train_videos": int(pop.loc[tr, "video-id"].nunique()),
        "n_cal_videos": int(pop.loc[ca, "video-id"].nunique()),
        "n_test_videos": int(pop.loc[te, "video-id"].nunique()),
        "n_test_hits": int(is_hit_te.sum()),
        "hit_rate_train": float(pop.loc[tr, "is_hit"].mean()),
        "hit_rate_cal": float(pop.loc[ca, "is_hit"].mean()),
        "hit_rate_test": float(is_hit_te.mean()),
        "unmatched_rate_test": float(1.0 - is_hit_te.mean()),
    }
    for pname, ledger in ledgers.items():
        counts[f"sel_realized_{pname}"] = ledger["realized_total"]
    # Per-video top-K defines the population upstream of the universe (no row
    # is excluded); these two columns tell a native 200/video stream apart
    # from one cut to 200.
    trunc = prov.get("truncation") or {}
    shared = {
        **counts,
        "n_rows_truncated": int(trunc.get("n_rows_dropped", 0)),
        "native_max_per_video": int(trunc.get("max_rows_per_video_before", 0)),
        # the target is exactly zero off the matched set
        "unmatched_target_max_abs": float(
            np.abs(y[~pop["is_hit"].to_numpy()]).max())
        if (~pop["is_hit"]).any() else 0.0,
        "tie_policy": next(iter(ledgers.values()))["tie_policy"],
    }

    return HeadContext(
        dataset=dataset, seed=seed,
        feature_interface=feature_interface, bundle=bundle,
        universe=universe, gt=gt, prov=prov,
        split_u=split_u, train_vids=train_vids,
        sel_mask=sel_mask, ledgers=ledgers, pop=pop, split=split,
        y=y, tr=tr, ca=ca, te=te,
        test_df=test_df, y_te=y_te, is_hit_te=is_hit_te, vids_test=vids_test,
        counts=counts, shared=shared,
    )
