"""Candidate universe, direct-q target, and the seed's video partition.

``build_universe`` returns the CANDIDATE UNIVERSE: every post-NMS row of the
exported stream, with the target and the complete observable feature
interface. The LRP selection then thins each TRAIN/CAL/TEST side of it before
any fitting (``head.build_head``).

Target (Sec. 3; tau=0 one-to-one matching):

    q_i = matched_tiou_i   if the matcher locked row i
        = 0                otherwise (exactly zero, not NaN)

The partition covers every video of the dataset's official evaluation split
(``data.splits``), with no hit-status filtering; an uncovered video is a hard
error, never an exclusion.
"""
import numpy as np
import pandas as pd

from ..data import cloud
from ..data import splits
from ..data import prepared_data
from ..data.cloud import CLOUD_FEATS, CLOUD_X_FEATS

#: the chain's single selection head
POPULATION_NAME = "lrp"
TARGET_NAME = "q_o2o_tau0"


class PopulationError(RuntimeError):
    """The population cannot be built with the guarantees this chain requires."""


def attach_target(df, dataset="?"):
    """Attach ``q``/``is_hit`` and prove the target invariants (pure)."""
    is_hit = df["matched_gt_idx"].to_numpy() != -1
    q = df["matched_tiou"].to_numpy(dtype=np.float64)
    if not np.array_equal(is_hit, q > 0):
        raise PopulationError(
            f"[{dataset}] matched flag and positive tIoU disagree under the "
            "tau=0 lock")
    if (q[~is_hit] != 0.0).any():
        raise PopulationError(
            f"[{dataset}] unmatched rows carry a non-zero target")
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise PopulationError(f"[{dataset}] target outside [0,1]")
    matched_dur = df["matched_gt_dur"].to_numpy(dtype=np.float64)
    if not np.isfinite(matched_dur[is_hit]).all():
        raise PopulationError(
            f"[{dataset}] matched rows carry a non-finite matched_gt_dur; "
            "the mechanism oracle would be undefined")
    df = df.copy()
    df["q"] = q
    df["is_hit"] = is_hit
    return df


#: dataset -> (df, gt, provenance); process-local memo shared by the seeds
_UNIVERSE = {}
#: dataset -> registered partition table (all seeds)
_SPLITS = {}


def clear_memo():
    """Drop the in-process memos (tests; long-lived processes)."""
    _UNIVERSE.clear()
    _SPLITS.clear()


def build_universe(dataset, log=print):
    """Return (df, gt, provenance) for the candidate universe of `dataset`.

    `df` carries every post-NMS row, the target ``q``, the matched flag
    ``is_hit`` and the observable features; `provenance` carries the universe
    counts and the top-K truncation ledger. Each caller gets its own copy of
    the frames.
    """
    hit = _UNIVERSE.get(dataset)
    if hit is not None:
        df, gt, provenance = hit
        return df.copy(), gt.copy(), provenance
    df, gt, prov_t = prepared_data.load_joined(dataset, with_neighbor=True,
                                               log=log)

    # Only detectors on the full interface get the pre-NMS cloud columns, so
    # a reduced arm that names a cloud feature fails its feature check.
    from ....datasets import load_datasets
    feature_interface = load_datasets([dataset])[dataset]["feature_interface"]
    n_before = len(df)
    if feature_interface == "full_observable":
        cloud_df = cloud.build(dataset, log=log)
        if (sorted(cloud_df["prediction_id"].astype(int))
                != sorted(df["prediction_id"].astype(int))):
            raise PopulationError(
                f"[{dataset}] the cloud part does not cover every post-NMS row")
        df = df.merge(cloud_df[["prediction_id"] + CLOUD_FEATS + CLOUD_X_FEATS],
                      on="prediction_id", how="inner", validate="1:1")
        if len(df) != n_before:
            raise PopulationError(
                f"[{dataset}] cloud join changed the row count "
                f"{n_before} -> {len(df)}")

    df = attach_target(df, dataset)

    registry_videos = set(map(str, split_table(dataset)["video_id"]))
    vids = df["video-id"].astype(str)
    covered = vids.isin(registry_videos).to_numpy()
    if not covered.all():
        missing = sorted(set(vids[~covered]))[:3]
        raise PopulationError(
            f"[{dataset}] {int((~covered).sum())} rows in videos outside the "
            f"evaluation split (e.g. {missing}); nothing may be excluded")
    df = df.reset_index(drop=True)

    provenance = {
        # a native 200/video stream and one cut to 200 from 2000 are
        # different populations; the released rows say which
        "truncation": prov_t.get("truncation"),
        "counts": {
            "n_rows": int(len(df)),
            "n_videos": int(df["video-id"].nunique()),
            "hit_rate": float(df["is_hit"].mean()),
        },
    }
    _UNIVERSE[dataset] = (df, gt, provenance)
    return df.copy(), gt.copy(), provenance


def split_table(dataset):
    """The registered partition of `dataset`'s evaluation videos, all seeds.
    Every detector stream of one dataset gets the same table."""
    table = _SPLITS.get(dataset)
    if table is None:
        table = splits.build_table(splits.all_video_population(dataset))
        _SPLITS[dataset] = table
    return table


def partition_for_seed(split_table, seed, video_series):
    """(split array, train video ids) for one seed. Every universe row must
    have a partition — a missing video is an error, never an implicit drop."""
    sub = split_table[split_table["seed"] == seed]
    if sub.empty:
        raise PopulationError(f"split registry has no seed {seed}")
    mapping = {str(v): int(p) for v, p in zip(sub["video_id"], sub["partition"])}
    split = video_series.astype(str).map(mapping)
    if split.isna().any():
        raise PopulationError(
            f"{int(split.isna().sum())} rows have no partition for seed {seed}")
    split = split.to_numpy(dtype=np.int64)
    if set(np.unique(split).tolist()) != {0, 1, 2}:
        raise PopulationError(
            f"seed {seed}: partitions present {sorted(set(split.tolist()))} != "
            "{0, 1, 2} (empty train/cal/test)")
    train_vids = {str(v) for v, p in mapping.items() if p == 0}
    return split, train_vids


def assert_video_disjoint(video_series, split):
    """No video may appear in more than one partition."""
    vids = pd.Series(video_series).astype(str).to_numpy()
    per_video = pd.DataFrame({"v": vids, "p": split}).groupby("v")["p"].nunique()
    bad = per_video[per_video > 1]
    if len(bad):
        raise PopulationError(
            f"{len(bad)} videos appear in multiple partitions, e.g. "
            f"{list(bad.index[:3])}")
    return True
