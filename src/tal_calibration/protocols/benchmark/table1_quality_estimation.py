"""Table 1 score-only localization-quality estimation.

Fills the R^2 cells of Table 1: the out-of-fold score-only R^2 of the
one-to-one localization-quality target q (matched tIoU at tau=0, unmatched rows
zero), read from ``build_pair_context(...)["official_df"]``.

Two independent axes with easily confused names:

``eval_population`` (``EVAL_POPULATIONS``)
    which rows exist: ``official_topk`` or ``lrp_crossfit``, from the shared
    population registry (what every other experiment calls "population").

``population`` (``POPULATIONS``)
    target conditioning within those rows: ``all_predictions`` (unmatched rows
    carry zero; what Table 1 prints) or ``hits_only`` (q > 0, refit).

The estimator's parameters equal those of quality_uncertainty's ``S0_score``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import common, protocol
from .protocol import OFFICIAL_POPULATION, TABLE1_PAIRS

# ==========================================================================
# pair registry
# ==========================================================================
# TABLE1_PAIRS (protocol) holds the 16 pairs of Table 1.

#: Registered pairs outside Table 1: the num_queries=10 TadTR, which Table 1's
#: footnote compares with the num_queries=100 ``anet:tadtr_nq100``.
EXCLUDED_PAIRS = (("anet", "tadtr"),)

#: Table 1's per-video budget per dataset, checked against the protocol at
#: import time.
TABLE1_BUDGETS = {"thumos": 200, "epic_verb": 2000, "epic_noun": 2000,
                  "ego4d": 2000, "anet": 200}


def _validate_registry():
    if len(set(TABLE1_PAIRS)) != len(TABLE1_PAIRS):
        raise RuntimeError("TABLE1_PAIRS has duplicate pairs")
    if len(TABLE1_PAIRS) != 16:
        raise RuntimeError(f"Table 1 is a 16-pair table; registry declares "
                           f"{len(TABLE1_PAIRS)}")
    overlap = set(TABLE1_PAIRS) & set(EXCLUDED_PAIRS)
    if overlap:
        raise RuntimeError(f"pairs both included and excluded: {sorted(overlap)}")
    declared = set(TABLE1_PAIRS) | set(EXCLUDED_PAIRS)
    registered = set(OFFICIAL_POPULATION)
    if declared != registered:
        raise RuntimeError(
            "Table 1 registry drifted from the protocol's official population: "
            f"missing {sorted(registered - declared)}, "
            f"unknown {sorted(declared - registered)}. Add the pair to "
            "TABLE1_PAIRS only together with the paper-scope change; otherwise "
            "list it in EXCLUDED_PAIRS.")
    for ds, mo in TABLE1_PAIRS:
        budget = OFFICIAL_POPULATION[(ds, mo)]["budget_k"]
        if budget != TABLE1_BUDGETS[ds]:
            raise RuntimeError(f"{ds}:{mo}: official budget_k={budget} != "
                               f"Table 1 budget {TABLE1_BUDGETS[ds]}")


_validate_registry()

# ==========================================================================
# protocol
# ==========================================================================
TARGET_COLUMN = "matched_tiou0"

#: target conditioning within the evaluation population (see module docstring)
POPULATIONS = ("all_predictions", "hits_only")

#: evaluation populations from the shared registry: the pool mAP is computed
#: on, and the LRP operating point
EVAL_POPULATIONS = ("official_topk", "lrp_crossfit")

#: the score-only GBR; identical to quality_uncertainty's S0_score arm
GBR_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    random_state=0,
)

N_SPLITS = 5
GROUP_KEY = "video-id"

#: minimum viable population; a smaller one fails the run
MIN_ROWS = 200
MIN_VIDEOS = 5


# ==========================================================================
# input validation
# ==========================================================================
def validate_official_pool(ctx):
    """Pre-run checks on one pair's official pool; raises on violation."""
    ds, mo = ctx["dataset"], ctx["model"]
    df = ctx["official_df"]
    tag = f"{ds}:{mo}"
    required = {"score", GROUP_KEY, "source_row_id", TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{tag}: official pool lacks columns {missing}")
    q = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(q).all():
        raise RuntimeError(f"{tag}: {TARGET_COLUMN} has non-finite values")
    if (q < 0).any() or (q > 1).any():
        raise RuntimeError(f"{tag}: {TARGET_COLUMN} outside [0,1]")
    hits = q > 0
    for col in ("score", GROUP_KEY, "source_row_id"):
        if df[col].isna().any():
            raise RuntimeError(f"{tag}: missing values in `{col}`")
    if df["source_row_id"].duplicated().any():
        raise RuntimeError(f"{tag}: duplicate source_row_id in the official pool")
    budget = OFFICIAL_POPULATION[(ds, mo)]["budget_k"]
    max_per_video = int(df.groupby(GROUP_KEY).size().max())
    if max_per_video > budget:
        raise RuntimeError(f"{tag}: {max_per_video} rows/video exceeds the "
                           f"declared budget {budget}")
    if budget != TABLE1_BUDGETS[ds]:
        raise RuntimeError(f"{tag}: budget {budget} != Table 1 budget "
                           f"{TABLE1_BUDGETS[ds]}")
    return {
        "n_rows": int(len(df)),
        "n_videos": int(df[GROUP_KEY].nunique()),
        "budget_k": int(budget),
        "pool": ctx["official_pool"],
        "source": ctx["source"],
        "hit_rate": float(hits.mean()),
    }


# ==========================================================================
# populations
# ==========================================================================
def eval_frame(ctx, eval_population):
    """The pair's rows for one evaluation population, plus its provenance.

    The mask comes from the shared registry and is applied positionally on
    ``official_df`` (already in ``add_tiou_columns`` order), as in
    ``benchmark_tasks.population_frame``. ``add_tiou_columns`` reorders rows, so
    the realised identity is checked against the registry's sha.

    The target is inherited, not recomputed: the LRP threshold removes a
    low-score suffix of each (video, label) group and the matcher assigns in
    score order, so surviving rows keep their official-pool assignment.
    """
    if eval_population not in EVAL_POPULATIONS:
        raise ValueError(f"unknown eval_population {eval_population!r}")
    mask, prov = ctx["population_view"](eval_population)
    df = ctx["official_df"][mask]
    realised = common.population_identity_sha256(df["source_row_id"])
    if realised != prov["population_identity_sha256"]:
        raise RuntimeError(
            f"{ctx['dataset']}:{ctx['model']}/{eval_population}: the selected "
            f"rows are not the rows the registry resolved ({realised} != "
            f"{prov['population_identity_sha256']}). A mask was applied by "
            "position to a frame in a different row order; map by "
            "`source_row_id` instead.")
    return df, prov


def build_population(frame, population):
    """The rows one workflow fits on: every row (``all_predictions``, unmatched
    rows carry zero) or only q > 0 (``hits_only``, folded and fitted afresh)."""
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}")
    cols = ["source_row_id", GROUP_KEY, "score", TARGET_COLUMN]
    df = frame[cols].copy()
    if population == "hits_only":
        df = df[df[TARGET_COLUMN] > 0]
    return df.reset_index(drop=True)


def check_population(df, population, tag):
    """Viability gate: too small / degenerate populations fail the run."""
    n, nv = len(df), df[GROUP_KEY].nunique()
    q = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if n < MIN_ROWS:
        raise RuntimeError(f"{tag}/{population}: {n} rows < {MIN_ROWS}")
    if nv < MIN_VIDEOS:
        raise RuntimeError(f"{tag}/{population}: {nv} videos < {MIN_VIDEOS}")
    if float(q.var()) <= 0.0:
        raise RuntimeError(f"{tag}/{population}: zero target variance")
    if population == "hits_only" and not (q > 0).all():
        raise RuntimeError(f"{tag}/{population}: non-hit rows leaked in")
    if population == "all_predictions" and not (q >= 0).all():
        raise RuntimeError(f"{tag}/{population}: negative target")


# ==========================================================================
# folds
# ==========================================================================
def fold_ids(videos, n_splits=N_SPLITS):
    """Video-grouped fold id per row from ``protocol.video_folds`` (every row
    held out once; depends only on the group labels and their counts)."""
    videos = np.asarray(videos)
    fold = np.full(len(videos), -1, dtype=np.int64)
    for k, (_tr, te) in enumerate(protocol.video_folds(videos, n_splits)):
        fold[te] = k
    if (fold < 0).any():
        raise RuntimeError("fold assignment left rows without a held-out fold")
    per_fold = np.bincount(fold, minlength=n_splits)
    if (per_fold == 0).any():
        raise RuntimeError(f"empty held-out fold: sizes {per_fold.tolist()}")
    return fold


def video_fold_map(videos, fold):
    """video -> fold, asserting a video never spans two folds (no leakage)."""
    mapping = {}
    for v, k in zip(np.asarray(videos), np.asarray(fold)):
        prev = mapping.setdefault(v, int(k))
        if prev != int(k):
            raise RuntimeError(f"video {v!r} appears in folds {prev} and {k}")
    return mapping


# ==========================================================================
# estimator
# ==========================================================================
def fit_oof(scores, y, fold):
    """OOF predictions of the score-only GBR: each row is predicted once, by a
    model fitted on the other folds' videos."""
    X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64)
    fold = np.asarray(fold)
    pred = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = fold == k
        tr = ~te
        if not te.any():
            raise RuntimeError(f"fold {k} has no held-out rows")
        if not tr.any():
            raise RuntimeError(f"fold {k} has no training rows")
        model = HistGradientBoostingRegressor(**GBR_PARAMS)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite OOF predictions")
    return pred


def r2_score(y, pred):
    """R^2 on the pooled OOF vector (sklearn definition; negatives kept)."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ((y - pred) ** 2).sum() / ss_tot)


# ==========================================================================
# one pair x population
# ==========================================================================
def run_pair_population(ctx, population, pool_info,
                        eval_population="official_topk", eval_df=None,
                        log=print):
    """Fit and score one (pair, eval population, conditioning); returns the
    summary row."""
    ds, mo = ctx["dataset"], ctx["model"]
    tag = f"{ds}:{mo}"
    if eval_df is None:
        eval_df, _ = eval_frame(ctx, eval_population)
    df = build_population(eval_df, population)
    check_population(df, population, f"{tag}/{eval_population}")

    videos = df[GROUP_KEY].to_numpy()
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    scores = df["score"].to_numpy(dtype=np.float64)
    fold = fold_ids(videos, N_SPLITS)
    video_fold_map(videos, fold)  # raises if a video spans two folds

    pred = fit_oof(scores, y, fold)
    r2 = r2_score(y, pred)
    hits = y > 0

    row = {
        "dataset": ds, "model": mo, "eval_population": eval_population,
        "population": population,
        "eval_population_n": int(len(eval_df)),
        "n_rows": int(len(df)), "n_videos": int(df[GROUP_KEY].nunique()),
        "n_folds": int(len(np.unique(fold))),
        "budget_k": pool_info["budget_k"], "official_pool": pool_info["pool"],
        "official_source": pool_info["source"],
        # ---- target
        "target_mean": float(y.mean()), "target_std": float(y.std()),
        "target_min": float(y.min()), "target_max": float(y.max()),
        "hit_rate": float(hits.mean()),
        "zero_target_share": float((y == 0).mean()),
        # ---- the Table 1 cell
        "r2": r2,
    }
    log(f"  [{tag}/{eval_population:<13s}/{population:<15s}] "
        f"n={len(df):>7d} videos={row['n_videos']:>4d} R2={r2:+.4f}")
    return row


def run_pair(ctx, log=print):
    """Every (evaluation population x conditioning) of one pair; each cell has
    its own rows, folds and fit."""
    pool_info = validate_official_pool(ctx)
    rows = []
    for eval_population in EVAL_POPULATIONS:
        df, _ = eval_frame(ctx, eval_population)
        for population in POPULATIONS:
            rows.append(run_pair_population(
                ctx, population, pool_info, eval_population=eval_population,
                eval_df=df, log=log))
    return rows


#: summary CSV column order (stable across runs)
SUMMARY_COLUMNS = [
    "dataset", "model", "eval_population", "population",
    "eval_population_n",
    "n_rows", "n_videos", "n_folds",
    "budget_k", "official_pool", "official_source",
    "target_mean", "target_std", "target_min", "target_max",
    "hit_rate", "zero_target_share",
    "r2",
]


def summary_frame(rows):
    df = pd.DataFrame(rows)
    missing = [c for c in SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"summary rows lack columns {missing}")
    extra = [c for c in df.columns if c not in SUMMARY_COLUMNS]
    if extra:
        raise RuntimeError(f"summary rows carry undeclared columns {extra}")
    return df[SUMMARY_COLUMNS]
