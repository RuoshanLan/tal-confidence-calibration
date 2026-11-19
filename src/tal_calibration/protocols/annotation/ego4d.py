"""Ego4D-MQ real-annotator worlds for the annotation audit (Table 2, supp. Sec. D).

The ActivityNet chain perturbs the boundaries of a FIXED instance set with a
generator fitted to DETAD. Ego4D-MQ needs no generator: each clip was labeled by
up to three annotators, and the released annotation (``ego4d.json``, also the
ActionFormer training annotation) is the UNION of their primary moments. An
alternative world keeps, per clip, the moments of ONE annotator drawn uniformly
from that clip's annotators. An annotator who marked no primary moment is a
legitimate draw and leaves the clip without GT in that world.

Everything else is the ANet chain, called rather than copied: the greedy
matcher (``lib.compute_quality_targets`` / ``compute_credit_targets``), pooled
OOF isotonic on crc32 clip folds, ``credit_ece`` / ``laece0``, the
``lrp_crossfit`` population (at Ego4D's own anchor tau),
``analysis.claim_a_floor``, ``claim_b_icc`` and ``target_shift``. A world is a
row subset of the union GT, so a GT row never moves: every target change is a
reassignment or a loss. Table 2 reads the loosest tau of the Ego4D grid, the
same rule that puts the ANet row at .5.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import analysis as A
from . import lib as L
from ..benchmark import protocol as P0
from ...datasets import load_annotator_sources, load_datasets
from ...matching import add_tiou_columns
from ...prediction_io import load_pickle_predictions

DATASET = "ego4d"
EVAL_SPLIT = "validation"
#: the benchmark's Ego4D tIoU grid; every tau is computed, Table 2 reads AUDIT_TAU
AUDIT_TAUS = tuple(float(t) for t in P0.CALIB_TAUS[DATASET])
AUDIT_TAU = min(AUDIT_TAUS)
#: the benchmark's primary tau, which anchors the LRP population (Table 1 too)
LRP_ANCHOR_TAU = float(P0.HEADLINE_TAU[DATASET])
N_WORLDS = L.N_WORLDS
WORLD_SEEDS = L.WORLD_SEEDS
#: ActionFormer's tools/convert_ego4d_trainval.py: drop moments < 0.25 s, round to 2 decimals
MIN_MOMENT_SEC = 0.25
ROUND_DIGITS = 2


# --------------------------------------------------------------------------
# raw multi-annotator tables + the parity gate against the released union
# --------------------------------------------------------------------------
def _derived_db():
    path = load_annotator_sources()[DATASET]["derived"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["database"]


def annotator_tables(raw_videos, derived_db, subset):
    """{clip_uid: [(annotator_uid, [(label_id, start, end), ...]), ...]}.

    Only clips present in `derived_db` under `subset`. The conversion rule is
    ActionFormer's: primary moments only, at least 0.25 s, times relative to
    max(video_start_sec, 0), rounded to 2 decimals. The released file carries no
    feature-grid alignment, which `parity` proves rather than assumes."""
    label_id = {a["label"]: int(a["label_id"])
                for v in derived_db.values() for a in v["annotations"]}
    out = {}
    for video in raw_videos:
        for clip in video["clips"]:
            cid = clip["clip_uid"]
            if cid not in derived_db or derived_db[cid]["subset"] != subset:
                continue
            ss = max(float(clip["video_start_sec"]), 0.0)
            per_ann = []
            for ann in clip["annotations"]:
                rows = []
                for item in ann["labels"]:
                    if not item["primary"]:
                        continue
                    s = item["video_start_time"] - ss
                    e = item["video_end_time"] - ss
                    if e - s < MIN_MOMENT_SEC:
                        continue
                    rows.append((label_id[item["label"]],
                                 round(s, ROUND_DIGITS), round(e, ROUND_DIGITS)))
                per_ann.append((ann["annotator_uid"], rows))
            out[cid] = per_ann
    return out


def parity(tables, derived_db, subset):
    """Union of the rebuilt per-annotator moments vs the released file.

    `ordered` compares the annotator-by-annotator concatenation with the
    released list as sequences; it is what licenses reading a union row's
    annotator off its position. `multiset` is the weaker order-free check."""
    want = {k for k, v in derived_db.items() if v["subset"] == subset}
    bad_ordered = bad_multiset = 0
    n_inst = 0
    hist = Counter()
    for cid in want:
        if cid not in tables:
            bad_ordered += 1
            bad_multiset += 1
            continue
        union = [r for _, rows in tables[cid] for r in rows]
        ref = [(int(a["label_id"]), float(a["segment"][0]), float(a["segment"][1]))
               for a in derived_db[cid]["annotations"]]
        n_inst += len(ref)
        hist[len(tables[cid])] += 1
        bad_ordered += union != ref
        bad_multiset += Counter(union) != Counter(ref)
    return {"subset": subset, "n_clips": len(want), "n_instances": n_inst,
            "clips_missing_from_raw": len(want - set(tables)),
            "mismatch_ordered": int(bad_ordered),
            "mismatch_multiset": int(bad_multiset),
            "clips_with_1_annotator": hist.get(1, 0),
            "clips_with_2_annotators": hist.get(2, 0),
            "clips_with_3_annotators": hist.get(3, 0),
            "clips_with_4plus_annotators": sum(v for k, v in hist.items() if k >= 4)}


def load_tables_with_parity():
    """Validation tables plus the parity rows of BOTH subsets (training too:
    that is what makes the released world the detector's training world)."""
    src = load_annotator_sources()[DATASET]
    derived = _derived_db()
    tables, rows = {}, []
    for subset in ("training", "validation"):
        with open(src[subset], "r", encoding="utf-8") as f:
            raw = json.load(f)["videos"]
        t = annotator_tables(raw, derived, subset)
        rows.append(parity(t, derived, subset))
        tables[subset] = t
    return tables[EVAL_SPLIT], rows, derived


# --------------------------------------------------------------------------
# predictions / union GT (the released world)
# --------------------------------------------------------------------------
def load_predictions():
    """Ego4D post-NMS pool, prepared exactly as lib.load_predictions prepares
    the ANet one: t-start clipped to >= 0, pkl row order kept as pred_row_id."""
    df = load_pickle_predictions(load_datasets([DATASET])[DATASET]["pred"])
    df["t-start"] = np.clip(df["t-start"].to_numpy(), 0.0, None)
    df = df.reset_index(drop=True)
    df["pred_row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def load_union_gt(tables, derived_db):
    """The released world as a lib.load_gt-shaped frame, in released order,
    sanitized exactly like load_gt, plus `annotator` (index within its clip)
    and `n_annotators` (every annotator of the clip, including those with no
    primary moment). Row order equals ego4d.json order because `parity`
    passed in ordered mode."""
    cols = {k: [] for k in ("video-id", "t-start", "t-end", "label", "duration",
                            "annotator", "n_annotators")}
    for cid, v in derived_db.items():
        if v["subset"] != EVAL_SPLIT:
            continue
        dur = float(v["duration"])
        anns = tables[cid]
        for k, (_, rows) in enumerate(anns):
            for (lab, a, b) in rows:
                s, e = sorted((float(a), float(b)))
                s = min(max(s, 0.0), dur)
                e = min(max(e, 0.0), dur)
                if e - s < 1.0e-3:
                    e = min(s + 1.0e-3, dur)
                for key, val in (("video-id", cid), ("t-start", s), ("t-end", e),
                                 ("label", int(lab)), ("duration", dur),
                                 ("annotator", k), ("n_annotators", len(anns))):
                    cols[key].append(val)
    gt = pd.DataFrame(cols)
    gt["instance_id"] = np.arange(len(gt), dtype=np.int64)
    gt["len"] = gt["t-end"].to_numpy() - gt["t-start"].to_numpy()
    return gt


# --------------------------------------------------------------------------
# worlds + targets
# --------------------------------------------------------------------------
def clip_order(gt):
    """Clips in released order with their annotator counts."""
    first = gt.drop_duplicates("video-id")
    return first["video-id"].to_numpy(), first["n_annotators"].to_numpy()


def sample_worlds(gt, seed, n_worlds=N_WORLDS):
    """keep (n_worlds, n_gt) bool and draws (n_worlds, n_clips): world w keeps,
    in every clip, the rows of ONE annotator drawn uniformly from that clip's
    annotators. Draw order is world-major, clip-minor in released order."""
    clips, n_ann = clip_order(gt)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_worlds, len(clips)), dtype=np.int64)
    for w in range(n_worlds):
        for j, n in enumerate(n_ann):
            draws[w, j] = rng.integers(int(n))
    clip_idx = pd.Index(clips).get_indexer(gt["video-id"].to_numpy())
    keep = gt["annotator"].to_numpy()[None, :] == draws[:, clip_idx]
    return keep, draws


def compute_world_targets(pred, gt, keep, taus=AUDIT_TAUS):
    """Q / MATCH_ID / credit in the ANet chain's shapes: row 0 = released
    (union) world, row w = world w. MATCH_ID holds union instance ids."""
    pred = pred.sort_values("pred_row_id").reset_index(drop=True)
    n_pred, n_worlds = len(pred), keep.shape[0]
    Q = np.zeros((n_worlds + 1, n_pred), dtype=np.float32)
    MID = np.full((n_worlds + 1, n_pred), -1, dtype=np.int64)
    credit = {t: np.zeros((n_worlds + 1, n_pred), dtype=np.int8) for t in taus}
    gs = gt["t-start"].to_numpy(np.float64)
    ge = gt["t-end"].to_numpy(np.float64)
    gv = gt["video-id"].to_numpy()
    gl = gt["label"].to_numpy()
    gi = gt["instance_id"].to_numpy()
    for w in range(n_worlds + 1):
        rows = np.arange(len(gt)) if w == 0 else np.flatnonzero(keep[w - 1])
        plan = L.build_match_plan(pred, gv[rows], gl[rows])
        Q[w], MID[w] = L.compute_quality_targets(
            pred, gs[rows], ge[rows], gv[rows], gl[rows], gi[rows],
            plan=plan, return_match_id=True)
        ctg = L.compute_credit_targets(pred, gs[rows], ge[rows], gv[rows],
                                       gl[rows], taus, plan=plan)
        for t in taus:
            credit[t][w] = ctg[t]
    return Q, MID, credit


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------
def _transferred(scores, folds, m, Q, credit, worlds):
    """Transferred (released-fit) credit-ECE and LaECE_0 on population m over
    `worlds`, with claim_a_floor's estimator (fit on the population, OOF)."""
    cr = credit[AUDIT_TAU]
    s, f = scores[m], folds[m]
    cal0 = L.oof_isotonic(s, cr[0][m].astype(np.float64), f)
    calq = L.oof_isotonic(s, Q[0][m], f)
    out = {}
    out["credit_transferred"] = float(np.mean(
        [L.credit_ece(cr[w][m].astype(bool), cal0) for w in worlds])) / 100.0
    out["laece0_transferred"] = float(np.mean(
        [L.laece0(calq, Q[w][m]) for w in worlds]))
    return out


def compute(log=print):
    """Transferred errors on the LRP population for every seed; the first
    (reference) seed also gives the Table 2 rows, the ICC and the target shift."""
    seeds = list(WORLD_SEEDS)
    log("[ego4d] raw annotator tables + parity vs the released union")
    tables, parity_rows, derived = load_tables_with_parity()
    for r in parity_rows:
        log(f"[ego4d] parity {r['subset']}: clips={r['n_clips']} "
            f"instances={r['n_instances']} mismatched(ordered)={r['mismatch_ordered']}")
    pred = load_predictions()
    gt = load_union_gt(tables, derived)
    pred_ord = pred.sort_values("pred_row_id").reset_index(drop=True)
    folds = L.crc32_fold(pred_ord["video-id"].to_numpy())
    scores = pred_ord["score"].to_numpy(np.float64)
    masks = L.population_masks(pred_ord, gt, tau=LRP_ANCHOR_TAU)
    log(f"[ego4d] preds={len(pred_ord)} union GT={len(gt)} "
        f"lrp_crossfit={int(masks['lrp_crossfit'].sum())} (anchor tau={LRP_ANCHOR_TAU})")

    res = {"parity": parity_rows, "gt": gt, "pred": pred_ord, "masks": masks,
           "folds": folds, "scores": scores}
    per_seed = []
    for sd in seeds:
        keep, _ = sample_worlds(gt, sd)
        Q, MID, credit = compute_world_targets(pred_ord, gt, keep)
        worlds = range(1, N_WORLDS + 1)
        m = masks["lrp_crossfit"]
        fl = _transferred(scores, folds, m, Q, credit, worlds)
        per_seed.append({
            "seed": sd, "tau": AUDIT_TAU,
            "credit_transferred": round(fl["credit_transferred"], 4),
            "laece0_transferred": round(fl["laece0_transferred"], 4)})
        log(f"[ego4d] seed {sd}: credit transferred={fl['credit_transferred']:.4f}  "
            f"laece0 transferred={fl['laece0_transferred']:.4f}")
        if sd == seeds[0]:
            log("[ego4d] reference seed: released-fit vs refit, ICC, target shift")
            res.update(Q=Q, credit=credit)
            res["claim_a"] = A.claim_a_floor(pred_ord, gt, Q, credit, folds,
                                             log=log, masks=masks, taus=AUDIT_TAUS)
            res["claim_b"] = A.claim_b_icc(Q, MID, log=log)
            res["target_shift"] = A.target_shift(Q, credit, masks, taus=AUDIT_TAUS)
    res["multiseed"] = _multiseed_rows(per_seed, res)
    return res


def _multiseed_rows(per_seed, res):
    rows = list(per_seed)
    num = [k for k in per_seed[0] if k not in ("seed", "tau")]
    agg = {"seed": "AGG_seeds", "tau": AUDIT_TAU, "n_seeds": len(per_seed)}
    for k in num:
        vals = [r[k] for r in per_seed]
        agg[k] = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
    rows.append(agg)
    # Clip-clustered bootstrap of the transferred error on the reference seed,
    # with the ANet chain's estimator (fixed OOF confidences, resampled rows,
    # same replicate draws for both metrics), over worlds 6-10 -- the world
    # range of the ActivityNet intervals printed beside these.
    m = res["masks"]["lrp_crossfit"]
    idx = np.flatnonzero(m)
    s, f = res["scores"][m], res["folds"][m]
    cr = res["credit"][AUDIT_TAU]
    Q = res["Q"]
    cal0 = L.oof_isotonic(s, cr[0][m].astype(np.float64), f)
    calq = L.oof_isotonic(s, Q[0][m], f)
    vids = res["pred"]["video-id"].to_numpy()[idx]
    wname, ws = "6-10", range(6, N_WORLDS + 1)
    y = {w: cr[w][idx].astype(bool) for w in ws}
    q = {w: Q[w][idx] for w in ws}
    cb, lb = [], []
    for sub in L.video_bootstrap_indices(vids):
        cb.append(np.mean([L.credit_ece(y[w][sub], cal0[sub]) for w in ws]) / 100.0)
        lb.append(np.mean([L.laece0(calq[sub], q[w][sub]) for w in ws]))
    for name, vals in (("credit_transferred", cb), ("laece0_transferred", lb)):
        lo, hi, neff = L.percentile_ci(vals)
        rows.append({"seed": "BOOT_ci", "tau": AUDIT_TAU, "stat": name,
                     "eval_worlds": wname, "boot_mean": round(float(np.mean(vals)), 4),
                     "ci_lo_2.5": round(lo, 4), "ci_hi_97.5": round(hi, 4),
                     "n_boot": neff})
    return rows


# --------------------------------------------------------------------------
# invariants (hard): any failure aborts before anything is written
# --------------------------------------------------------------------------
class InvariantError(AssertionError):
    pass


def validate(res):
    # the per-annotator union equals the released annotation clip for clip
    # (supp. Sec. D states this; 00_parity.csv carries the counts)
    for r in res["parity"]:
        if r["mismatch_ordered"] or r["clips_missing_from_raw"]:
            raise InvariantError(
                f"per-annotator union != released annotation ({r['subset']}): {r}")
    # the released-world credit at the anchor tau must equal the matching the
    # LRP population was fitted on, row for row (catches a mask applied in the
    # wrong row order)
    pred, gt = res["pred"], res["gt"]
    dfm = add_tiou_columns(pred, gt, lock_tau=LRP_ANCHOR_TAU)
    hit = np.zeros(len(pred), dtype=bool)
    hit[dfm["pred_row_id"].to_numpy()[dfm["matched_tiou"].to_numpy() > 0]] = True
    if not np.array_equal(hit, res["credit"][LRP_ANCHOR_TAU][0].astype(bool)):
        raise InvariantError("released-world credit at the anchor tau != the "
                             "matching the LRP population was fitted on")
    return True
