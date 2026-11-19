"""ActivityNet-1.3 annotation audit: compute (reference seed) + multi-seed CI.

Pure compute + an in-memory result dict; the experiment entry writes the files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis as A
from . import steps as S
from . import lib as L

#: Population the multi-seed / bootstrap robustness stage reports on: the
#: operating point the paper reports. HEADLINE_FAMILY only labels the CSV:
#: pooled isotonic is the single calibrator family the chain fits.
HEADLINE_POPULATION = "lrp_crossfit"
HEADLINE_FAMILY = "pooled"
#: Worlds the reference-seed bootstrap intervals of the transferred credit-ECE
#: and LaECE_0 average over (the intervals supp. Sec. D prints). They are the second
#: half of the ten worlds; the point estimates average over all ten.
BOOT_EVAL_WORLDS = (6, 7, 8, 9, 10)

# result name -> CSV basename
OFFICIAL_CSV = {
    "transfer": "01_transfer_check.csv",
    "noise": "02_noise_curve.csv",
    "claim_a": "04_claimA_floor.csv",
    "claim_b": "05_claimB_icc.csv",
    "multiseed": "09_multiseed_ci.csv",
    "target_shift": "10_target_shift.csv",
}


def _rows_df(rows):
    if isinstance(rows, dict):
        rows = [rows]
    return pd.DataFrame(list(rows))


def _cal0(scores, credit, folds, mask, tau):
    """World-0 (released) OOF isotonic credit confidence, fit on `mask`'s rows
    with their own folds. Returned on the masked rows only."""
    return L.oof_isotonic(scores[mask], credit[tau][0][mask].astype(np.float64),
                          folds[mask])


def compute(log=print):
    """Every result for the reference seed (the first, 43), then the multi-seed
    rows."""
    seeds = list(L.WORLD_SEEDS)
    log("[load] predictions + GT")
    pred = L.load_predictions()
    gt = L.load_gt()
    res = {}

    res["transfer"] = S.step1_transfer_check(gt, log=log)
    res["noise"] = S.step2_noise_curve(L.REFERENCE_SEED, log=log)

    primary = seeds[0]
    log(f"\n[worlds] reference seed {primary}: sampling worlds + targets")
    w = S.sample_worlds(gt, primary, log=log)
    Q, credit, MATCH_ID = S.compute_world_targets(pred, gt, w["worlds"], log=log)
    folds = L.crc32_fold(pred.sort_values("pred_row_id")["video-id"].to_numpy())
    pred_ord = pred.sort_values("pred_row_id").reset_index(drop=True)

    log("[floor] released-fit vs same-world refit")
    res["claim_a"] = A.claim_a_floor(pred_ord, gt, Q, credit, folds, log=log)
    log("[icc] hit-level ICC")
    res["claim_b"] = A.claim_b_icc(Q, MATCH_ID, log=log)
    res["target_shift"] = A.target_shift(Q, credit, L.population_masks(pred_ord, gt),
                                         taus=L.CREDIT_TAUS)

    primary_pack = {"Q": Q, "credit": credit, "MATCH_ID": MATCH_ID,
                    "folds": folds, "mean_swap_tiou": w["mean_swap_tiou"]}
    res["multiseed"] = multiseed_ci(pred_ord, gt, seeds, primary_pack, log=log)
    return res


def multiseed_ci(pred_ord, gt, seeds, primary_pack, log=print):
    """Rows of 09_multiseed_ci.csv: per seed, the same-instance ICC and the
    transferred credit-ECE@.5 and LaECE_0 on the LRP population; their mean and
    std over seeds; the reference-seed bootstrap intervals."""
    folds = primary_pack["folds"]
    scores = pred_ord["score"].to_numpy(np.float64)
    vids = pred_ord["video-id"].to_numpy()
    masks = L.population_masks(pred_ord, gt)
    m_head = masks[HEADLINE_POPULATION]

    rows = []
    icc_vals, floor_vals, sigma_vals, laece_vals = [], [], [], []
    for sd in seeds:
        if sd == seeds[0]:
            Q, credit = primary_pack["Q"], primary_pack["credit"]
            MID = primary_pack["MATCH_ID"]
            mean_swap = primary_pack["mean_swap_tiou"]
        else:
            w = S.sample_worlds(gt, sd, log=lambda *_: None)
            Q, credit, MID = S.compute_world_targets(pred_ord, gt, w["worlds"],
                                                     log=lambda *_: None)
            mean_swap = w["mean_swap_tiou"]
        b = A.claim_b_icc(Q, MID, log=lambda *_: None)
        same = next(r for r in b if r["stable_def"] == "same_instance")
        cr = credit[0.5]
        cal0_pop = _cal0(scores, credit, folds, m_head, 0.5)
        floor_pop = float(np.mean([L.credit_ece(cr[ww][m_head].astype(bool), cal0_pop)
                                   for ww in range(1, Q.shape[0])])) / 100.0
        # LaECE_0 floor, structurally identical to the credit-ECE floor above:
        # same seed's worlds, same folds, world-0 fit on the head population,
        # averaged over the same world range.
        calq_pop = L.oof_isotonic(scores[m_head], Q[0][m_head], folds[m_head])
        laece_pop = float(np.mean([L.laece0(calq_pop, Q[ww][m_head])
                                   for ww in range(1, Q.shape[0])]))
        icc_vals.append(same["icc_corrected"])
        floor_vals.append(floor_pop)
        sigma_vals.append(same["sigma_within"])
        laece_vals.append(laece_pop)
        rows.append({"seed": sd, "icc_corrected_same_instance": same["icc_corrected"],
                     "sigma_within": same["sigma_within"],
                     "stable_frac_same_instance": same["stable_frac"],
                     "head_credit_floor@0.5_paired_popfit": round(floor_pop, 4),
                     "head_laece0_floor_paired_popfit": round(laece_pop, 4),
                     "calibrator_family": HEADLINE_FAMILY,
                     "mean_swap_tiou": round(mean_swap, 4)})
        log(f"[multiseed] seed {sd}: ICC_corr(same)={same['icc_corrected']:.3f} "
            f"sigma_within={same['sigma_within']:.3f} "
            f"head floor={floor_pop:.4f} laece0={laece_pop:.4f}")

    def agg(vals):
        return (round(float(np.mean(vals)), 4),
                round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4))
    icc_m, icc_s = agg(icc_vals)
    fl_m, fl_s = agg(floor_vals)
    sg_m, sg_s = agg(sigma_vals)
    la_m, la_s = agg(laece_vals)
    rows.append({"seed": "AGG_seeds", "icc_corrected_same_instance": icc_m,
                 "icc_corrected_same_instance_std": icc_s, "sigma_within": sg_m,
                 "sigma_within_std": sg_s, "head_credit_floor@0.5_paired_popfit": fl_m,
                 "head_credit_floor@0.5_paired_popfit_std": fl_s,
                 "head_laece0_floor_paired_popfit": la_m,
                 "head_laece0_floor_paired_popfit_std": la_s,
                 "calibrator_family": HEADLINE_FAMILY, "n_seeds": len(seeds)})

    Q, credit, MID = (primary_pack["Q"], primary_pack["credit"],
                      primary_pack["MATCH_ID"])
    cr = credit[0.5]
    idx_head = np.flatnonzero(m_head)
    vids_head = vids[idx_head]
    cal0_pop = _cal0(scores, credit, folds, m_head, 0.5)
    cal0q_pop = L.oof_isotonic(scores[m_head], Q[0][m_head], folds[m_head])
    y_head = {ww: cr[ww][idx_head].astype(bool) for ww in BOOT_EVAL_WORLDS}
    q_head = {ww: Q[ww][idx_head] for ww in BOOT_EVAL_WORLDS}
    icc_bs, floor_bs, laece_bs = [], [], []
    # ONE loop over the replicate draws, so the credit and LaECE_0 floors are
    # bootstrapped on identical video resamples.
    for sub in L.video_bootstrap_indices(vids_head):
        floor_bs.append(np.mean([L.credit_ece(y_head[ww][sub], cal0_pop[sub])
                                 for ww in BOOT_EVAL_WORLDS]) / 100.0)
        laece_bs.append(np.mean([L.laece0(cal0q_pop[sub], q_head[ww][sub])
                                 for ww in BOOT_EVAL_WORLDS]))
    # the corrected ICC is computed by claim_b_icc on ALL ten worlds
    for idx in L.video_bootstrap_indices(vids):
        b = A.claim_b_icc(Q[:, idx], MID[:, idx], log=lambda *_: None)
        same = next(r for r in b if r["stable_def"] == "same_instance")
        icc_bs.append(same["icc_corrected"])
    all_worlds = f"1-{Q.shape[0] - 1}"
    boot_worlds = f"{BOOT_EVAL_WORLDS[0]}-{BOOT_EVAL_WORLDS[-1]}"
    for name, vals, worlds in (("icc_corrected_same_instance", icc_bs, all_worlds),
                               ("head_credit_floor@0.5_paired_popfit", floor_bs, boot_worlds),
                               ("head_laece0_floor_paired_popfit", laece_bs, boot_worlds)):
        lo, hi, neff = L.percentile_ci(vals)
        rows.append({"seed": "BOOT_ci", "stat": name,
                     "fit_population": HEADLINE_POPULATION if "head" in name else "full",
                     "metric": ("LaECE0" if "laece0" in name
                                else "credit_ECE" if "credit" in name else "ICC"),
                     "calibrator_family": HEADLINE_FAMILY,
                     "eval_worlds": worlds,
                     "boot_mean": round(float(np.mean(vals)), 4),
                     "ci_lo_2.5": round(lo, 4), "ci_hi_97.5": round(hi, 4),
                     "n_boot": neff})
        log(f"[bootstrap] {name}: {np.mean(vals):.4f} "
            f"[{lo:.4f}, {hi:.4f}] (B={neff}, worlds {worlds})")
    return rows


def results_to_csv_frames(res):
    """{csv_basename: DataFrame} for every result CSV present in res."""
    out = {}
    for name, csv in OFFICIAL_CSV.items():
        rows = res.get(name)
        if rows is None:
            continue
        out[csv] = _rows_df(rows)
    return out
