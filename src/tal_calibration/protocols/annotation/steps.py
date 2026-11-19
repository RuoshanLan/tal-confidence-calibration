"""Compute steps for the ActivityNet-1.3 annotation audit.

Each function takes the loaded predictions/GT and returns rows/arrays;
pipeline.compute sequences them and the entry point writes the files.

    Step 1  test->val transfer check  -> 01_transfer_check.csv
    Step 2  noise model curve         -> 02_noise_curve.csv
    worlds  annotation worlds (seed 43 + multi-seed) and per-world targets
"""
import numpy as np

from . import lib as L


# ==========================================================================
# Step 1 -- test->val transfer check
# ==========================================================================
def step1_transfer_check(gt, log=print):
    """val_len[b] = fraction of val instances in length bucket b; implied val
    agreement marginal = val_len @ P(agr|len), compared bucket by bucket with
    DETAD's test marginal."""
    lb = L.bucket_index(gt["len"].to_numpy(), L.LEN_EDGES)
    val_len = np.bincount(lb, minlength=5) / len(gt) * 100.0
    pcond = L.conditional_agr_given_len()
    implied_agr = (val_len / 100.0) @ pcond * 100.0
    gap = np.abs(implied_agr - L.DETAD_AGR_MARGINAL)
    rows = []
    agr_names = ["XW", "W", "M", "H", "XH"]
    for i, nm in enumerate(agr_names):
        rows.append({"bucket": nm,
                     "implied_val_agr_pct": round(float(implied_agr[i]), 1),
                     "test_agr_pct": float(L.DETAD_AGR_MARGINAL[i]),
                     "gap_pp": round(float(gap[i]), 2)})
    log(f"[step1] implied val agreement {implied_agr.round(1).tolist()} "
        f"vs test {L.DETAD_AGR_MARGINAL.tolist()}; max gap {gap.max():.2f}pp")
    return rows


# ==========================================================================
# Step 2 -- noise model curve
# ==========================================================================
def step2_noise_curve(seed, log=print):
    """Rows of 02_noise_curve.csv: the simulated agreement at each sigma_rel of
    the grid, for the curve built with `seed`."""
    sg, ag, a_min = L.build_noise_curve(seed)
    rows = [{"sigma_rel": round(float(s), 4), "mean_median_pairwise_tiou": round(float(a), 4)}
            for s, a in zip(sg, ag)]
    log(f"[step2] seed={seed} A_MIN={a_min:.4f}")
    return rows


# ==========================================================================
# Annotation worlds
# ==========================================================================
def sample_worlds(gt, seed, n_worlds=L.N_WORLDS, log=print):
    """M independent alternative annotations per instance. Returns worlds
    (n_worlds, n_inst, 2) = [start,end] of each alternative, and the mean
    swap-vs-original tIoU."""
    rng = np.random.default_rng(seed)
    curve = L.build_noise_curve(seed)
    sigma_abs, clamped = L.sample_instance_sigma(gt, curve, rng)
    n = len(gt)
    worlds = np.empty((n_worlds, n, 2), dtype=np.float64)
    s0 = gt["t-start"].to_numpy()
    e0 = gt["t-end"].to_numpy()
    swap_tious = []
    for w in range(n_worlds):
        cs, ce = L.sample_single_swap(gt, sigma_abs, rng)
        worlds[w, :, 0] = cs
        worlds[w, :, 1] = ce
        inter = np.minimum(e0, ce) - np.maximum(s0, cs)
        inter = np.clip(inter, 0.0, None)
        union = np.maximum(e0, ce) - np.minimum(s0, cs)
        swap_tious.append(np.where(union > 0, inter / union, 0.0))
    swap_tious = np.stack(swap_tious)  # (n_worlds, n)
    mean_swap = float(swap_tious.mean())
    log(f"[worlds] seed={seed} sampled {n_worlds} worlds; "
        f"mean swap-vs-original tIoU={mean_swap:.3f}; clamped={clamped.mean()*100:.1f}%")
    return {"worlds": worlds, "mean_swap_tiou": mean_swap}


def compute_world_targets(pred, gt, worlds, log=print):
    """Per-world quality target q^{o2o}, credit y(tau), and matched instance id
    for every prediction. World 0 uses the original GT geometry;
    world w (1..M) replaces every instance's segment by its w-th alternative.
    Returns:
      Q        (M+1, n_pred) float32   quality target per world
      credit   {tau: (M+1, n_pred) int8}
      MATCH_ID (M+1, n_pred) int64      global GT instance_id per world (-1 = unmatched)
    all in pred_row_id (pkl) order. MATCH_ID lets the ICC readout separate the
    same instance moving from a reassignment."""
    n_worlds = worlds.shape[0]
    pred = pred.sort_values("pred_row_id").reset_index(drop=True)
    n_pred = len(pred)
    Q = np.zeros((n_worlds + 1, n_pred), dtype=np.float32)
    MATCH_ID = np.full((n_worlds + 1, n_pred), -1, dtype=np.int64)
    credit = {t: np.zeros((n_worlds + 1, n_pred), dtype=np.int8) for t in L.CREDIT_TAUS}

    gt_video = gt["video-id"].to_numpy()
    gt_label = gt["label"].to_numpy()
    gt_instance = gt["instance_id"].to_numpy()

    # the grouping is world-independent: build it once
    plan = L.build_match_plan(pred, gt_video, gt_label)

    for w in range(n_worlds + 1):
        if w == 0:
            gs = gt["t-start"].to_numpy(np.float64)
            ge = gt["t-end"].to_numpy(np.float64)
        else:
            gs = worlds[w - 1, :, 0]
            ge = worlds[w - 1, :, 1]
        q, mid = L.compute_quality_targets(
            pred, gs, ge, gt_video, gt_label, gt_instance,
            plan=plan, return_match_id=True)
        Q[w] = q
        MATCH_ID[w] = mid
        ctg = L.compute_credit_targets(pred, gs, ge, gt_video, gt_label,
                                       L.CREDIT_TAUS, plan=plan)
        for t in L.CREDIT_TAUS:
            credit[t][w] = ctg[t]
        if w % 2 == 0 or w == n_worlds:
            log(f"[worlds]   world {w}/{n_worlds} targets: hit rate={np.mean(Q[w] > 0):.4f}")
    return Q, credit, MATCH_ID
