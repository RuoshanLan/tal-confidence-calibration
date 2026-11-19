"""Calibration analysis from the per-world targets.

Consumes Q (M+1, n_pred), credit{tau} and the raw scores, plus population
masks, and produces:
  claim_a_floor  credit-ECE@tau and LaECE_0 per population: raw / released-fit
                 / same-world refit  (Table 2)
  claim_b_icc    stable fraction, sigma_within, ICC
  target_shift   released-vs-alternative TP rate, label changes, mean quality

Calibrators are pooled score-only isotonic OOF under crc32 folds.
"""
import numpy as np

from . import lib as L


# ==========================================================================
# Released-fit vs same-world refit (Table 2)
# ==========================================================================
def claim_a_floor(pred, gt, Q, credit, folds, log=print, masks=None, taus=None):
    """Credit-ECE@tau and LaECE_0 per population (Table 2), each averaged over
    the alternative worlds w=1..M:

      raw_cross        : raw score as confidence                     (Raw)
      cal_pop0_floor   : calibrator fit on the released world (w=0) on this
                         population, evaluated in world w             (Rel.)
      same_world_refit : calibrator fit and evaluated within world w  (Same)

    Both calibrated columns use the same population and folds, so their
    difference is the transfer-refit gap.

    `masks` / `taus` default to this chain's ActivityNet populations and credit
    grid; the Ego4D chain passes its own."""
    n_worlds = Q.shape[0] - 1
    scores = pred["score"].to_numpy(np.float64)
    masks = L.population_masks(pred, gt) if masks is None else masks
    taus = L.CREDIT_TAUS if taus is None else tuple(taus)

    def _cred_ece_mean(conf, m, worlds, cr):
        return float(np.mean([L.credit_ece(cr[w][m].astype(bool), conf)
                              for w in worlds])) / 100.0

    rows = []
    # ---- credit-ECE @ each tau ----
    for tau in taus:
        cr = credit[tau]
        worlds = range(1, n_worlds + 1)
        for pop in L.POPULATIONS:
            m = masks[pop]
            raw_f = _cred_ece_mean(scores[m], m, worlds, cr)
            # paired: fit world-0 ON pop
            cal_pop0 = L.oof_isotonic(scores[m], cr[0][m], folds[m])
            floor_f = _cred_ece_mean(cal_pop0, m, worlds, cr)
            # same-world refit on pop
            refit_vals = []
            for w in worlds:
                refit = L.oof_isotonic(scores[m], cr[w][m], folds[m])
                refit_vals.append(L.credit_ece(cr[w][m].astype(bool), refit))
            refit_f = float(np.mean(refit_vals)) / 100.0
            rows.append({
                "claim": "A_credit_ece", "metric": "credit_ECE",
                "population": pop, "tau": tau, "n": int(m.sum()),
                "fit_population": pop, "calibrator_family": "pooled",
                "raw_cross": round(raw_f, 4),
                "cal_pop0_floor": round(floor_f, 4),
                "same_world_refit": round(refit_f, 4),
                "floor_over_refit": round(floor_f / max(refit_f, 1e-9), 1),
            })
            log(f"[floor] credit-ECE tau={tau} {pop}: raw={raw_f:.4f} "
                f"released-fit={floor_f:.4f} refit={refit_f:.4f}")

    # ---- LaECE_0 (pooled), quality target ----
    q0 = Q[0]
    for pop in L.POPULATIONS:
        m = masks[pop]
        raw_f = float(np.mean([L.laece0(scores[m], Q[w][m])
                               for w in range(1, n_worlds + 1)]))
        cal_pop0 = L.oof_isotonic(scores[m], q0[m], folds[m])
        floor_f = float(np.mean([L.laece0(cal_pop0, Q[w][m])
                                 for w in range(1, n_worlds + 1)]))
        refit_vals = []
        for w in range(1, n_worlds + 1):
            refit = L.oof_isotonic(scores[m], Q[w][m], folds[m])
            refit_vals.append(L.laece0(refit, Q[w][m]))
        refit_f = float(np.mean(refit_vals))
        rows.append({
            "claim": "A_laece0", "metric": "LaECE0",
            "population": pop, "tau": float("nan"), "n": int(m.sum()),
            "fit_population": pop, "calibrator_family": "pooled",
            "raw_cross": round(raw_f, 4),
            "cal_pop0_floor": round(floor_f, 4),
            "same_world_refit": round(refit_f, 4),
            "floor_over_refit": round(floor_f / max(refit_f, 1e-9), 2),
        })
        log(f"[floor] LaECE0 {pop}: raw={raw_f:.4f} "
            f"released-fit={floor_f:.4f} refit={refit_f:.4f}")
    return rows


# ==========================================================================
# Hit-level stability and ICC
# ==========================================================================
def _stable_masks(Q, MATCH_ID):
    """Two stability definitions on world-0 hits:
      protocol-stable   : q_w > 0 in every world (matched to *some* GT of the
                          class each world -- the evaluation protocol's view);
      same-instance     : additionally matched to the SAME global instance_id as
                          world 0 in every world (pure annotation jitter, no
                          matching reassignment).
    Returns (hit_idx, protocol_stable_idx, same_instance_idx, n_switch) where
    n_switch = protocol-stable hits whose matched instance changed in >=1 world."""
    q0 = Q[0]
    qw = Q[1:]
    hit_idx = np.flatnonzero(q0 > 0)
    matched_all = np.all(qw[:, hit_idx] > 0, axis=0)
    proto_idx = hit_idx[matched_all]
    # same matched instance across all worlds (0..M)
    mid0 = MATCH_ID[0, proto_idx]
    same_all = np.all(MATCH_ID[1:, proto_idx] == mid0, axis=0)
    same_idx = proto_idx[same_all]
    n_switch = int(len(proto_idx) - len(same_idx))
    return hit_idx, proto_idx, same_idx, n_switch


def _icc_row(qw, idx, label, n_hits, n_worlds):
    """ICC decomposition on a stable subset. Reports observed AND finite-M
    corrected ICC: the observed between-variance includes ~within/M
    of mean measurement noise, so between_latent = max(0, between - within/M)."""
    sub = qw[:, idx]                    # (M, n_sub)
    within = float(np.mean(np.var(sub, axis=0, ddof=1))) if idx.size else float("nan")
    means = np.mean(sub, axis=0)
    between = float(np.var(means, ddof=1)) if idx.size > 1 else float("nan")
    icc = between / (between + within) if (between + within) > 0 else float("nan")
    between_latent = max(0.0, between - within / n_worlds)
    icc_corr = (between_latent / (between_latent + within)
                if (between_latent + within) > 0 else float("nan"))
    return {"claim": "B_icc", "stable_def": label,
            "n_hits": int(n_hits), "n_stable": int(idx.size),
            "stable_frac": round(idx.size / max(n_hits, 1), 4),
            "flip_rate": round(1.0 - idx.size / max(n_hits, 1), 4),
            "sigma_within": round(float(np.sqrt(within)), 4),
            "var_within": round(within, 6),
            "var_between_observed": round(between, 6),
            "var_between_latent": round(between_latent, 6),
            "icc_observed": round(icc, 4),
            "icc_corrected": round(icc_corr, 4)}


def claim_b_icc(Q, MATCH_ID, log=print):
    """Hit-level ICC decomposition under both stability definitions, with
    observed and finite-M-corrected ICC. Returns two rows (protocol-stable,
    same-instance); the protocol row carries the switch count."""
    n_worlds = Q.shape[0] - 1
    qw = Q[1:]
    hit_idx, proto_idx, same_idx, n_switch = _stable_masks(Q, MATCH_ID)
    rows = []
    for label, idx in (("protocol", proto_idx), ("same_instance", same_idx)):
        row = _icc_row(qw, idx, label, len(hit_idx), n_worlds)
        row["n_identity_switch"] = n_switch if label == "protocol" else 0
        rows.append(row)
        log(f"[icc:{label}] hits={len(hit_idx)} stable={idx.size} "
            f"({idx.size/max(len(hit_idx),1)*100:.1f}%) "
            f"sigma_within={row['sigma_within']:.3f} "
            f"ICC obs={row['icc_observed']:.3f} corr={row['icc_corrected']:.3f}")
    log(f"[icc] identity switches among protocol-stable hits: {n_switch} "
        f"({n_switch/max(len(proto_idx),1)*100:.2f}%)")
    return rows


# ==========================================================================
# Target shift: released world vs alternative worlds (both chains)
# ==========================================================================
def target_shift(Q, credit, masks, taus):
    """Released world (row 0) vs the alternative worlds, per population: for
    the TP/FP target at each tau, the TP rates and the fractions of labels that
    change (fractions of prediction-world pairs or of predictions); for the
    quality target, the mean q."""
    rows = []
    for pop, m in masks.items():
        for t in taus:
            cr = credit[t][:, m].astype(bool)
            rel, alt = cr[0], cr[1:]
            lost = rel[None, :] & ~alt
            gained = ~rel[None, :] & alt
            lost_any, gained_any = lost.any(axis=0), gained.any(axis=0)
            rows.append({
                "target": "credit", "population": pop, "tau": t, "n": int(m.sum()),
                "rel_rate": float(rel.mean()), "alt_rate": float(alt.mean()),
                "rate_drop": float(rel.mean() - alt.mean()),
                "differ_frac": float((alt != rel[None, :]).mean()),
                "moved_frac": float((alt != rel[None, :]).any(axis=0).mean()),
                "loss_frac": float(lost.mean()), "gain_frac": float(gained.mean()),
                "only_loss_frac": float((lost_any & ~gained_any).mean()),
                "only_gain_frac": float((gained_any & ~lost_any).mean()),
                "both_frac": float((lost_any & gained_any).mean())})
        q = Q[:, m].astype(np.float64)
        rows.append({"target": "quality", "population": pop, "tau": float("nan"),
                     "n": int(m.sum()), "rel_rate": float(q[0].mean()),
                     "alt_rate": float(q[1:].mean()),
                     "rate_drop": float(q[0].mean() - q[1:].mean())})
    return rows
