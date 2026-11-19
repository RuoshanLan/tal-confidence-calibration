"""Single-pass video-cluster bootstrap for many statistics at once.

TEST videos are resampled with replacement and rows weighted by how often
their video was drawn; one pass serves every arm, so every paired contrast is
paired on identical replicates. ``point_r2_bootstrap`` recomputes

    R^2_arm = 1 - E_w[(y - pred)^2] / (E_w[y^2] - E_w[y]^2)

per replicate from the same weight draw (R^2 is not a weighted mean).
"""
import numpy as np
import pandas as pd

CI_PERCENTILES = (2.5, 97.5)


def _weight_draws(vids_test, rng, n_boot):
    vids = pd.Series(np.asarray(vids_test)).reset_index(drop=True)
    uniq = np.unique(vids.to_numpy())
    for _ in range(int(n_boot)):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        counts = pd.Series(take).value_counts()
        w = vids.map(counts).fillna(0).to_numpy(dtype=np.float64)
        total = w.sum()
        if total == 0:
            continue
        yield w, total


def weighted_mean_ci(vids_test, M, rng, n_boot, pairs=()):
    """Video-bootstrap CIs for the weighted mean of every column of `M`.

    M      : (n_rows, k) float matrix of per-row quantities
    pairs  : sequence of (i, j) column indices -> CI of mean(col_i) - mean(col_j)

    Returns (ci_lo, ci_hi, pair_lo, pair_hi) as float arrays of length k and
    len(pairs). Both arms of a pair see the same resampled videos in every
    replicate.
    """
    M = np.asarray(M, dtype=np.float64)
    _, k = M.shape
    draws = [(M.T @ w) / total
             for w, total in _weight_draws(vids_test, rng, n_boot)]
    if not draws:
        nan = np.full(k, np.nan)
        return (nan, nan.copy(), np.full(len(pairs), np.nan),
                np.full(len(pairs), np.nan))
    D = np.vstack(draws)                                  # (n_boot, k)
    ci_lo, ci_hi = np.percentile(D, CI_PERCENTILES, axis=0)
    if pairs:
        idx_a = [p[0] for p in pairs]
        idx_b = [p[1] for p in pairs]
        P = D[:, idx_a] - D[:, idx_b]
        pair_lo, pair_hi = np.percentile(P, CI_PERCENTILES, axis=0)
    else:
        pair_lo = pair_hi = np.empty(0)
    return ci_lo, ci_hi, pair_lo, pair_hi


def point_r2_bootstrap(vids_test, y, preds, rng, n_boot, pairs=()):
    """Video-bootstrap CIs for per-arm TEST R^2 and paired R^2 differences.

    preds  : {arm_name: per-row prediction array}
    pairs  : sequence of (arm_a, arm_b) -> CI of R^2_a - R^2_b

    Returns ({arm: (lo, hi)}, {(a, b): (lo, hi)}).
    """
    y = np.asarray(y, dtype=np.float64)
    names = list(preds)
    se = np.column_stack([(y - np.asarray(preds[n], dtype=np.float64)) ** 2
                          for n in names])
    aux = np.column_stack([y, y ** 2])
    draws = []
    for w, total in _weight_draws(vids_test, rng, n_boot):
        mse = (se.T @ w) / total
        m1, m2 = (aux.T @ w) / total
        var = m2 - m1 ** 2
        if var <= 0:
            draws.append(np.full(len(names), np.nan))
        else:
            draws.append(1.0 - mse / var)
    if not draws:
        nan2 = (np.nan, np.nan)
        return {n: nan2 for n in names}, {p: nan2 for p in pairs}
    D = np.vstack(draws)
    lo, hi = np.nanpercentile(D, CI_PERCENTILES, axis=0)
    arm_ci = {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)}
    pair_ci = {}
    for a, b in pairs:
        diff = D[:, names.index(a)] - D[:, names.index(b)]
        plo, phi = np.nanpercentile(diff, CI_PERCENTILES)
        pair_ci[(a, b)] = (float(plo), float(phi))
    return arm_ci, pair_ci
