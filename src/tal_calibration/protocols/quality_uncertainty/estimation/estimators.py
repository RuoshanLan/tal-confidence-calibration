"""Point and quantile estimators (histogram gradient boosting).

The hyperparameters are those of supp. Table S7. The class-duration priors
take `train_video_ids` explicitly: there is no full-GT path.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..data.observable_features import prior_feature_arrays

GBR_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=40, random_state=0)
QGBR_PARAMS = dict(max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                   l2_regularization=1.0, early_stopping=True,
                   validation_fraction=0.2, n_iter_no_change=15)


def fold_clean_prior_features(df, gt, train_video_ids):
    """Class-duration priors from the explicit train-video GT subset only."""
    gtr = gt[gt["video-id"].isin(set(train_video_ids))]
    return prior_feature_arrays(df, gtr)


def pinball(y, f, q):
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _gbr_pair(X_train, y_train, X_eval, seed, alpha):
    preds = []
    for q in (alpha / 2, 1 - alpha / 2):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=seed, **QGBR_PARAMS)
        model.fit(X_train, y_train)
        preds.append(model.predict(X_eval))
    lo, hi = preds
    # quantile-crossing repair, per row
    return np.minimum(lo, hi), np.maximum(lo, hi)


def fit_quantile_pair(X_train, y_train, X_eval, seed, groups_train, alpha):
    """(lo, hi) quantile heads at alpha/2 and 1 - alpha/2, with fallback to
    the constant TRAIN quantiles when the heads do not beat them in pinball
    loss on a held-out fifth of the TRAIN videos (ties go to the constant)."""
    const = (np.full(len(X_eval), np.quantile(y_train, alpha / 2)),
             np.full(len(X_eval), np.quantile(y_train, 1 - alpha / 2)))
    if X_train.shape[1] == 0:
        return const
    uniq = np.unique(groups_train)
    uniq = uniq[np.random.default_rng(seed + 7).permutation(len(uniq))]
    val_groups = set(uniq[: max(1, len(uniq) // 5)])
    val_i = np.isin(groups_train, list(val_groups))
    fit_i = ~val_i
    lo_v, hi_v = _gbr_pair(X_train[fit_i], y_train[fit_i], X_train[val_i], seed, alpha)
    yv = y_train[val_i]
    loss_gbr = pinball(yv, lo_v, alpha / 2) + pinball(yv, hi_v, 1 - alpha / 2)
    loss_const = (pinball(yv, np.quantile(y_train[fit_i], alpha / 2), alpha / 2)
                  + pinball(yv, np.quantile(y_train[fit_i], 1 - alpha / 2), 1 - alpha / 2))
    if loss_const <= loss_gbr:
        return const
    return _gbr_pair(X_train, y_train, X_eval, seed, alpha)


# ---- one-sided quantile fitter for the floor ----
# The floor spends the FULL alpha on the lower tail: the fitted level is
# Q_alpha (.20), not the pair's Q_{alpha/2}.

def _gbr_single(X_train, y_train, X_eval, seed, q):
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, random_state=seed, **QGBR_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def fit_quantile_single(X_train, y_train, X_eval, seed, groups_train, alpha):
    """Conditional lower quantile Q_alpha with the same fallback rule as
    `fit_quantile_pair` (video-grouped validation split, rng(seed+7), 1/5 of
    the TRAIN videos)."""
    const = np.full(len(X_eval), np.quantile(y_train, alpha))
    if X_train.shape[1] == 0:
        return const
    uniq = np.unique(groups_train)
    uniq = uniq[np.random.default_rng(seed + 7).permutation(len(uniq))]
    val_groups = set(uniq[: max(1, len(uniq) // 5)])
    val_i = np.isin(groups_train, list(val_groups))
    fit_i = ~val_i
    lo_v = _gbr_single(X_train[fit_i], y_train[fit_i], X_train[val_i], seed, alpha)
    yv = y_train[val_i]
    loss_gbr = pinball(yv, lo_v, alpha)
    loss_const = pinball(yv, np.quantile(y_train[fit_i], alpha), alpha)
    if loss_const <= loss_gbr:
        return const
    return _gbr_single(X_train, y_train, X_eval, seed, alpha)
