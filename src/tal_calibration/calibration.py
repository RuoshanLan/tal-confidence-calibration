"""Calibration primitives: the binned ECE and the four netcal calibrators
(histogram binning, Platt, beta, temperature; ``netcal==1.4.0``)."""
from __future__ import annotations

import numpy as np

__all__ = [
    "ece",
    "make_calibrator",
    "CALIBRATOR_FACTORIES",
]


def ece(y_true, y_prob, norm="l1", n_bins=10, strategy="uniform"):
    """Expected calibration error over ``n_bins`` left-closed bins, equal-width
    (``uniform``) or equal-mass (``quantile``)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0

    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]

    n_bins = int(n_bins)
    if strategy == "quantile":
        quantiles = np.percentile(y_prob, np.arange(0, 1, 1.0 / n_bins) * 100)
    elif strategy == "uniform":
        quantiles = np.arange(0, 1, 1.0 / n_bins)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    threshold_indices = np.searchsorted(y_prob, quantiles).tolist()
    threshold_indices.append(y_true.shape[0])
    avg_pred_true = np.zeros(n_bins, dtype=np.float64)
    bin_centroid = np.zeros(n_bins, dtype=np.float64)
    delta_count = np.zeros(n_bins, dtype=np.float64)

    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]
        if i_end <= i_start:
            continue
        delta_count[i] = float(i_end - i_start)
        avg_pred_true[i] = np.sum(y_true[i_start:i_end]) / delta_count[i]
        bin_centroid[i] = np.sum(y_prob[i_start:i_end]) / delta_count[i]

    if norm == "max":
        return float(np.max(np.abs(avg_pred_true - bin_centroid)))
    if norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        return float(np.sum(delta_loss) / float(y_true.shape[0]))
    raise ValueError(f"Unknown norm: {norm}")


def _hist(bins: int = 20):
    from netcal.binning import HistogramBinning

    return HistogramBinning(bins=bins)


def _beta():
    from netcal.scaling import BetaCalibration

    return BetaCalibration()


def _logistic():
    from netcal.scaling import LogisticCalibration

    return LogisticCalibration()


def _temperature():
    from netcal.scaling import TemperatureScaling

    return TemperatureScaling()


CALIBRATOR_FACTORIES = {
    "hist": _hist,
    "platt": _logistic,
    "beta": _beta,
    "temp": _temperature,
}


def make_calibrator(arm: str, **kwargs):
    """A fresh netcal calibrator for ``arm`` in {"hist", "platt", "beta", "temp"};
    ``kwargs`` (``bins=``) apply to the histogram only."""
    if arm not in CALIBRATOR_FACTORIES:
        raise ValueError(
            f"unknown calibrator arm {arm!r}; expected one of "
            f"{sorted(CALIBRATOR_FACTORIES)}"
        )
    if arm == "hist":
        return CALIBRATOR_FACTORIES[arm](**kwargs)
    if kwargs:
        raise TypeError(f"{arm} calibrator takes no kwargs; got {kwargs}")
    return CALIBRATOR_FACTORIES[arm]()
