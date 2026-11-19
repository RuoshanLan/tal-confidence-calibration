"""Pre-NMS proposal-cloud features and their in-memory build.

For every post-NMS row, statistics of the pre-NMS candidates that overlap it
(supp. Table S6, the pre-NMS proposal groups): same-class and class-agnostic
duration, boundary and score summaries. The caps MAX_CLOUD / MAX_CLOUD_CA /
TOP_CORE are part of the feature definition.

They are computed over ALL post-NMS rows (a hits-only build would put NaN
exactly on the unmatched rows and hand the arms the matched status), and are
GT-neutral: each row's features read only its own coordinates and the
pre-NMS pool.
"""
import numpy as np
from pathlib import Path
from ....datasets import load_datasets
from ....prediction_io import load_predictions_from_path

MAX_CLOUD = 3000        # cap pre-NMS candidates per (video, label) group by score
MAX_CLOUD_CA = 6000     # cap class-agnostic per-video pool by score
TOP_CORE = 20           # size of the high-score core subset

CLOUD_FEATS = [
    "cl_n", "cl_mass", "cl_share",
    "cl_wm_logdur", "cl_ws_logdur",
    "cl_q25_logdur", "cl_q50_logdur", "cl_q75_logdur", "cl_top_logdur",
    "cl_start_std", "cl_end_std", "cl_left_ext", "cl_right_ext",
    "cl_span", "cl_plugin_sw", "cl_n03", "cl_wm_logdur03",
]
CLOUD_X_FEATS = [
    # containment votes: same-class candidates that contain the prediction —
    # for a fragment of a long GT these carry the full-extent hypothesis
    "cn_n", "cn_mass_frac", "cn_wm_logdur",
    # high-score core of the same-class cloud (dilution control)
    "tp_wm_logdur", "tp_n03",
    # class-agnostic per-video pool: boundary evidence is not class-specific
    "ca_n", "ca_wm_logdur", "ca_q50_logdur", "ca_start_std", "ca_end_std",
    "ca_left_ext", "ca_right_ext", "ca_span",
]


def add_cloud_features(queries, pre_df):
    """Pre-NMS proposal-cloud features for each query row: the same-(video,
    label) pre-NMS candidates overlapping it (self excluded by exact
    coordinates) and the class-agnostic per-video pool. Locations inside a GT
    action vote for its full extent, so these duration / boundary statistics
    can carry L_g even when the surviving prediction is truncated.
    """
    hits = queries.reset_index(drop=True)
    out = {f: np.full(len(hits), np.nan) for f in CLOUD_FEATS + CLOUD_X_FEATS}

    pre_s = pre_df["t-start"].to_numpy(dtype=np.float64)
    pre_e = pre_df["t-end"].to_numpy(dtype=np.float64)
    pre_w = pre_df["score"].to_numpy(dtype=np.float64)
    groups = pre_df.groupby(["video-id", "label"], sort=False).indices
    groups_v = pre_df.groupby("video-id", sort=False).indices

    def overlap_mask(s, e, L, cs, ce, cl):
        inter = np.minimum(e[:, None], ce[None, :]) - np.maximum(s[:, None], cs[None, :])
        inter = np.clip(inter, 0.0, None)
        union = L[:, None] + cl[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            tiou = np.where(union > 0, inter / union, 0.0)
        is_self = (np.abs(cs[None, :] - s[:, None]) < 1e-6) & (np.abs(ce[None, :] - e[:, None]) < 1e-6)
        return (inter > 0) & ~is_self, tiou

    # ---- same-class cloud (+ containment, high-score core) ----
    for (vid, label), g in hits.groupby(["video-id", "label"], sort=False):
        gi = groups.get((vid, label))
        if gi is None:
            continue
        if len(gi) > MAX_CLOUD:
            gi = gi[np.argsort(pre_w[gi])[::-1][:MAX_CLOUD]]
        cs, ce, cw = pre_s[gi], pre_e[gi], pre_w[gi]
        cl = ce - cs
        s = g["t-start"].to_numpy(); e = g["t-end"].to_numpy()
        L = e - s
        idx = g.index.to_numpy()
        scores = g["score"].to_numpy()
        ov, tiou = overlap_mask(s, e, L, cs, ce, cl)

        for i in range(len(g)):
            m = ov[i]
            out["cl_n"][idx[i]] = m.sum()
            if not m.any():
                continue
            wj = cw[m]; lj = cl[m]; sw = wj.sum()
            logdr = np.log(lj / L[i])
            out["cl_mass"][idx[i]] = sw
            out["cl_share"][idx[i]] = scores[i] / max(wj.max(), 1e-12)
            wm = (wj * logdr).sum() / sw
            out["cl_wm_logdur"][idx[i]] = wm
            out["cl_ws_logdur"][idx[i]] = np.sqrt(max((wj * logdr**2).sum() / sw - wm**2, 0.0))
            q25, q50, q75 = np.percentile(logdr, [25, 50, 75])
            out["cl_q25_logdur"][idx[i]] = q25
            out["cl_q50_logdur"][idx[i]] = q50
            out["cl_q75_logdur"][idx[i]] = q75
            out["cl_top_logdur"][idx[i]] = logdr[np.argmax(wj)]
            ds = (cs[m] - s[i]) / L[i]; de = (ce[m] - e[i]) / L[i]
            for name, dv in (("cl_start_std", ds), ("cl_end_std", de)):
                mu = (wj * dv).sum() / sw
                out[name][idx[i]] = np.sqrt(max((wj * dv**2).sum() / sw - mu**2, 0.0))
            out["cl_left_ext"][idx[i]] = (wj * np.clip(s[i] - cs[m], 0, None)).sum() / sw / L[i]
            out["cl_right_ext"][idx[i]] = (wj * np.clip(ce[m] - e[i], 0, None)).sum() / sw / L[i]
            out["cl_span"][idx[i]] = (max(ce[m].max(), e[i]) - min(cs[m].min(), s[i])) / L[i]
            out["cl_plugin_sw"][idx[i]] = (wj * np.minimum(L[i], lj) / np.maximum(L[i], lj)).sum() / sw
            m3 = m & (tiou[i] > 0.3)
            out["cl_n03"][idx[i]] = m3.sum()
            if m3.any():
                wj3 = cw[m3]
                out["cl_wm_logdur03"][idx[i]] = (wj3 * np.log(cl[m3] / L[i])).sum() / wj3.sum()

            contain = m & (cs <= s[i] + 0.02 * L[i]) & (ce >= e[i] - 0.02 * L[i])
            out["cn_n"][idx[i]] = contain.sum()
            out["cn_mass_frac"][idx[i]] = cw[contain].sum() / sw
            if contain.any():
                wc = cw[contain]
                out["cn_wm_logdur"][idx[i]] = (wc * np.log(cl[contain] / L[i])).sum() / wc.sum()
            core = np.where(m)[0][np.argsort(wj)[::-1][:TOP_CORE]]
            wt = cw[core]
            out["tp_wm_logdur"][idx[i]] = (wt * np.log(cl[core] / L[i])).sum() / wt.sum()
            out["tp_n03"][idx[i]] = (tiou[i][core] > 0.3).sum()

    # ---- class-agnostic per-video cloud ----
    for vid, g in hits.groupby("video-id", sort=False):
        gi = groups_v.get(vid)
        if gi is None:
            continue
        if len(gi) > MAX_CLOUD_CA:
            gi = gi[np.argsort(pre_w[gi])[::-1][:MAX_CLOUD_CA]]
        cs, ce, cw = pre_s[gi], pre_e[gi], pre_w[gi]
        cl = ce - cs
        s = g["t-start"].to_numpy(); e = g["t-end"].to_numpy()
        L = e - s
        idx = g.index.to_numpy()
        ov, _ = overlap_mask(s, e, L, cs, ce, cl)

        for i in range(len(g)):
            m = ov[i]
            out["ca_n"][idx[i]] = m.sum()
            if not m.any():
                continue
            wj = cw[m]; lj = cl[m]; sw = wj.sum()
            logdr = np.log(lj / L[i])
            out["ca_wm_logdur"][idx[i]] = (wj * logdr).sum() / sw
            out["ca_q50_logdur"][idx[i]] = np.percentile(logdr, 50)
            ds = (cs[m] - s[i]) / L[i]; de = (ce[m] - e[i]) / L[i]
            for name, dv in (("ca_start_std", ds), ("ca_end_std", de)):
                mu = (wj * dv).sum() / sw
                out[name][idx[i]] = np.sqrt(max((wj * dv**2).sum() / sw - mu**2, 0.0))
            out["ca_left_ext"][idx[i]] = (wj * np.clip(s[i] - cs[m], 0, None)).sum() / sw / L[i]
            out["ca_right_ext"][idx[i]] = (wj * np.clip(ce[m] - e[i], 0, None)).sum() / sw / L[i]
            out["ca_span"][idx[i]] = (max(ce[m].max(), e[i]) - min(cs[m].min(), s[i])) / L[i]
    for f, v in out.items():
        hits[f] = v
    return hits


CLOUD_COLUMNS = CLOUD_FEATS + CLOUD_X_FEATS

#: dataset -> frame; process-local memo, one build per dataset
_CLOUD = {}


class CloudBuildError(RuntimeError):
    pass


def clear_memo():
    """Drop the in-process cloud memo (tests; long-lived processes)."""
    _CLOUD.clear()


def _pre_nms_path(dataset):
    cfg = load_datasets([dataset])[dataset]
    path = cfg.get("pre_nms")
    if path is None:
        raise CloudBuildError(
            f"[{dataset}] no pre_nms path declared in datasets.yaml; the "
            "registered full-observable interface needs the pre-NMS pool")
    path = Path(path)
    if not path.exists():
        raise CloudBuildError(f"[{dataset}] declared pre_nms missing: {path}")
    return path


def build(dataset, log=print):
    """The cloud features of every post-NMS row of `dataset`, computed in
    memory: a frame of ``prediction_id`` + CLOUD_COLUMNS. Memoised per dataset.
    """
    from . import prepared_data

    hit = _CLOUD.get(dataset)
    if hit is not None:
        return hit

    base = prepared_data.load(dataset, parts=("base",))["base"]
    pre_path = _pre_nms_path(dataset)
    log(f"[{dataset}] building the all-rows cloud part for "
        f"{len(base)} post-NMS rows from {pre_path.name}...")
    pre_df = load_predictions_from_path(pre_path, None, "pickle")
    featured = add_cloud_features(base.reset_index(drop=True), pre_df)
    frame = featured[["prediction_id"] + list(CLOUD_COLUMNS)]
    if len(frame) != len(base):
        raise CloudBuildError(
            f"[{dataset}] cloud build changed the row count "
            f"{len(base)} -> {len(frame)}")
    log(f"[{dataset}] cloud part ready: {len(frame)} rows")
    _CLOUD[dataset] = frame
    return frame
