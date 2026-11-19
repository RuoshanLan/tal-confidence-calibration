"""Per-prediction observable features: the prediction's own quantities and its
same-class post-NMS neighbourhood (supp. Table S6).

`prior_feature_arrays(df, gt)` computes the class-duration priors from the GT
table it is given; callers pass the TRAIN-video GT only.
"""
import numpy as np

from ....matching import minmax_ratio

MAX_GROUP = 600          # cap neighbor context to top-K by score in huge groups
SELF_FEATS = ["score", "log_len", "prior_ratio", "prior_cdf"]
PRIOR_FEATS = ["prior_ratio", "prior_cdf"]
NBR_FEATS = [
    "n_ov", "plugin_sw", "plugin03", "union_ratio", "comp_span_ratio",
    "comp_n", "overhang", "jitter", "len_gt_frac", "wlog_len_diff", "score_share",
]


def prior_feature_arrays(df, gt):
    """(prior_ratio, prior_cdf) for df rows using ONLY the given GT table."""
    gt_len = (gt["t-end"] - gt["t-start"]).to_numpy(dtype=np.float64)
    labels_gt = gt["label"].to_numpy()
    L = (df["t-end"] - df["t-start"]).to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy()
    ratio = np.full(len(df), np.nan)
    cdf = np.full(len(df), np.nan)
    for cls in np.unique(labels_gt):
        lens = gt_len[labels_gt == cls]
        m = labels == cls
        if not m.any() or len(lens) == 0:
            continue
        med = float(np.median(lens))
        r = L[m] / med
        ratio[m] = np.minimum(r, 1.0 / r)
        cdf[m] = minmax_ratio(L[m][:, None], lens[None, :]).mean(1)
    return ratio, cdf


def add_neighbor_features(df):
    """Same-class overlap-neighbourhood features (NBR_FEATS)."""
    n = len(df)
    out = {f: np.full(n, np.nan) for f in NBR_FEATS}
    df = df.reset_index(drop=True)
    for (_, _), g in df.groupby(["video-id", "label"], sort=False):
        idx = g.index.to_numpy()
        if len(g) > MAX_GROUP:
            keep = g["score"].to_numpy().argsort()[::-1][:MAX_GROUP]
            ctx = g.iloc[keep]
        else:
            ctx = g
        cs = ctx["t-start"].to_numpy(); ce = ctx["t-end"].to_numpy()
        cw = ctx["score"].to_numpy(); cl = ce - cs
        cidx = ctx.index.to_numpy()

        s = g["t-start"].to_numpy(); e = g["t-end"].to_numpy()
        w = g["score"].to_numpy(); L = e - s

        inter = np.minimum(e[:, None], ce[None, :]) - np.maximum(s[:, None], cs[None, :])
        inter = np.clip(inter, 0.0, None)
        union = L[:, None] + cl[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            tiou = np.where(union > 0, inter / union, 0.0)
        self_mask = idx[:, None] == cidx[None, :]
        ov = (inter > 0) & ~self_mask
        ov03 = (tiou > 0.3) & ~self_mask

        # transitive overlap components (cheap sweep on intervals)
        order = np.argsort(s)
        comp_id = np.empty(len(g), dtype=np.int64)
        cur_end, cid = -np.inf, -1
        comp_bounds = []
        for oi in order:
            if s[oi] > cur_end:
                cid += 1
                comp_bounds.append([s[oi], e[oi]])
                cur_end = e[oi]
            else:
                cur_end = max(cur_end, e[oi])
                comp_bounds[-1][1] = cur_end
            comp_id[oi] = cid
        comp_bounds = np.asarray(comp_bounds, dtype=np.float64)
        comp_span = comp_bounds[:, 1] - comp_bounds[:, 0]
        comp_sizes = np.bincount(comp_id)
        gmax = np.zeros(cid + 1)
        np.maximum.at(gmax, comp_id, w)

        for i in range(len(g)):
            m = ov[i]
            out["n_ov"][idx[i]] = m.sum()
            out["comp_span_ratio"][idx[i]] = comp_span[comp_id[i]] / L[i]
            out["comp_n"][idx[i]] = comp_sizes[comp_id[i]]
            out["score_share"][idx[i]] = w[i] / gmax[comp_id[i]]
            if m.any():
                wj = cw[m]; lj = cl[m]; sw = wj.sum()
                out["plugin_sw"][idx[i]] = (wj * minmax_ratio(L[i], lj)).sum() / sw
                segs = np.stack([np.append(cs[m], s[i]), np.append(ce[m], e[i])], 1)
                segs = segs[np.argsort(segs[:, 0])]
                tot, ce_run = 0.0, -np.inf
                for a, b in segs:
                    if a > ce_run:
                        tot += b - a; ce_run = b
                    elif b > ce_run:
                        tot += b - ce_run; ce_run = b
                out["union_ratio"][idx[i]] = tot / L[i]
                outside = (lj - inter[i][m]) / lj
                out["overhang"][idx[i]] = (wj * outside).sum() / sw
                out["len_gt_frac"][idx[i]] = (wj * (lj > L[i])).sum() / sw
                out["wlog_len_diff"][idx[i]] = (wj * np.log(lj / L[i])).sum() / sw
            m3 = ov03[i]
            if m3.any():
                wj = cw[m3]; sw = wj.sum()
                out["plugin03"][idx[i]] = (wj * minmax_ratio(L[i], cl[m3])).sum() / sw
                ds = (cs[m3] - s[i]) / L[i]; de = (ce[m3] - e[i]) / L[i]
                vs = (wj * ds**2).sum() / sw - ((wj * ds).sum() / sw) ** 2
                ve = (wj * de**2).sum() / sw - ((wj * de).sum() / sw) ** 2
                out["jitter"][idx[i]] = np.sqrt(max(vs, 0) + max(ve, 0))
    for f, v in out.items():
        df[f] = v
    return df
