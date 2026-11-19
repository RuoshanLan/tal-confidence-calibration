"""The quality_uncertainty chain: localization-quality estimation and its
uncertainty at the LRP operating point (Sec. 5 and supp. Sec. E).

One population, four questions on identical rows:

    point audit           can detector-exported features resolve the
                          localization-quality target q on the LRP head?
    mechanism diagnostic  on the same head's matched subset, does revealing
                          the GT duration ratio xi = log(L_g / L_p) recover
                          the missing resolution? (diagnostic arms; the
                          oracle feature exists nowhere else)
    interval              conformalized nominal-80% [L, U] on the head
    one-sided floor       P{q >= L_q(x)} >= 0.8 on the head

The population is every post-NMS prediction scoring at or above ONE
oLRP-minimising threshold, fitted on the seed's TRAIN partition at the tau=0
one-to-one matching that also defines q, and applied verbatim to CAL and TEST.

Layers:

    data/        prepared data (match + features), pre-NMS cloud features,
                 the video partition
    estimation/  population, LRP selection, arms, estimators, conformal step,
                 metrics, bootstrap
    driver/      the (dataset, seed) shard, readout tables, figure data,
                 output checks, the run driver
"""
