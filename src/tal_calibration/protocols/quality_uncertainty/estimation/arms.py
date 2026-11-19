"""The arm registries and the ground-truth feature ban.

    POINT_ARMS        E[q|x] point regressors on the LRP head.
    UNCERTAINTY_ARMS  interval and floor arms.
    MECHANISM_ARMS    The matched-subset obs-vs-oracle diagnostic (Sec. 5,
                      Table 4 "+xi"); the only place the oracle feature exists.
                      The gap is taken between these two same-subset fits.

No GT-derived quantity may enter an inference arm: on this population every
such column is NaN or a sentinel exactly on the unmatched rows, so it would
encode the matched status. The one carve-out is the oracle's ``oracle_r``,
inside the diagnostic mechanism registry. The class-duration priors
(``prior_ratio`` / ``prior_cdf``) are allowed: they are class-level
statistics from TRAIN-video GT only.
"""
from dataclasses import dataclass, field

from ..data.cloud import CLOUD_FEATS, CLOUD_X_FEATS
from ..data.observable_features import NBR_FEATS, SELF_FEATS

#: every per-row ground-truth column of the base table, plus the oracle
#: column the diagnostic arms use
FORBIDDEN_INFERENCE_FEATURES = (
    "matched_tiou", "matched_gt_dur", "matched_gt_idx",
    "max_tiou", "ceiling", "best_gt_dur", "best_gt_idx",
    "oracle_y", "oracle_r", "oracle_matched",
    "is_hit", "q", "target",
)

#: the oracle feature xi = log(matched_gt_dur / pred_len), on matched rows only
ORACLE_FEATURE = "oracle_r"

#: point-arm estimator kinds
POINT_ESTIMATORS = ("gbr",)
#: interval/floor estimator kinds
#:   trivial         fixed [0,0] / floor 0, not fitted
#:   marginal        constant TRAIN quantiles, then conformalized
#:   quantile        GBM conditional head WITH fallback-to-marginal
UNC_ESTIMATORS = ("trivial", "marginal", "quantile")
UNFITTED_ESTIMATORS = ("trivial",)


@dataclass(frozen=True)
class Arm:
    name: str
    features: tuple
    estimator: str
    fold_clean_priors: bool = False
    fitted: bool = True          # trivial_zero is a fixed output, never fitted
    diagnostic: bool = False     # mechanism arms only
    allow: tuple = field(default=())   # explicit per-arm ban carve-out

    def check(self, df):
        feats = list(self.features)
        if len(set(feats)) != len(feats):
            raise ValueError(f"arm {self.name}: duplicate features")
        banned = [f for f in feats
                  if f in FORBIDDEN_INFERENCE_FEATURES and f not in self.allow]
        if banned:
            raise ValueError(
                f"arm {self.name}: GT-derived inference features {banned} are "
                "forbidden")
        if self.allow and not self.diagnostic:
            raise ValueError(f"arm {self.name}: only a diagnostic arm may "
                             "carry a ban carve-out")
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise ValueError(f"arm {self.name}: missing columns {missing}")
        return feats


FULL_OBSERVABLE_FEATS = tuple(
    SELF_FEATS + NBR_FEATS + CLOUD_FEATS + CLOUD_X_FEATS)

# --------------------------------------------------------------------------
# Point estimation
# --------------------------------------------------------------------------
POINT_ARMS = {a.name: a for a in (
    Arm("score_only", ("score",), "gbr"),
    # self features + post-NMS neighbourhood + pre-NMS cloud summaries, with
    # fold-clean class-duration priors
    Arm(
        "full_observable", FULL_OBSERVABLE_FEATS, "gbr",
        fold_clean_priors=True),
)}
#: the primary point arm
PRIMARY_POINT_ARM = "full_observable"

# --------------------------------------------------------------------------
# Interval and floor arms
# --------------------------------------------------------------------------
UNCERTAINTY_ARMS = {a.name: a for a in (
    # fixed interval [0,0] and floor 0, not fitted but scored by the same
    # metric code; the interval covers exactly the unmatched (q = 0) rows,
    # which makes it the negative control for coverage
    Arm("trivial_zero", (), "trivial", fitted=False),
    # pooled TRAIN quantiles of the head, then conformalized
    Arm("population_marginal", (), "marginal"),
    Arm("score_only", ("score",), "quantile"),
    Arm(
        "full_observable", FULL_OBSERVABLE_FEATS, "quantile",
        fold_clean_priors=True),
)}
#: the null every paired CI is taken against
NULL_ARM = "population_marginal"
#: the arm whose conditional quantiles the paper reports
FOCUS_ARM = "full_observable"
PAIR_BASELINES = (NULL_ARM,)
BASELINE_TAGS = {NULL_ARM: "PM"}

# --------------------------------------------------------------------------
# Matched-subset obs-vs-oracle mechanism diagnostic
# --------------------------------------------------------------------------
MECH_OBS_ARM = "matched_obs"
MECH_ORACLE_ARM = "matched_oracle"
MECHANISM_ARMS = {a.name: a for a in (
    # the observable interface refitted on the head's matched subset; the
    # oracle gap is taken against this arm only
    Arm(
        MECH_OBS_ARM, FULL_OBSERVABLE_FEATS, "gbr", fold_clean_priors=True,
        diagnostic=True),
    # the same interface plus xi: GT-derived and not deployable, so it exists
    # only in this diagnostic registry
    Arm(
        MECH_ORACLE_ARM, FULL_OBSERVABLE_FEATS + (ORACLE_FEATURE,), "gbr",
        fold_clean_priors=True, diagnostic=True, allow=(ORACLE_FEATURE,)),
)}


# --------------------------------------------------------------------------
# Registered feature interfaces.
#
# The dense detectors run the 45-feature interface above off their pre-NMS
# candidate pool. TadTR and TE-TAD are set-prediction decoders; their
# observable models use only the 15 post-NMS features (self + post-NMS
# neighbourhood), and their raw pre-NMS outputs are not used, since the 30
# pre-NMS proposal features were designed around the dense detectors'
# candidate pools.
#
# The reduced arms form a SEPARATE BUNDLE rather than extra entries in the
# registries above, because the bundle width affects the numbers: `shard`
# stacks every (arm, output, stat) into one matrix, `bootstrap` computes
# `M.T @ w`, and BLAS picks its blocking from the matrix width, so a different
# width can move unchanged columns at the 1e-13 level. Every bundle therefore
# registers the SAME number of arms per table.
#
# A name shared between bundles means the identical estimand: `score_only` is
# the same one-feature arm either way, while the arms that carry an interface
# have bundle-specific names (`full_observable` vs `post_nms_observable`).
# --------------------------------------------------------------------------
POST_NMS_OBSERVABLE_FEATS = tuple(SELF_FEATS + NBR_FEATS)

#: the reduced interface's point / uncertainty arm name
POST_NMS_ARM = "post_nms_observable"
#: its mechanism twins -- reduced TOGETHER, or the obs-vs-oracle gap measures
#: nothing (and is structurally impossible: the cloud columns are never joined,
#: so a 45-feature arm would die in Arm.check)
MECH_OBS_ARM_PN = "matched_post_nms_obs"
MECH_ORACLE_ARM_PN = "matched_post_nms_oracle"

POINT_ARMS_PN = {a.name: a for a in (
    POINT_ARMS["score_only"],
    # the full interface without the pre-NMS cloud summaries: self features
    # plus the post-NMS same-class neighbourhood, with fold-clean
    # class-duration priors
    Arm(
        POST_NMS_ARM, POST_NMS_OBSERVABLE_FEATS, "gbr",
        fold_clean_priors=True),
)}

UNCERTAINTY_ARMS_PN = {a.name: a for a in (
    UNCERTAINTY_ARMS["trivial_zero"], UNCERTAINTY_ARMS["population_marginal"],
    UNCERTAINTY_ARMS["score_only"],
    Arm(
        POST_NMS_ARM, POST_NMS_OBSERVABLE_FEATS, "quantile",
        fold_clean_priors=True),
)}

MECHANISM_ARMS_PN = {a.name: a for a in (
    Arm(
        MECH_OBS_ARM_PN, POST_NMS_OBSERVABLE_FEATS, "gbr",
        fold_clean_priors=True, diagnostic=True),
    # this pair's gap is a within-interface contrast and is not comparable
    # with the full interface's gap
    Arm(
        MECH_ORACLE_ARM_PN, POST_NMS_OBSERVABLE_FEATS + (ORACLE_FEATURE,),
        "gbr", fold_clean_priors=True, diagnostic=True,
        allow=(ORACLE_FEATURE,)),
)}

@dataclass(frozen=True)
class ArmBundle:
    """One registered feature interface, as the four tables a shard runs."""
    name: str
    point: dict
    uncertainty: dict
    mechanism: dict
    observable_arm: str
    mech_obs: str
    mech_oracle: str


BUNDLES = {
    "full_observable": ArmBundle(
        "full_observable", POINT_ARMS, UNCERTAINTY_ARMS, MECHANISM_ARMS,
        PRIMARY_POINT_ARM, MECH_OBS_ARM, MECH_ORACLE_ARM),
    "post_nms_observable": ArmBundle(
        "post_nms_observable", POINT_ARMS_PN, UNCERTAINTY_ARMS_PN,
        MECHANISM_ARMS_PN, POST_NMS_ARM, MECH_OBS_ARM_PN, MECH_ORACLE_ARM_PN),
}


def bundle_for(feature_interface):
    """The registered bundle a detector arm runs. KeyError on an unknown name:
    there is no default interface."""
    return BUNDLES[feature_interface]


def assert_registries_wellformed():
    """Structural invariants of the registries (asserted at shard entry and
    by tests, independent of any dataframe)."""
    for name, arm in POINT_ARMS.items():
        if arm.estimator not in POINT_ESTIMATORS:
            raise ValueError(f"point arm {name}: unknown estimator "
                             f"{arm.estimator}")
        _assert_clean(name, arm)
    for name, arm in UNCERTAINTY_ARMS.items():
        if arm.estimator not in UNC_ESTIMATORS:
            raise ValueError(f"uncertainty arm {name}: unknown estimator "
                             f"{arm.estimator}")
        if arm.fitted == (arm.estimator in UNFITTED_ESTIMATORS):
            raise ValueError(f"uncertainty arm {name}: fitted flag "
                             f"contradicts estimator {arm.estimator}")
        _assert_clean(name, arm)
    if NULL_ARM not in UNCERTAINTY_ARMS or FOCUS_ARM not in UNCERTAINTY_ARMS:
        raise ValueError("null/focus arm missing from the uncertainty registry")
    if UNCERTAINTY_ARMS[NULL_ARM].features:
        raise ValueError("the marginal null must be feature-free")

    # the three registries must agree on the registered observable interface
    for reg, name in ((POINT_ARMS, PRIMARY_POINT_ARM),
                      (UNCERTAINTY_ARMS, FOCUS_ARM),
                      (MECHANISM_ARMS, MECH_OBS_ARM)):
        if reg[name].features != FULL_OBSERVABLE_FEATS:
            raise ValueError(f"{name}: full-observable interface drifted")
    # the oracle arm is the interface plus exactly the oracle feature
    oracle = MECHANISM_ARMS[MECH_ORACLE_ARM]
    if oracle.features != FULL_OBSERVABLE_FEATS + (ORACLE_FEATURE,):
        raise ValueError("matched_oracle must be the registered interface "
                         "plus exactly oracle_r")
    for name, arm in MECHANISM_ARMS.items():
        if not arm.diagnostic:
            raise ValueError(f"mechanism arm {name} must be diagnostic")
    # the oracle feature exists nowhere outside the mechanism registry
    for reg_name, reg in (("point", POINT_ARMS),
                          ("uncertainty", UNCERTAINTY_ARMS)):
        for name, arm in reg.items():
            if ORACLE_FEATURE in arm.features:
                raise ValueError(f"{reg_name} arm {name} carries the oracle "
                                 "feature")

    return True


def _assert_clean(name, arm):
    if arm.diagnostic or arm.allow:
        raise ValueError(f"inference arm {name} must not be diagnostic or "
                         "carry a ban carve-out")
    banned = [f for f in arm.features if f in FORBIDDEN_INFERENCE_FEATURES]
    if banned:
        raise ValueError(f"arm {name} carries banned features {banned}")


