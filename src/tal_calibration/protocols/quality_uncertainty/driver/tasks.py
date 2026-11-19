"""Task identity and config loading for the quality_uncertainty chain.

One task per (registered arm, seed). The shard runs the point audit, the
matched-subset mechanism diagnostic, the interval and the one-sided floor on
ONE LRP selection, so point, interval and floor consume identical rows by
construction.

The registered (dataset, detector) arms are REGISTRATION KEYS into
``configs/datasets.yaml``, written here as a literal and reconciled against the
yaml by ``registered_arms()`` in both directions: an arm declared in one place
and missing from the other fails loudly. The matrix is ragged (TadTR runs on
THUMOS14 only; EPIC noun, Ego4D-MQ and ActivityNet-1.3 have ActionFormer
alone), and iteration order is owned by REGISTERED_ARMS, so CSV row order is
stable however a config lists its datasets.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..data.schemas import REGISTERED_SEEDS

EXPERIMENT = "quality_uncertainty"

REGISTERED_ARMS = (
    "thumos", "thumos_tridet", "thumos_tadtr", "thumos_tetad",
    "epic_verb", "epic_verb_tridet", "epic_verb_tetad",
    "epic_noun",
    "ego4d",
    "anet",
)


class ArmRegistrationError(RuntimeError):
    """configs/datasets.yaml and REGISTERED_ARMS disagree, or a config asks
    for an arm or seed that is not registered."""


def registered_arms():
    """``key -> entry`` for every registered arm, in REGISTERED_ARMS order.

    Also asserts every arm's canonical anchor is itself registered and
    self-anchored, so a detector can never resolve its annotation and video
    partition through an entry that is itself a detector.
    """
    from ....datasets import load_datasets
    cfg = load_datasets()
    missing = sorted(set(REGISTERED_ARMS) - set(cfg))
    unexpected = sorted(set(cfg) - set(REGISTERED_ARMS))
    if missing or unexpected:
        raise ArmRegistrationError(
            "configs/datasets.yaml != REGISTERED_ARMS: "
            f"missing {missing}, unexpected {unexpected}")
    for key in REGISTERED_ARMS:
        canon = cfg[key]["canonical"]
        if canon not in cfg or cfg[canon]["canonical"] != canon:
            raise ArmRegistrationError(
                f"{key}: canonical anchor {canon!r} is not a registered, "
                "self-anchored dataset")
    return {k: cfg[k] for k in REGISTERED_ARMS}


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    #: the registered feature interface this arm runs. Resolved from the
    #: registration, never chosen by a config: a detector arm on the reduced
    #: interface must not be handed the 30 pre-NMS cloud columns.
    feature_interface: str = "full_observable"
    #: the detector that produced this arm's predictions
    detector: str = "actionformer"
    #: the dataset whose official evaluation split defines the population
    canonical_dataset: str = ""

    @property
    def task_id(self):
        return f"{self.dataset}__seed{self.seed}"


def expand_tasks(config, arms=None):
    """One Task per (registered arm, seed), in REGISTERED_ARMS order.

    The feature interface, the detector and the canonical anchor come from the
    REGISTRATION: a config chooses WHICH registered arms to run, never what an
    arm is.
    """
    arms = registered_arms() if arms is None else arms
    sub = config["experiments"][EXPERIMENT]
    wanted = set(sub["datasets"])
    unknown = sorted(wanted - set(arms))
    if unknown:
        raise ArmRegistrationError(f"config names unregistered arms {unknown}")
    seeds = [int(s) for s in sub["seeds"]]
    bad = sorted(set(seeds) - set(REGISTERED_SEEDS))
    if bad or len(seeds) != len(set(seeds)):
        raise ArmRegistrationError(
            f"seeds {seeds} must be distinct and drawn from {REGISTERED_SEEDS}")
    return [Task(d, s,
                 feature_interface=arms[d]["feature_interface"],
                 detector=arms[d]["detector"],
                 canonical_dataset=arms[d]["canonical"])
            for d in arms if d in wanted for s in seeds]


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_PATH = CONFIG_DIR / "formal_all_datasets.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
