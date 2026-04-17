"""Explicit benchmark algorithm registry.

This file is intentionally tiny. The task scripts and Optuna scripts should be able to import one
obvious mapping and nothing more.
"""

from __future__ import annotations

from algorithms.ippo import IPPO
from algorithms.iql import IQL
from algorithms.mappo import MAPPO
from algorithms.pimac_v0 import PIMACV0
from algorithms.pimac_v1 import PIMACV1
from algorithms.pimac_v2 import PIMACV2
from algorithms.pimac_v3 import PIMACV3
from algorithms.pimac_v4 import PIMACV4
from algorithms.pimac_v5 import PIMACV5
from algorithms.pimac_v6 import PIMACV6
from algorithms.pimac_v7 import PIMACV7
from algorithms.qmix import QMIX
from algorithms.random import RandomPolicy
from algorithms.vdn import VDN


ALGORITHM_REGISTRY = {
    "random": RandomPolicy,
    "iql": IQL,
    "ippo": IPPO,
    "mappo": MAPPO,
    "qmix": QMIX,
    "vdn": VDN,
    "pimac_v0": PIMACV0,
    "pimac_v1": PIMACV1,
    "pimac_v2": PIMACV2,
    "pimac_v3": PIMACV3,
    "pimac_v4": PIMACV4,
    "pimac_v5": PIMACV5,
    "pimac_v6": PIMACV6,
    "pimac_v7": PIMACV7,
}

ALGORITHM_ORDER = (
    "random",
    "iql",
    "ippo",
    "mappo",
    "qmix",
    "vdn",
    "pimac_v0",
    "pimac_v1",
    "pimac_v2",
    "pimac_v3",
    "pimac_v4",
    "pimac_v5",
    "pimac_v6",
    "pimac_v7",
)


def get_algorithm_class(name: str):
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unsupported algorithm: {name}")
    return ALGORITHM_REGISTRY[name]
