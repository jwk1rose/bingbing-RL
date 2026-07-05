"""领域对象层，对应 tex §142-331。

这里只放 MatchFormat、Loadout、Team、Plan、Observation 和规范化 hash。
本层不依赖 oracle、training、backend 或 reporting。
"""

from .formats import MatchFormat
from .hashing import Slot, _cached_canonical_hash, cached_canonical_hash, canonical_hash
from .loadouts import HeroRecord, Loadout, _freeze_mapping, freeze_mapping
from .observations import Observation, VisibleSlot, observe_defense
from .plans import AttackPlan, DefensePlan, ResultMetadata, Team

__all__ = [
    "AttackPlan",
    "DefensePlan",
    "HeroRecord",
    "Loadout",
    "MatchFormat",
    "Observation",
    "ResultMetadata",
    "Slot",
    "Team",
    "VisibleSlot",
    "_cached_canonical_hash",
    "_freeze_mapping",
    "cached_canonical_hash",
    "canonical_hash",
    "freeze_mapping",
    "observe_defense",
]
