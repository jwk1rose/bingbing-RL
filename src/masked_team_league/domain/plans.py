from __future__ import annotations

from dataclasses import dataclass, field

from .formats import MatchFormat
from .hashing import cached_canonical_hash, canonical_hash
from .loadouts import Loadout


@dataclass(frozen=True)
class Team:
    slots: tuple[Loadout, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        if len(self.slots) != 5:
            raise ValueError("Team must contain exactly 5 loadouts")

    @property
    def hero_ids(self) -> tuple[int, ...]:
        return tuple(loadout.hero_id for loadout in self.slots)

    @property
    def unique_equip_ids(self) -> tuple[int, ...]:
        return tuple(loadout.unique_equip_id for loadout in self.slots if loadout.unique_equip_id is not None)

    @property
    def total_cost(self) -> float:
        return sum(loadout.cost for loadout in self.slots)

    @property
    def total_power(self) -> float:
        return sum(loadout.final_power for loadout in self.slots)

    def hash(self) -> str:
        return cached_canonical_hash(self)


@dataclass(frozen=True)
class AttackPlan:
    format: MatchFormat
    teams: tuple[Team, ...]
    source: str
    plan_id: str | None = None
    version: str = "v4"
    season: str = "unknown"
    rank_segment: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "teams", tuple(self.teams))
        if len(self.teams) != self.format.n_teams:
            raise ValueError("AttackPlan team count does not match format")

    def hash(self) -> str:
        return cached_canonical_hash(self)


@dataclass(frozen=True)
class DefensePlan:
    format: MatchFormat
    teams: tuple[Team, ...]
    mask: tuple[tuple[int, ...], ...]
    source: str
    plan_id: str | None = None
    version: str = "v4"
    season: str = "unknown"
    rank_segment: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "teams", tuple(self.teams))
        object.__setattr__(self, "mask", tuple(tuple(int(value) for value in row) for row in self.mask))
        if len(self.teams) != self.format.n_teams:
            raise ValueError("DefensePlan team count does not match format")
        if len(self.mask) != self.format.n_teams:
            raise ValueError("mask team count does not match format")
        if any(len(row) != self.format.team_size for row in self.mask):
            raise ValueError("each mask row must have exactly 5 entries")

    def roster_hash(self) -> str:
        cached = getattr(self, "_roster_hash_cache", None)
        if cached is not None:
            return str(cached)
        value = canonical_hash((self.format, self.teams, self.version, self.season, self.rank_segment))
        object.__setattr__(self, "_roster_hash_cache", value)
        return value

    def hash(self) -> str:
        return cached_canonical_hash(self)


@dataclass(frozen=True)
class ResultMetadata:
    model_version: str = "none"
    data_version: str = "none"
    simulator_version: str = "none"
    league_iteration: int = 0
    random_seed: int = 0
    generation_config_hash: str = "none"
    calibration_version: str = "none"
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)
