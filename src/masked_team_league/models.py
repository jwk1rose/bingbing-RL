from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
from typing import Any, Mapping, TypeAlias

Slot: TypeAlias = tuple[int, int]


def _freeze_mapping(values: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
    if not values:
        return ()
    return tuple(sorted((str(key), float(value)) for key, value in values.items()))


def _canonical(obj: Any) -> Any:
    if is_dataclass(obj):
        return {item.name: _canonical(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, dict):
        return {str(key): _canonical(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (tuple, list)):
        return [_canonical(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical(value) for value in obj)
    return obj


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatchFormat:
    n_teams: int
    team_size: int = 5
    win_required: int | None = None
    max_hidden_per_team: int = 2
    max_hidden_total: int = 10

    def __post_init__(self) -> None:
        if self.win_required is None:
            object.__setattr__(self, "win_required", self.n_teams // 2 + 1)
        if self.n_teams not in (3, 5):
            raise ValueError("n_teams must be 3 or 5")
        if self.team_size != 5:
            raise ValueError("team_size must be 5")
        if self.win_required is None or not (1 <= self.win_required <= self.n_teams):
            raise ValueError("win_required must be within the match length")
        if self.max_hidden_per_team < 0 or self.max_hidden_total < 0:
            raise ValueError("mask limits must be non-negative")


@dataclass(frozen=True)
class HeroRecord:
    hero_id: int
    name: str
    standing_rank: float
    standing_bucket: str
    role_tags: tuple[str, ...] = ()
    base_stats: tuple[tuple[str, float], ...] = ()
    base_power: float = 0.0
    default_unique_equip_id: int | None = None

    @classmethod
    def from_mapping(
        cls,
        *,
        hero_id: int,
        name: str,
        standing_rank: float,
        standing_bucket: str,
        role_tags: tuple[str, ...] = (),
        base_stats: Mapping[str, float] | None = None,
        base_power: float = 0.0,
        default_unique_equip_id: int | None = None,
    ) -> "HeroRecord":
        return cls(
            hero_id=hero_id,
            name=name,
            standing_rank=float(standing_rank),
            standing_bucket=standing_bucket,
            role_tags=tuple(role_tags),
            base_stats=_freeze_mapping(base_stats),
            base_power=float(base_power),
            default_unique_equip_id=default_unique_equip_id,
        )


@dataclass(frozen=True)
class Loadout:
    hero_id: int
    unique_equip_id: int | None
    unique_equip_star: int | None
    normal_equip_ids: tuple[int, ...] = ()
    normal_equip_features: tuple[tuple[str, float], ...] = ()
    level_features: tuple[tuple[str, float], ...] = ()
    final_stats: tuple[tuple[str, float], ...] = ()
    final_power: float = 0.0
    standing_rank: float = 0.0
    standing_bucket: str = "custom"

    def __post_init__(self) -> None:
        if self.unique_equip_id is None:
            if self.unique_equip_star is not None:
                raise ValueError("unique_equip_star must be None when unique_equip_id is None")
        elif self.unique_equip_star not in (3, 4, 5):
            raise ValueError("unique_equip_star must be 3, 4, or 5 when unique_equip_id is set")
        object.__setattr__(self, "normal_equip_ids", tuple(self.normal_equip_ids))
        object.__setattr__(self, "normal_equip_features", tuple(sorted(self.normal_equip_features)))
        object.__setattr__(self, "level_features", tuple(sorted(self.level_features)))
        object.__setattr__(self, "final_stats", tuple(sorted(self.final_stats)))
        object.__setattr__(self, "final_power", float(self.final_power))
        object.__setattr__(self, "standing_rank", float(self.standing_rank))

    @classmethod
    def from_hero(
        cls,
        hero: HeroRecord,
        *,
        unique_equip_id: int | None | object = object(),
        unique_equip_star: int | None = 5,
        normal_equip_ids: tuple[int, ...] = (),
        normal_equip_features: Mapping[str, float] | None = None,
        level_features: Mapping[str, float] | None = None,
        final_stats: Mapping[str, float] | None = None,
        final_power: float | None = None,
    ) -> "Loadout":
        equip_id = hero.default_unique_equip_id if not isinstance(unique_equip_id, int) and unique_equip_id is not None else unique_equip_id
        if equip_id is None:
            unique_equip_star = None
        return cls(
            hero_id=hero.hero_id,
            unique_equip_id=equip_id if isinstance(equip_id, int) else None,
            unique_equip_star=unique_equip_star,
            normal_equip_ids=tuple(normal_equip_ids),
            normal_equip_features=_freeze_mapping(normal_equip_features),
            level_features=_freeze_mapping(level_features),
            final_stats=_freeze_mapping(final_stats),
            final_power=hero.base_power if final_power is None else float(final_power),
            standing_rank=hero.standing_rank,
            standing_bucket=hero.standing_bucket,
        )

    @property
    def cost(self) -> float:
        star_cost = 0.0 if self.unique_equip_star is None else 50.0 * self.unique_equip_star
        normal_cost = 10.0 * len(self.normal_equip_ids)
        return float(self.final_power + star_cost + normal_cost)


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
        return canonical_hash(self)


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
        return canonical_hash(self)


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
        return canonical_hash((self.format, self.teams, self.version, self.season, self.rank_segment))

    def hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class VisibleSlot:
    hero_id: int | None
    unique_equip_id: int | None
    unique_equip_star: int | None
    normal_equip_summary: tuple[int, ...] | None
    final_power: float | None
    standing_rank: float | None
    is_hidden: bool
    loadout: Loadout | None = None

    @classmethod
    def hidden(cls) -> "VisibleSlot":
        return cls(None, None, None, None, None, None, True, None)

    @classmethod
    def from_loadout(cls, loadout: Loadout) -> "VisibleSlot":
        return cls(
            hero_id=loadout.hero_id,
            unique_equip_id=loadout.unique_equip_id,
            unique_equip_star=loadout.unique_equip_star,
            normal_equip_summary=loadout.normal_equip_ids,
            final_power=loadout.final_power,
            standing_rank=loadout.standing_rank,
            is_hidden=False,
            loadout=loadout,
        )


@dataclass(frozen=True)
class Observation:
    format: MatchFormat
    slots: tuple[tuple[VisibleSlot, ...], ...]
    hidden_slots: tuple[Slot, ...]
    visible_heroes: frozenset[int]
    visible_unique_equip_ids: frozenset[int]
    visible_unique_equip_stars: tuple[tuple[int, int], ...]
    version: str = "v4"
    season: str = "unknown"
    rank_segment: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(tuple(row) for row in self.slots))
        object.__setattr__(self, "hidden_slots", tuple(self.hidden_slots))
        object.__setattr__(self, "visible_heroes", frozenset(self.visible_heroes))
        object.__setattr__(self, "visible_unique_equip_ids", frozenset(self.visible_unique_equip_ids))
        object.__setattr__(self, "visible_unique_equip_stars", tuple(sorted(self.visible_unique_equip_stars)))
        if len(self.slots) != self.format.n_teams:
            raise ValueError("Observation team count does not match format")
        if any(len(row) != self.format.team_size for row in self.slots):
            raise ValueError("each observation row must have exactly 5 slots")

    def hash(self) -> str:
        return canonical_hash(self)


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


def observe_defense(plan: DefensePlan) -> Observation:
    rows: list[tuple[VisibleSlot, ...]] = []
    hidden_slots: list[Slot] = []
    visible_heroes: set[int] = set()
    visible_equip_ids: set[int] = set()
    visible_stars: dict[int, int] = {}
    for team_idx, (team, mask_row) in enumerate(zip(plan.teams, plan.mask), start=1):
        obs_row: list[VisibleSlot] = []
        for slot_idx, (loadout, hidden) in enumerate(zip(team.slots, mask_row), start=1):
            if hidden:
                obs_row.append(VisibleSlot.hidden())
                hidden_slots.append((team_idx, slot_idx))
                continue
            obs_row.append(VisibleSlot.from_loadout(loadout))
            visible_heroes.add(loadout.hero_id)
            if loadout.unique_equip_id is not None:
                visible_equip_ids.add(loadout.unique_equip_id)
                if loadout.unique_equip_star is not None:
                    visible_stars[loadout.unique_equip_id] = loadout.unique_equip_star
        rows.append(tuple(obs_row))
    return Observation(
        format=plan.format,
        slots=tuple(rows),
        hidden_slots=tuple(hidden_slots),
        visible_heroes=frozenset(visible_heroes),
        visible_unique_equip_ids=frozenset(visible_equip_ids),
        visible_unique_equip_stars=tuple(sorted(visible_stars.items())),
        version=plan.version,
        season=plan.season,
        rank_segment=plan.rank_segment,
    )
