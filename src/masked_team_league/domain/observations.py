from __future__ import annotations

from dataclasses import dataclass

from .formats import MatchFormat
from .hashing import Slot, cached_canonical_hash
from .loadouts import Loadout
from .plans import DefensePlan


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
        return cached_canonical_hash(self)


def observe_defense(plan: DefensePlan) -> Observation:
    # 对应 tex §272-331：攻击方只能看到 mask 后的 observation，后续 belief 不能偷看隐藏 loadout。
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
