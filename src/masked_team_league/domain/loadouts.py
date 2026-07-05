from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def freeze_mapping(values: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
    if not values:
        return ()
    return tuple(sorted((str(key), float(value)) for key, value in values.items()))


_freeze_mapping = freeze_mapping


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
            base_stats=freeze_mapping(base_stats),
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
            normal_equip_features=freeze_mapping(normal_equip_features),
            level_features=freeze_mapping(level_features),
            final_stats=freeze_mapping(final_stats),
            final_power=hero.base_power if final_power is None else float(final_power),
            standing_rank=hero.standing_rank,
            standing_bucket=hero.standing_bucket,
        )

    @property
    def cost(self) -> float:
        # 对应 tex §501-541：资源成本和下克上目标使用同一套 loadout 成本定义。
        star_cost = 0.0 if self.unique_equip_star is None else 50.0 * self.unique_equip_star
        normal_cost = 10.0 * len(self.normal_equip_ids)
        return float(self.final_power + star_cost + normal_cost)
