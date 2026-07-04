from __future__ import annotations

import pytest

from masked_team_league.models import HeroRecord, Loadout, MatchFormat


@pytest.fixture
def fmt3() -> MatchFormat:
    return MatchFormat(n_teams=3)


@pytest.fixture
def fmt5() -> MatchFormat:
    return MatchFormat(n_teams=5)


@pytest.fixture
def heroes() -> tuple[HeroRecord, ...]:
    records = []
    buckets = ["front", "front", "mid", "mid", "back"]
    for idx in range(1, 41):
        bucket = buckets[(idx - 1) % len(buckets)]
        records.append(
            HeroRecord.from_mapping(
                hero_id=idx,
                name=f"hero-{idx}",
                standing_rank=float(idx),
                standing_bucket=bucket,
                role_tags=(bucket,),
                base_stats={"hp": 1000 + idx, "atk": 100 + idx},
                base_power=1000.0 + idx * 15.0,
                default_unique_equip_id=1000 + idx,
            )
        )
    return tuple(records)


@pytest.fixture
def loadouts(heroes: tuple[HeroRecord, ...]) -> tuple[Loadout, ...]:
    result = []
    for hero in heroes:
        star = 3 + (hero.hero_id % 3)
        result.append(
            Loadout.from_hero(
                hero,
                unique_equip_star=star,
                normal_equip_ids=(2000 + hero.hero_id % 7,),
                normal_equip_features={"normal_count": 1.0},
                level_features={"level": 100.0},
                final_stats={"hp": 1000.0 + hero.hero_id},
                final_power=hero.base_power + star * 20.0,
            )
        )
    return tuple(result)
