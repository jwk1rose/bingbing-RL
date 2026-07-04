from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .models import HeroRecord, Loadout


POSITION_BUCKET_BASE = {
    "front": 0.0,
    "前排": 0.0,
    "mid": 1000.0,
    "middle": 1000.0,
    "中排": 1000.0,
    "back": 2000.0,
    "rear": 2000.0,
    "后排": 2000.0,
    "custom": 3000.0,
    "": 3000.0,
}


@dataclass(frozen=True)
class HeroResource:
    record: HeroRecord
    level: int
    stars: int
    rank: int
    equip_ids: tuple[int, ...]
    stats: tuple[tuple[str, float], ...]
    script: str = ""
    image: str = ""

    @property
    def default_unique_equip_id(self) -> int | None:
        return self.record.default_unique_equip_id


@dataclass(frozen=True)
class HeroResourceBundle:
    heroes: tuple[HeroResource, ...]
    loadouts: tuple[Loadout, ...]
    by_hero_id: dict[int, Loadout]
    resources_by_hero_id: dict[int, HeroResource]

    def to_hero_proto(self, loadout: Loadout, *, instance_id: int | None = None) -> dict[str, Any]:
        resource = self.resources_by_hero_id.get(int(loadout.hero_id))
        if resource is None:
            raise KeyError(f"unknown hero id for proto conversion: {loadout.hero_id}")
        item_ids = list(resource.equip_ids[:6])
        while len(item_ids) < 6:
            item_ids.append(0)
        proto: dict[str, Any] = {
            "_id": int(instance_id or loadout.hero_id),
            "_tid": int(loadout.hero_id),
            "_level": int(resource.level),
            "_rank": int(resource.rank),
            "_exp": 0,
            "_gs": int(round(loadout.final_power)),
            "_stars": int(resource.stars),
            "_skill_levels": [int(resource.level) for _ in range(6)],
            "_items": [
                {"_index": index + 1, "_item_id": int(item_id), "_exp": 0}
                for index, item_id in enumerate(item_ids)
            ],
            "_state": "idle",
        }
        if loadout.unique_equip_id is not None:
            proto["_legend_equip"] = {
                "_equip": {
                    "_type_id": int(loadout.unique_equip_id),
                    "_star": int(loadout.unique_equip_star or 0),
                    "_rank": 0,
                }
            }
        return proto


def load_hero_resource_bundle(
    hero_catalog_path: Path,
    *,
    hero_ids: Iterable[int] | None = None,
    unique_equip_star: int = 5,
    standing_overrides: dict[int, float] | None = None,
) -> HeroResourceBundle:
    raw = json.loads(Path(hero_catalog_path).read_text(encoding="utf-8"))
    records = raw["heroes"] if isinstance(raw, dict) and "heroes" in raw else raw
    if not isinstance(records, list):
        raise ValueError("hero catalog must be a list or an object with a heroes list")
    selected = {int(hero_id) for hero_id in hero_ids} if hero_ids is not None else None
    overrides = {int(hero_id): float(value) for hero_id, value in (standing_overrides or {}).items()}
    resources: list[HeroResource] = []
    loadouts: list[Loadout] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        hero_id = int(item["id"])
        if selected is not None and hero_id not in selected:
            continue
        equip_ids = tuple(int(value) for value in item.get("equipIds", []) if int(value) > 0)
        default_unique = equip_ids[-1] if equip_ids else None
        bucket = str(item.get("positionType") or item.get("standingBucket") or "custom").lower()
        standing_rank = overrides.get(hero_id, _standing_rank(bucket, hero_id, index))
        stats = tuple(sorted((str(key), float(value)) for key, value in dict(item.get("stats") or {}).items() if isinstance(value, (int, float))))
        gs = _stat_value(stats, "GS", fallback=float(item.get("power") or 0.0))
        record = HeroRecord(
            hero_id=hero_id,
            name=str(item.get("displayName") or item.get("display_name") or item.get("name") or hero_id),
            standing_rank=standing_rank,
            standing_bucket=bucket or "custom",
            role_tags=tuple(str(tag) for tag in item.get("roleTags", ())) if isinstance(item.get("roleTags"), list) else (),
            base_stats=stats,
            base_power=gs,
            default_unique_equip_id=default_unique,
        )
        resource = HeroResource(
            record=record,
            level=int(item.get("level") or 100),
            stars=int(item.get("stars") or 5),
            rank=int(item.get("rank") or 0),
            equip_ids=equip_ids,
            stats=stats,
            script=str(item.get("script") or ""),
            image=str(item.get("image") or ""),
        )
        loadout = Loadout(
            hero_id=hero_id,
            unique_equip_id=default_unique,
            unique_equip_star=unique_equip_star if default_unique is not None else None,
            normal_equip_ids=tuple(equip_ids[:-1]),
            normal_equip_features=(("normal_equip_count", float(max(0, len(equip_ids) - 1))),),
            level_features=(("level", float(resource.level)), ("rank", float(resource.rank)), ("stars", float(resource.stars))),
            final_stats=stats,
            final_power=gs,
            standing_rank=standing_rank,
            standing_bucket=bucket or "custom",
        )
        resources.append(resource)
        loadouts.append(loadout)
    loadouts.sort(key=lambda loadout: (loadout.standing_rank, loadout.hero_id))
    resources_by_id = {resource.record.hero_id: resource for resource in resources}
    return HeroResourceBundle(
        heroes=tuple(resources),
        loadouts=tuple(loadouts),
        by_hero_id={loadout.hero_id: loadout for loadout in loadouts},
        resources_by_hero_id=resources_by_id,
    )


def _standing_rank(bucket: str, hero_id: int, index: int) -> float:
    base = POSITION_BUCKET_BASE.get(bucket.lower(), POSITION_BUCKET_BASE["custom"])
    return base + float(index) + float(hero_id) / 10_000.0


def _stat_value(stats: tuple[tuple[str, float], ...], key: str, *, fallback: float) -> float:
    for stat_key, value in stats:
        if stat_key == key:
            return float(value)
    return float(fallback)
