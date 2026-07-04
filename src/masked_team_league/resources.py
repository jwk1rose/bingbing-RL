from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
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
class RuntimeResourceRules:
    unique_legend_equip_ids: tuple[int, ...] = ()
    normal_legend_equip_ids: tuple[int, ...] = ()
    shard_by_hero_id: dict[int, int] | None = None
    astrolabe_by_hero_id: dict[int, dict[str, Any]] | None = None
    astrolabe_attr_values_by_hero_id: dict[int, dict[int, float]] | None = None


@dataclass(frozen=True)
class HeroResourceBundle:
    heroes: tuple[HeroResource, ...]
    loadouts: tuple[Loadout, ...]
    by_hero_id: dict[int, Loadout]
    resources_by_hero_id: dict[int, HeroResource]
    runtime_rules: RuntimeResourceRules | None = None

    def to_hero_proto(
        self,
        loadout: Loadout,
        *,
        instance_id: int | None = None,
        legend_equip_id: int | None = None,
        legend_equip_star: int = 5,
        astrolabe_seed: int | None = None,
        astrolabe_value_ratio: float = 1.0,
    ) -> dict[str, Any]:
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
        if legend_equip_id is not None:
            proto["_legend_equip"] = _legend_equip_proto(int(legend_equip_id), int(legend_equip_star))
        elif loadout.unique_equip_id is not None:
            proto["_legend_equip"] = {
                "_equip": {
                    "_type_id": int(loadout.unique_equip_id),
                    "_star": int(loadout.unique_equip_star or 0),
                    "_rank": 0,
                }
            }
        elif self.runtime_rules and self.runtime_rules.normal_legend_equip_ids:
            proto["_legend_equip"] = _legend_equip_proto(
                _deterministic_normal_legend_equip_id(loadout.hero_id, self.runtime_rules.normal_legend_equip_ids),
                5,
            )
        if self.runtime_rules:
            shard_id = (self.runtime_rules.shard_by_hero_id or {}).get(int(loadout.hero_id))
            if shard_id:
                proto["_shard"] = {"_id": int(shard_id), "_level": 25}
            astrolabe = _astrolabe_proto_from_rules(
                self.runtime_rules,
                int(loadout.hero_id),
                seed=astrolabe_seed,
                value_ratio=astrolabe_value_ratio,
            )
            if astrolabe:
                proto["_astrolabe"] = astrolabe
        return proto


def load_hero_resource_bundle(
    hero_catalog_path: Path,
    *,
    hero_ids: Iterable[int] | None = None,
    unique_equip_star: int = 5,
    unique_legend_equip_ids: Iterable[int] | None = None,
    runtime_rules: RuntimeResourceRules | None = None,
    standing_overrides: dict[int, float] | None = None,
) -> HeroResourceBundle:
    raw = json.loads(Path(hero_catalog_path).read_text(encoding="utf-8"))
    records = raw["heroes"] if isinstance(raw, dict) and "heroes" in raw else raw
    if not isinstance(records, list):
        raise ValueError("hero catalog must be a list or an object with a heroes list")
    selected = {int(hero_id) for hero_id in hero_ids} if hero_ids is not None else None
    unique_source = unique_legend_equip_ids
    if unique_source is None and runtime_rules is not None:
        unique_source = runtime_rules.unique_legend_equip_ids
    unique_ids = {int(equip_id) for equip_id in unique_source or ()}
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
        default_unique = _default_unique_equip_id(item, equip_ids, unique_ids)
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
            normal_equip_ids=tuple(equip_id for equip_id in equip_ids if equip_id != default_unique),
            normal_equip_features=(("normal_equip_count", float(len([equip_id for equip_id in equip_ids if equip_id != default_unique]))),),
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
        runtime_rules=runtime_rules,
    )


def load_decoded_runtime_rules(decoded_dir: Path) -> RuntimeResourceRules:
    decoded = Path(decoded_dir)
    unique_ids, normal_ids = load_legend_equip_pools(decoded / "LegendEquip.lua")
    astrolabe_protos, astrolabe_attr_values = load_astrolabe_resources(decoded)
    return RuntimeResourceRules(
        unique_legend_equip_ids=unique_ids,
        normal_legend_equip_ids=normal_ids,
        shard_by_hero_id=load_shard_by_hero_id(decoded / "ShardToHero.lua"),
        astrolabe_by_hero_id=astrolabe_protos,
        astrolabe_attr_values_by_hero_id=astrolabe_attr_values,
    )


def load_unique_legend_equip_ids(legend_equip_lua_path: Path) -> tuple[int, ...]:
    return load_legend_equip_pools(legend_equip_lua_path)[0]


def load_legend_equip_pools(legend_equip_lua_path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    path = Path(legend_equip_lua_path)
    if not path.exists():
        raise FileNotFoundError(f"legend equip lua does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    unique_ids: list[int] = []
    normal_ids: list[int] = []
    pattern = re.compile(
        r"\[(\d+)\]\s*=\s*\{\d+,\"[^\"]+\",\"[^\"]*\",\"[^\"]*\",\"[^\"]*\",(true|false),",
        re.S,
    )
    for match in pattern.finditer(text):
        if match.group(2) == "true":
            unique_ids.append(int(match.group(1)))
        else:
            normal_ids.append(int(match.group(1)))
    return tuple(sorted(unique_ids)), tuple(sorted(normal_ids))


def load_shard_by_hero_id(shard_lua_path: Path) -> dict[int, int]:
    path = Path(shard_lua_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    shards: dict[int, int] = {}
    pattern = re.compile(r"\[(\d+)\]\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*\"[^\"]+\"", re.S)
    for match in pattern.finditer(text):
        shards[int(match.group(3))] = int(match.group(1))
    return shards


def load_astrolabe_protos(decoded_dir: Path, *, value_ratio: float = 1.0) -> dict[int, dict[str, Any]]:
    return load_astrolabe_resources(decoded_dir, value_ratio=value_ratio)[0]


def load_astrolabe_resources(
    decoded_dir: Path,
    *,
    value_ratio: float = 1.0,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[int, float]]]:
    decoded = Path(decoded_dir)
    astrolabe_path = decoded / "Astrolabe.lua"
    random_attr_path = decoded / "AstrolabeRandomAttr.lua"
    if not astrolabe_path.exists() or not random_attr_path.exists():
        return {}, {}
    random_text = random_attr_path.read_text(encoding="utf-8")
    max_values: dict[int, dict[int, float]] = {}
    random_pattern = re.compile(
        r"\[(\d+)\]\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*\"[^\"]+\".*?\}\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(true|false)\s*,\s*(\d+(?:\.\d+)?)\s*\}",
        re.S,
    )
    for match in random_pattern.finditer(random_text):
        group_id = int(match.group(2))
        attr_id = int(match.group(3))
        max_values.setdefault(group_id, {})[attr_id] = float(match.group(4))

    astrolabe_text = astrolabe_path.read_text(encoding="utf-8")
    result: dict[int, dict[str, Any]] = {}
    attr_values_by_hero: dict[int, dict[int, float]] = {}
    astrolabe_pattern = re.compile(
        r"\[(\d+)\]\s*=\s*\{\s*\d+\s*,\s*\"[^\"]+\"\s*,\s*\"[^\"]*\"\s*,\s*\"[^\"]*\"\s*,\s*\"[^\"]*\"\s*,\s*(\d+)\s*,\s*(\d+)\s*,",
        re.S,
    )
    for match in astrolabe_pattern.finditer(astrolabe_text):
        hero_id = int(match.group(1))
        random_group = int(match.group(3))
        attr_ids = sorted(max_values.get(random_group, {}))[:5]
        if len(attr_ids) < 5:
            continue
        attr_values_by_hero[hero_id] = dict(max_values[random_group])
        result[hero_id] = _astrolabe_proto_from_attr_values(
            {attr_id: max_values[random_group][attr_id] for attr_id in attr_ids},
            value_ratio=value_ratio,
        )
    return result, attr_values_by_hero


def load_peak_arena_camp_hero_ids(decoded_dir: Path, *, camp_group: int = 3) -> tuple[int, ...]:
    decoded = Path(decoded_dir)
    group_text = (decoded / "PeakArenaCampGroup.lua").read_text(encoding="utf-8")
    camp_text = (decoded / "PeakArenaCampList.lua").read_text(encoding="utf-8")
    hero_ids: list[int] = []
    for camp_id in _second_field_numbers(_lua_row_body(group_text, int(camp_group))):
        hero_ids.extend(_second_field_numbers(_lua_row_body(camp_text, camp_id)))
    return tuple(dict.fromkeys(hero_ids))


def _default_unique_equip_id(item: dict[str, Any], equip_ids: tuple[int, ...], unique_ids: set[int]) -> int | None:
    for key in ("uniqueEquipId", "unique_equip_id", "legendEquipId", "legend_equip_id"):
        value = item.get(key)
        if value not in (None, ""):
            return int(value)
    legend = item.get("legendEquip") or item.get("legend_equip") or item.get("_legend_equip")
    if isinstance(legend, dict):
        equip = legend.get("_equip") or legend.get("equip") or legend
        if isinstance(equip, dict):
            value = equip.get("_type_id", equip.get("type_id", equip.get("id")))
            if value not in (None, ""):
                return int(value)
    matching = [equip_id for equip_id in equip_ids if equip_id in unique_ids]
    if not matching:
        return None
    return matching[-1]


def _lua_row_body(text: str, row_id: int) -> str:
    marker = f"[{int(row_id)}]"
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError(f"lua row not found: {row_id}")
    start = text.find("{", marker_index)
    if start < 0:
        raise ValueError(f"lua row has no body: {row_id}")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"lua row body is not balanced: {row_id}")


def _second_field_numbers(row_body: str) -> tuple[int, ...]:
    fields = _split_lua_fields(row_body.strip()[1:-1])
    if len(fields) < 2:
        raise ValueError("lua row has no second field")
    return tuple(int(value) for value in re.findall(r"\[\d+\]\s*=\s*(\d+)", fields[1]))


def _split_lua_fields(text: str) -> list[str]:
    fields: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            fields.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def _legend_equip_proto(equip_id: int, star: int) -> dict[str, Any]:
    return {"_equip": {"_type_id": int(equip_id), "_star": int(star), "_rank": 0}}


def _deterministic_normal_legend_equip_id(hero_id: int, normal_ids: tuple[int, ...]) -> int:
    return int(normal_ids[int(hero_id) % len(normal_ids)])


def _astrolabe_proto_from_rules(
    rules: RuntimeResourceRules,
    hero_id: int,
    *,
    seed: int | None,
    value_ratio: float,
) -> dict[str, Any] | None:
    attr_values = (rules.astrolabe_attr_values_by_hero_id or {}).get(hero_id)
    if attr_values and seed is not None and len(attr_values) >= 5:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(list(attr_values), 5))
        return _astrolabe_proto_from_attr_values(
            {attr_id: attr_values[attr_id] for attr_id in chosen},
            value_ratio=value_ratio,
        )
    return (rules.astrolabe_by_hero_id or {}).get(hero_id)


def _astrolabe_proto_from_attr_values(
    attr_values: dict[int, float],
    *,
    value_ratio: float,
) -> dict[str, Any]:
    return {
        "_level": 80,
        "_gs": 234,
        "_is_unlock": True,
        "_stars": [
            {"_index": index, "_id": attr_id, "_value": int(attr_values[attr_id] * value_ratio + 0.5)}
            for index, attr_id in enumerate(sorted(attr_values), start=1)
        ],
    }


def _standing_rank(bucket: str, hero_id: int, index: int) -> float:
    base = POSITION_BUCKET_BASE.get(bucket.lower(), POSITION_BUCKET_BASE["custom"])
    return base + float(index) + float(hero_id) / 10_000.0


def _stat_value(stats: tuple[tuple[str, float], ...], key: str, *, fallback: float) -> float:
    for stat_key, value in stats:
        if stat_key == key:
            return float(value)
    return float(fallback)
