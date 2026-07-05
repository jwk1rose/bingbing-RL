from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..domain import AttackPlan, DefensePlan, Loadout, Observation, Team, canonical_hash
from ..league.manager import StrategyRecord
from ..scoring import match_win_probability


CORE_TABLE_SCHEMA_VERSION = "core_tables.v1"


@dataclass(frozen=True)
class LoadoutTableRecord:
    loadout_id: str
    hero_id: int
    unique_equip_id: int | None
    unique_equip_star: int | None
    normal_equip_ids: tuple[int, ...]
    normal_equip_features: tuple[tuple[str, float], ...]
    level_features: tuple[tuple[str, float], ...]
    final_stats: tuple[tuple[str, float], ...]
    final_power: float
    standing_rank: float
    standing_bucket: str
    rarity_cost: float
    season: str
    data_version: str
    table: str = "LoadoutTable"
    schema_version: str = CORE_TABLE_SCHEMA_VERSION

    @classmethod
    def from_loadout(cls, loadout: Loadout, *, data_version: str, season: str) -> "LoadoutTableRecord":
        return cls(
            loadout_id=canonical_hash(loadout),
            hero_id=loadout.hero_id,
            unique_equip_id=loadout.unique_equip_id,
            unique_equip_star=loadout.unique_equip_star,
            normal_equip_ids=tuple(loadout.normal_equip_ids),
            normal_equip_features=tuple(loadout.normal_equip_features),
            level_features=tuple(loadout.level_features),
            final_stats=tuple(loadout.final_stats),
            final_power=float(loadout.final_power),
            standing_rank=float(loadout.standing_rank),
            standing_bucket=loadout.standing_bucket,
            rarity_cost=float(loadout.cost),
            season=season,
            data_version=data_version,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class SingleMatchupTableRecord:
    attack_team_hash: str
    defense_team_hash: str
    sim_or_real: str
    num_games: int
    wins: int
    losses: int
    empirical_winrate: float
    confidence_lower: float
    confidence_upper: float
    mean_duration: float
    mean_margin: float
    simulator_version: str
    model_version: str
    cache_key_hash: str
    table: str = "SingleMatchupTable"
    schema_version: str = CORE_TABLE_SCHEMA_VERSION

    @classmethod
    def from_matchup(
        cls,
        attack: Team,
        defense: Team,
        *,
        sim_or_real: str,
        num_games: int,
        wins: int,
        mean_duration: float,
        mean_margin: float,
        simulator_version: str,
        model_version: str,
    ) -> "SingleMatchupTableRecord":
        games = max(int(num_games), 0)
        attack_hash = attack.hash()
        defense_hash = defense.hash()
        win_count = min(max(int(wins), 0), games)
        winrate = win_count / games if games else 0.0
        lower, upper = _wilson_interval(win_count, games)
        return cls(
            attack_team_hash=attack_hash,
            defense_team_hash=defense_hash,
            sim_or_real=sim_or_real,
            num_games=games,
            wins=win_count,
            losses=max(games - win_count, 0),
            empirical_winrate=winrate,
            confidence_lower=lower,
            confidence_upper=upper,
            mean_duration=float(mean_duration),
            mean_margin=float(mean_margin),
            simulator_version=simulator_version,
            model_version=model_version,
            cache_key_hash=canonical_hash((attack_hash, defense_hash, sim_or_real, simulator_version, model_version)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class PlanMatchTableRecord:
    attack_plan_hash: str
    defense_plan_hash: str
    defense_roster_hash: str
    format_teams: int
    win_required: int
    sim_or_real: str
    num_games: int
    round_win_rates: tuple[float, ...]
    empirical_winrate: float
    simulator_version: str
    model_version: str
    table: str = "PlanMatchTable"
    schema_version: str = CORE_TABLE_SCHEMA_VERSION

    @classmethod
    def from_plan_match(
        cls,
        attack: AttackPlan,
        defense: DefensePlan,
        *,
        sim_or_real: str,
        num_games: int,
        round_win_rates: tuple[float, ...],
        simulator_version: str,
        model_version: str,
    ) -> "PlanMatchTableRecord":
        return cls(
            attack_plan_hash=attack.hash(),
            defense_plan_hash=defense.hash(),
            defense_roster_hash=defense.roster_hash(),
            format_teams=attack.format.n_teams,
            win_required=int(attack.format.win_required or 0),
            sim_or_real=sim_or_real,
            num_games=int(num_games),
            round_win_rates=tuple(float(value) for value in round_win_rates),
            empirical_winrate=match_win_probability(tuple(float(value) for value in round_win_rates), attack.format.win_required),
            simulator_version=simulator_version,
            model_version=model_version,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ObservationTableRecord:
    observation_hash: str
    format_teams: int
    hidden_slots: tuple[tuple[int, int], ...]
    visible_heroes: tuple[int, ...]
    visible_unique_equip_ids: tuple[int, ...]
    visible_unique_equip_stars: tuple[tuple[int, int], ...]
    position_bounds: tuple[tuple[str, float, float], ...]
    domain_sizes: tuple[tuple[str, int], ...]
    real_frequency: float
    belief_candidate_count: int
    belief_entropy: float
    season: str
    rank_segment: str
    table: str = "ObservationTable"
    schema_version: str = CORE_TABLE_SCHEMA_VERSION

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        real_frequency: float = 0.0,
        belief_candidate_count: int = 0,
        belief_entropy: float = 0.0,
        position_bounds: Mapping[tuple[int, int], tuple[float, float]] | None = None,
        domain_sizes: Mapping[tuple[int, int], int] | None = None,
    ) -> "ObservationTableRecord":
        return cls(
            observation_hash=observation.hash(),
            format_teams=observation.format.n_teams,
            hidden_slots=tuple((int(team), int(slot)) for team, slot in observation.hidden_slots),
            visible_heroes=tuple(sorted(int(value) for value in observation.visible_heroes)),
            visible_unique_equip_ids=tuple(sorted(int(value) for value in observation.visible_unique_equip_ids)),
            visible_unique_equip_stars=tuple((int(equip), int(star)) for equip, star in observation.visible_unique_equip_stars),
            position_bounds=_slot_float_rows(position_bounds or {}),
            domain_sizes=_slot_int_rows(domain_sizes or {}),
            real_frequency=float(real_frequency),
            belief_candidate_count=int(belief_candidate_count),
            belief_entropy=float(belief_entropy),
            season=observation.season,
            rank_segment=observation.rank_segment,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class LeagueStrategyTableRecord:
    strategy_id: str
    strategy_type: str
    role: str
    plan_hash: str
    created_iteration: int
    sim_score: float
    real_score: float | None
    cluster_id: str
    resource_cost: float
    underdog_gap: float
    active: bool
    retired_reason: str | None
    source: str
    table: str = "LeagueStrategyTable"
    schema_version: str = CORE_TABLE_SCHEMA_VERSION

    @classmethod
    def from_strategy(cls, record: StrategyRecord, *, real_score: float | None = None) -> "LeagueStrategyTableRecord":
        return cls(
            strategy_id=record.strategy_id,
            strategy_type=record.side,
            role=record.role,
            plan_hash=record.plan_hash,
            created_iteration=int(record.created_iteration),
            sim_score=float(record.strength),
            real_score=None if real_score is None else float(real_score),
            cluster_id=record.diversity_cluster,
            resource_cost=float(record.resource_cost),
            underdog_gap=float(record.underdog_gap),
            active=bool(record.active),
            retired_reason=record.retired_reason,
            source=record.source,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def write_table_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(_row_json(row), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_table_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(payload)
    return rows


def _row_json(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_json_dict"):
        payload = row.to_json_dict()
    else:
        payload = _jsonable(row)
    if not isinstance(payload, dict):
        raise ValueError("table rows must serialize to JSON objects")
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _wilson_interval(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    z = 1.96
    phat = wins / games
    denominator = 1.0 + z * z / games
    centre = phat + z * z / (2.0 * games)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * games)) / games)
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def _slot_float_rows(values: Mapping[tuple[int, int], tuple[float, float]]) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (f"{int(slot[0])}:{int(slot[1])}", float(bounds[0]), float(bounds[1]))
        for slot, bounds in sorted(values.items())
    )


def _slot_int_rows(values: Mapping[tuple[int, int], int]) -> tuple[tuple[str, int], ...]:
    return tuple((f"{int(slot[0])}:{int(slot[1])}", int(value)) for slot, value in sorted(values.items()))
