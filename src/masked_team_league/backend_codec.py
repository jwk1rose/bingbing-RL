from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .evaluation import match_win_probability
from .models import AttackPlan, DefensePlan, Loadout, Team
from .resources import HeroResourceBundle


@dataclass(frozen=True)
class PlanBattleScore:
    attack_match_win_rate: float
    round_win_rates: tuple[float, ...]
    requests: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]


def build_plan_battle_requests(
    attack: AttackPlan,
    defense: DefensePlan,
    resources: HeroResourceBundle,
    *,
    request_prefix: str,
    base_seed: int,
    season_buff_ids: int | list[int] | None = None,
    camp_group: int | None = None,
) -> list[dict[str, Any]]:
    if attack.format != defense.format:
        raise ValueError("attack and defense formats must match")
    attack_teams_proto = _side_to_proto(attack.teams, resources, seed=int(base_seed) + 17)
    defense_teams_proto = _side_to_proto(defense.teams, resources, seed=int(base_seed) + 29)
    requests: list[dict[str, Any]] = []
    for round_index in range(attack.format.n_teams):
        round_id = round_index + 1
        requests.append(
            {
                "request_id": f"{request_prefix}-r{round_id}",
                "seed": int(base_seed) + round_index,
                "round": round_id,
                "battleIdx": round_id,
                "mode": "peakArena",
                "entry": "enterRelay2",
                "seasonBuffIds": season_buff_ids,
                "peakArenaCampGroup": camp_group,
                "self_heroes_proto": attack_teams_proto[round_index],
                "oppo_heroes_proto": defense_teams_proto[round_index],
                "self_teams_proto": attack_teams_proto,
                "oppo_teams_proto": defense_teams_proto,
                "extra": {"shareFight": False},
            }
        )
    return requests


def score_plan_battle_results(
    attack: AttackPlan,
    requests: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> PlanBattleScore:
    result_by_id = {str(result.get("request_id") or ""): result for result in results}
    round_scores: list[float] = []
    ordered_results: list[dict[str, Any]] = []
    for request in requests:
        result = result_by_id.get(str(request.get("request_id") or ""))
        if result is None:
            raise ValueError(f"missing oracle result for request_id={request.get('request_id')}")
        round_scores.append(result_to_attack_win_rate(result))
        ordered_results.append(result)
    return PlanBattleScore(
        attack_match_win_rate=match_win_probability(round_scores, attack.format.win_required),
        round_win_rates=tuple(round_scores),
        requests=tuple(requests),
        results=tuple(ordered_results),
    )


def result_to_attack_win_rate(result: dict[str, Any]) -> float:
    if result.get("status") == "error":
        raise ValueError(f"oracle result is error: {result.get('error')}")
    try:
        value = int(result.get("battle_result"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported battle_result: {result.get('battle_result')!r}") from exc
    if value == 0:
        return 1.0
    if value == 1:
        return 0.0
    if value == 3:
        soft = _mixed_result_soft_label(result)
        if soft is not None:
            return soft
        return 0.5
    raise ValueError(f"unsupported battle_result: {value}")


def _team_to_proto(team: Team, resources: HeroResourceBundle) -> list[dict[str, Any]]:
    return [resources.to_hero_proto(loadout, instance_id=index + 1) for index, loadout in enumerate(team.slots)]


def _side_to_proto(teams: tuple[Team, ...], resources: HeroResourceBundle, *, seed: int) -> list[list[dict[str, Any]]]:
    assignments = _legend_assignments(resources, sum(len(team.slots) for team in teams), seed=seed)
    result: list[list[dict[str, Any]]] = []
    flat_index = 0
    for team in teams:
        proto_team: list[dict[str, Any]] = []
        for slot_index, loadout in enumerate(team.slots, start=1):
            equip_id, equip_star = assignments[flat_index] if assignments else (None, 5)
            proto_team.append(
                resources.to_hero_proto(
                    loadout,
                    instance_id=slot_index,
                    legend_equip_id=equip_id,
                    legend_equip_star=equip_star,
                    astrolabe_seed=seed + flat_index * 9973,
                )
            )
            flat_index += 1
        result.append(proto_team)
    return result


def _legend_assignments(resources: HeroResourceBundle, slot_count: int, *, seed: int) -> list[tuple[int | None, int]]:
    rules = resources.runtime_rules
    if rules is None or not rules.normal_legend_equip_ids:
        return []
    unique_ids = list(rules.unique_legend_equip_ids)
    normal_ids = list(rules.normal_legend_equip_ids)
    if len(unique_ids) > slot_count:
        raise ValueError(f"cannot assign {len(unique_ids)} unique legend equips to {slot_count} slots")
    rng = random.Random(seed)
    assignments: list[tuple[int | None, int]] = [(None, 5) for _ in range(slot_count)]
    unique_slots = rng.sample(range(slot_count), len(unique_ids))
    rng.shuffle(unique_ids)
    for slot, equip_id in zip(unique_slots, unique_ids):
        assignments[slot] = (int(equip_id), 5)
    for index, (equip_id, _star) in enumerate(assignments):
        if equip_id is None:
            assignments[index] = (int(rng.choice(normal_ids)), 5)
    return assignments


def _mixed_result_soft_label(result: dict[str, Any]) -> float | None:
    units = result.get("units")
    if not isinstance(units, list) or not units:
        return None
    attack_alive, attack_hp = _side_unit_stats(units, 1)
    defense_alive, defense_hp = _side_unit_stats(units, -1)
    if attack_alive == 0 and defense_alive == 0 and attack_hp <= 0.0 and defense_hp <= 0.0:
        return 0.5
    alive_score = _ratio_or_half(float(attack_alive), float(attack_alive + defense_alive))
    hp_score = _ratio_or_half(attack_hp, attack_hp + defense_hp)
    return min(1.0, max(0.0, 0.5 * alive_score + 0.5 * hp_score))


def _side_unit_stats(units: list[Any], side: int) -> tuple[int, float]:
    alive_count = 0
    hp_sum = 0.0
    for unit in units:
        if not isinstance(unit, dict):
            continue
        if int(unit.get("side") or unit.get("_side") or 0) != side:
            continue
        hp = float(unit.get("hp") or unit.get("_hp") or 0.0)
        hp_sum += max(0.0, hp)
        if bool(unit.get("alive")) or hp > 0.0:
            alive_count += 1
    return alive_count, hp_sum


def _ratio_or_half(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.5
    return min(1.0, max(0.0, numerator / denominator))
