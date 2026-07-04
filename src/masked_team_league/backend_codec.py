from __future__ import annotations

from dataclasses import dataclass
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
    attack_teams_proto = [_team_to_proto(team, resources) for team in attack.teams]
    defense_teams_proto = [_team_to_proto(team, resources) for team in defense.teams]
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
