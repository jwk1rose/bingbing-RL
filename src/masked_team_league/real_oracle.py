from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .backend_codec import build_plan_battle_requests, result_to_attack_win_rate
from .cache import MatchupCacheKey, SimulationCache, SimulationResult
from .evaluation import match_win_probability
from .models import AttackPlan, DefensePlan
from .resources import HeroResourceBundle


@dataclass(frozen=True)
class OracleEvaluationRecord:
    attack_id: str
    defense_id: str
    attack_hash: str
    defense_hash: str
    attack_success: float
    round_win_rates: tuple[float, ...]
    oracle_request_ids: tuple[str, ...]
    requests: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]


class OracleBatchEvaluator:
    def __init__(
        self,
        client: Any,
        resources: HeroResourceBundle,
        *,
        cache: SimulationCache | None = None,
        season_buff_ids: int | list[int] | None = None,
        camp_group: int | None = None,
        simulator_version: str = "oracle_backend",
        seed_policy: str = "single_seed_v1",
    ) -> None:
        self.client = client
        self.resources = resources
        self.cache = cache if cache is not None else SimulationCache()
        self.season_buff_ids = season_buff_ids
        self.camp_group = camp_group
        self.simulator_version = simulator_version
        self.seed_policy = seed_policy

    def evaluate_pairs(
        self,
        pairs: Sequence[tuple[str, AttackPlan, str, DefensePlan]],
        *,
        job_prefix: str,
        base_seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[OracleEvaluationRecord]:
        pair_states: list[dict[str, Any]] = []
        pending_by_request_id: dict[str, tuple[int, int, MatchupCacheKey]] = {}
        missing_requests: list[dict[str, Any]] = []
        for pair_index, (attack_id, attack, defense_id, defense) in enumerate(pairs):
            requests = build_plan_battle_requests(
                attack,
                defense,
                self.resources,
                request_prefix=f"{job_prefix}-p{pair_index + 1:06d}",
                base_seed=int(base_seed) + pair_index * max(attack.format.n_teams, 1),
                season_buff_ids=self.season_buff_ids,
                camp_group=self.camp_group,
            )
            state = {
                "attack_id": attack_id,
                "attack": attack,
                "defense_id": defense_id,
                "defense": defense,
                "requests": requests,
                "round_scores": [None] * len(requests),
                "results": [None] * len(requests),
            }
            for round_index, request in enumerate(requests):
                attack_team = attack.teams[round_index]
                defense_team = defense.teams[round_index]
                key = MatchupCacheKey.from_teams(
                    attack_team,
                    defense_team,
                    simulator_version=self.simulator_version,
                    seed_policy=self.seed_policy,
                )
                cached = self.cache.get(key)
                if cached is not None:
                    state["round_scores"][round_index] = cached.win_rate
                    state["results"][round_index] = {
                        "request_id": request["request_id"],
                        "status": "cached",
                        "battle_result": None,
                        "attack_win_rate": cached.win_rate,
                    }
                    continue
                missing_requests.append(request)
                pending_by_request_id[str(request["request_id"])] = (pair_index, round_index, key)
            pair_states.append(state)

        if missing_requests:
            submitted = self.client.submit_and_wait(
                missing_requests,
                metadata={
                    "kind": "masked_team_league_round",
                    "job_prefix": job_prefix,
                    "pairs": len(pairs),
                    "requests": len(missing_requests),
                    **(metadata or {}),
                },
            )
            job_id = str(submitted.get("job_id") or "")
            results = self.client.read_results(job_id)
            for result in results:
                request_id = str(result.get("request_id") or "")
                pending = pending_by_request_id.get(request_id)
                if pending is None:
                    continue
                pair_index, round_index, key = pending
                win_rate = result_to_attack_win_rate(result)
                self.cache.put(key, _simulation_result_from_win_rate(win_rate))
                copied = dict(result)
                copied["attack_win_rate"] = win_rate
                pair_states[pair_index]["round_scores"][round_index] = win_rate
                pair_states[pair_index]["results"][round_index] = copied

        records: list[OracleEvaluationRecord] = []
        for state in pair_states:
            round_scores = state["round_scores"]
            if any(score is None for score in round_scores):
                missing = [
                    request["request_id"]
                    for score, request in zip(round_scores, state["requests"])
                    if score is None
                ]
                raise ValueError(f"missing oracle results for request ids: {missing}")
            attack = state["attack"]
            defense = state["defense"]
            scores = tuple(float(score) for score in round_scores)
            records.append(
                OracleEvaluationRecord(
                    attack_id=str(state["attack_id"]),
                    defense_id=str(state["defense_id"]),
                    attack_hash=attack.hash(),
                    defense_hash=defense.hash(),
                    attack_success=match_win_probability(scores, attack.format.win_required),
                    round_win_rates=scores,
                    oracle_request_ids=tuple(str(request["request_id"]) for request in state["requests"]),
                    requests=tuple(state["requests"]),
                    results=tuple(result for result in state["results"] if result is not None),
                )
            )
        return records


def _simulation_result_from_win_rate(win_rate: float) -> SimulationResult:
    games = 1000
    return SimulationResult(wins=int(round(float(win_rate) * games)), games=games, mean_margin=0.0, mean_duration=0.0)
