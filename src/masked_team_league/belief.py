from __future__ import annotations

from dataclasses import dataclass
import math

from .constraints import ConstraintEngine
from .models import Observation, Team


@dataclass(frozen=True)
class BeliefOutput:
    candidates: tuple[tuple[Team, ...], ...]
    weights: tuple[float, ...]
    entropy: float
    feasible_count_estimate: int
    top1_top2_gap: float
    domain_stats: tuple[tuple[str, float], ...]


class BeliefEngine:
    def __init__(self, constraint_engine: ConstraintEngine) -> None:
        self.constraint_engine = constraint_engine

    def build(self, observation: Observation, *, max_k: int = 64) -> BeliefOutput:
        domains = self.constraint_engine.build_domains(observation)
        completions = self.constraint_engine.enumerate_completions(observation, max_k=max_k)
        if not completions:
            return BeliefOutput((), (), 0.0, 0, 0.0, tuple((f"domain_{slot}", float(len(domain))) for slot, domain in domains.items()))
        scores = [_roster_strength(roster) for roster in completions]
        weights = _softmax(scores)
        entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
        ordered = sorted(zip(completions, weights), key=lambda pair: pair[1], reverse=True)
        top_weights = [weight for _candidate, weight in ordered[:2]]
        top1_top2_gap = top_weights[0] - top_weights[1] if len(top_weights) > 1 else top_weights[0]
        domain_stats = tuple((f"{slot[0]}:{slot[1]}", float(len(domain))) for slot, domain in domains.items())
        return BeliefOutput(
            candidates=tuple(candidate for candidate, _weight in ordered),
            weights=tuple(weight for _candidate, weight in ordered),
            entropy=entropy,
            feasible_count_estimate=len(completions),
            top1_top2_gap=top1_top2_gap,
            domain_stats=domain_stats,
        )


def _roster_strength(roster: tuple[Team, ...]) -> float:
    return sum(team.total_power + 10.0 * sum(loadout.unique_equip_star or 0 for loadout in team.slots) for team in roster)


def _softmax(scores: list[float]) -> tuple[float, ...]:
    if not scores:
        return ()
    center = max(scores)
    values = [math.exp((score - center) / 500.0) for score in scores]
    total = sum(values)
    return tuple(value / total for value in values)
