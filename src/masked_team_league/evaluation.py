from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .models import AttackPlan, Team


def clip_probability(value: float, eps: float = 1e-4) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def match_win_probability(probabilities: list[float] | tuple[float, ...], win_required: int) -> float:
    dp = [0.0] * (len(probabilities) + 1)
    dp[0] = 1.0
    for probability in probabilities:
        p = clip_probability(probability)
        ndp = [0.0] * (len(probabilities) + 1)
        for wins in range(len(probabilities)):
            ndp[wins] += dp[wins] * (1.0 - p)
            ndp[wins + 1] += dp[wins] * p
        dp = ndp
    return sum(dp[win_required:])


def team_cost(team: Team) -> float:
    return sum(loadout.cost for loadout in team.slots)


def plan_cost(plan: AttackPlan) -> float:
    return sum(team_cost(team) for team in plan.teams)


def jaccard_similarity(left: AttackPlan, right: AttackPlan) -> float:
    left_heroes = {loadout.hero_id for team in left.teams for loadout in team.slots}
    right_heroes = {loadout.hero_id for team in right.teams for loadout in team.slots}
    union = left_heroes | right_heroes
    if not union:
        return 0.0
    return len(left_heroes & right_heroes) / len(union)


@dataclass(frozen=True)
class RankedItem:
    item: AttackPlan
    value: float
    diversity: float
    score: float


def diversity_select(
    scored_items: list[tuple[AttackPlan, float]],
    *,
    keep: int,
    diversity_weight: float = 0.05,
) -> list[RankedItem]:
    selected: list[RankedItem] = []
    remaining = list(scored_items)
    while remaining and len(selected) < keep:
        best_index = 0
        best_item: RankedItem | None = None
        for index, (plan, value) in enumerate(remaining):
            similarity = max((jaccard_similarity(plan, chosen.item) for chosen in selected), default=0.0)
            diversity = 1.0 - similarity
            score = value + diversity_weight * diversity
            ranked = RankedItem(plan, value, diversity, score)
            if best_item is None or ranked.score > best_item.score:
                best_index = index
                best_item = ranked
        assert best_item is not None
        selected.append(best_item)
        remaining.pop(best_index)
    return selected


def independent_match_loss_probability(probabilities: tuple[float, ...], win_required: int) -> float:
    return 1.0 - match_win_probability(probabilities, win_required)


def all_win_probability(probabilities: tuple[float, ...]) -> float:
    return prod(clip_probability(probability) for probability in probabilities)
