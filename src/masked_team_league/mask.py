from __future__ import annotations

from itertools import combinations, product

from .constraints import ConstraintEngine
from .models import DefensePlan, MatchFormat, Team, observe_defense


def legal_team_mask_patterns(team_size: int = 5, max_hidden: int = 2) -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = []
    for count in range(max_hidden + 1):
        for hidden_indices in combinations(range(team_size), count):
            row = [0] * team_size
            for index in hidden_indices:
                row[index] = 1
            patterns.append(tuple(row))
    return tuple(patterns)


def enumerate_legal_masks(match_format: MatchFormat, *, limit: int | None = None) -> list[tuple[tuple[int, ...], ...]]:
    patterns = legal_team_mask_patterns(match_format.team_size, match_format.max_hidden_per_team)
    masks: list[tuple[tuple[int, ...], ...]] = []
    for rows in product(patterns, repeat=match_format.n_teams):
        if sum(sum(row) for row in rows) > match_format.max_hidden_total:
            continue
        masks.append(tuple(rows))
        if limit is not None and len(masks) >= limit:
            break
    return masks


class MaskSearcher:
    def __init__(self, constraint_engine: ConstraintEngine) -> None:
        self.constraint_engine = constraint_engine

    def search(
        self,
        match_format: MatchFormat,
        roster: tuple[Team, ...],
        *,
        keep: int = 8,
        max_masks: int | None = None,
    ) -> list[tuple[tuple[tuple[int, ...], ...], float, dict[str, float]]]:
        scored: list[tuple[tuple[tuple[int, ...], ...], float, dict[str, float]]] = []
        for mask in enumerate_legal_masks(match_format, limit=max_masks):
            plan = DefensePlan(format=match_format, teams=roster, mask=mask, source="mask_search")
            if not self.constraint_engine.is_legal_defense(plan):
                continue
            observation = observe_defense(plan)
            domains = self.constraint_engine.build_domains(observation)
            domain_product_log = sum(_log_domain(len(domain)) for domain in domains.values())
            hidden_count = sum(sum(row) for row in mask)
            leakage_penalty = sum(1.0 for domain in domains.values() if len(domain) <= 1)
            score = hidden_count + 0.20 * domain_product_log - 0.50 * leakage_penalty
            scored.append(
                (
                    mask,
                    score,
                    {
                        "hidden_count": float(hidden_count),
                        "domain_log": float(domain_product_log),
                        "leakage_penalty": float(leakage_penalty),
                    },
                )
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:keep]


def _log_domain(value: int) -> float:
    if value <= 0:
        return 0.0
    import math

    return math.log(value + 1.0)
