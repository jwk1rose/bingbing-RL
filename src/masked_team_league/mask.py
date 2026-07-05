from __future__ import annotations

from itertools import combinations, product
from typing import Any, Callable, Sequence

from .constraints import ConstraintEngine
from .models import DefensePlan, MatchFormat, Team, observe_defense

MaskSlotScoreProvider = Callable[[tuple[Team, ...], MatchFormat], Sequence[Sequence[float]]]


def legal_team_mask_patterns(team_size: int = 5, max_hidden: int = 2) -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = []
    for count in range(max_hidden, -1, -1):
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
    def __init__(
        self,
        constraint_engine: ConstraintEngine,
        *,
        slot_score_provider: MaskSlotScoreProvider | None = None,
        learned_score_weight: float = 0.05,
    ) -> None:
        self.constraint_engine = constraint_engine
        self.slot_score_provider = slot_score_provider
        self.learned_score_weight = float(learned_score_weight)

    def search(
        self,
        match_format: MatchFormat,
        roster: tuple[Team, ...],
        *,
        keep: int = 8,
        max_masks: int | None = None,
    ) -> list[tuple[tuple[tuple[int, ...], ...], float, dict[str, Any]]]:
        scored: list[tuple[tuple[tuple[int, ...], ...], float, dict[str, Any]]] = []
        learned_scores = _normalize_slot_scores(
            self.slot_score_provider(roster, match_format) if self.slot_score_provider is not None else None,
            match_format,
        )
        for mask in enumerate_legal_masks(match_format, limit=max_masks):
            plan = DefensePlan(format=match_format, teams=roster, mask=mask, source="mask_search")
            if not self.constraint_engine.is_legal_defense(plan):
                continue
            observation = observe_defense(plan)
            domains = self.constraint_engine.build_domains(observation)
            domain_product_log = sum(_log_domain(len(domain)) for domain in domains.values())
            hidden_count = sum(sum(row) for row in mask)
            leakage_penalty = sum(1.0 for domain in domains.values() if len(domain) <= 1)
            learned_mask_score = _mask_score(mask, learned_scores)
            score = hidden_count + 0.20 * domain_product_log - 0.50 * leakage_penalty
            if learned_scores is not None:
                score += self.learned_score_weight * learned_mask_score
            scored.append(
                (
                    mask,
                    score,
                    {
                        "hidden_count": float(hidden_count),
                        "domain_log": float(domain_product_log),
                        "leakage_penalty": float(leakage_penalty),
                        "learned_mask_score": float(learned_mask_score),
                        "learned_score_weight": self.learned_score_weight,
                        "learned_slot_scores": _slot_scores_for_output(learned_scores, match_format),
                        "hidden_slot_explanations": _hidden_slot_explanations(mask, roster, learned_scores),
                        "top_learned_slots": _top_learned_slots(roster, learned_scores, limit=match_format.max_hidden_total),
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


def _normalize_slot_scores(
    scores: Sequence[Sequence[float]] | None,
    match_format: MatchFormat,
) -> tuple[tuple[float, ...], ...] | None:
    if scores is None:
        return None
    rows = tuple(tuple(float(value) for value in row) for row in scores)
    if len(rows) != match_format.n_teams or any(len(row) != match_format.team_size for row in rows):
        raise ValueError("slot_score_provider must return [n_teams][team_size] scores")
    return rows


def _mask_score(mask: tuple[tuple[int, ...], ...], scores: tuple[tuple[float, ...], ...] | None) -> float:
    if scores is None:
        return 0.0
    return sum(float(value) * float(hidden) for row, score_row in zip(mask, scores) for hidden, value in zip(row, score_row))


def _slot_scores_for_output(
    scores: tuple[tuple[float, ...], ...] | None,
    match_format: MatchFormat,
) -> tuple[tuple[float, ...], ...]:
    if scores is not None:
        return scores
    return tuple(tuple(0.0 for _slot in range(match_format.team_size)) for _team in range(match_format.n_teams))


def _hidden_slot_explanations(
    mask: tuple[tuple[int, ...], ...],
    roster: tuple[Team, ...],
    scores: tuple[tuple[float, ...], ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_idx, (mask_row, team) in enumerate(zip(mask, roster)):
        for slot_idx, hidden in enumerate(mask_row):
            if not hidden:
                continue
            loadout = team.slots[slot_idx]
            rows.append(
                {
                    "team_index": team_idx,
                    "slot_index": slot_idx,
                    "hidden": True,
                    "hero_id": loadout.hero_id,
                    "unique_equip_id": loadout.unique_equip_id,
                    "unique_equip_star": loadout.unique_equip_star,
                    "standing_rank": loadout.standing_rank,
                    "standing_bucket": loadout.standing_bucket,
                    "final_power": loadout.final_power,
                    "learned_slot_score": _slot_score(scores, team_idx, slot_idx),
                }
            )
    rows.sort(key=lambda row: (float(row["learned_slot_score"]), -int(row["team_index"]), -int(row["slot_index"])), reverse=True)
    return rows


def _top_learned_slots(
    roster: tuple[Team, ...],
    scores: tuple[tuple[float, ...], ...] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_idx, team in enumerate(roster):
        for slot_idx, loadout in enumerate(team.slots):
            rows.append(
                {
                    "team_index": team_idx,
                    "slot_index": slot_idx,
                    "hero_id": loadout.hero_id,
                    "unique_equip_id": loadout.unique_equip_id,
                    "unique_equip_star": loadout.unique_equip_star,
                    "learned_slot_score": _slot_score(scores, team_idx, slot_idx),
                }
            )
    rows.sort(key=lambda row: (float(row["learned_slot_score"]), -int(row["team_index"]), -int(row["slot_index"])), reverse=True)
    return rows[: max(int(limit), 0)]


def _slot_score(scores: tuple[tuple[float, ...], ...] | None, team_idx: int, slot_idx: int) -> float:
    if scores is None:
        return 0.0
    return float(scores[team_idx][slot_idx])
