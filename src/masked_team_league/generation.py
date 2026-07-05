from __future__ import annotations

from dataclasses import dataclass
import random

from .constraints import ConstraintEngine
from .models import AttackPlan, DefensePlan, Loadout, MatchFormat, Team


@dataclass(frozen=True)
class GenerationGoal:
    target_power_ratio: float = 1.0
    explore_beta: float = 0.0
    diversity_weight: float = 0.0


class LegalPlanGenerator:
    def __init__(self, loadout_pool: tuple[Loadout, ...], *, seed: int = 0, use_future_feasibility: bool = True) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.rng = random.Random(seed)
        self.engine = ConstraintEngine(self.loadout_pool)
        self.use_future_feasibility = bool(use_future_feasibility)
        self._cost_sorted_loadout_pool = tuple(
            sorted(
                self.loadout_pool,
                key=lambda item: (item.cost, item.standing_rank, item.hero_id, item.unique_equip_id or -1),
            )
        )

    def generate_attack_plan(
        self,
        match_format: MatchFormat,
        *,
        source: str = "legal_random",
        goal: GenerationGoal | None = None,
        reference_cost: float | None = None,
        max_attempts: int = 256,
    ) -> AttackPlan:
        teams = self._generate_teams(
            match_format,
            goal=goal or GenerationGoal(),
            reference_cost=reference_cost,
            max_attempts=max_attempts,
        )
        return AttackPlan(format=match_format, teams=teams, source=source)

    def generate_defense_plan(
        self,
        match_format: MatchFormat,
        *,
        mask: tuple[tuple[int, ...], ...] | None = None,
        source: str = "legal_random",
        goal: GenerationGoal | None = None,
        reference_cost: float | None = None,
        max_attempts: int = 256,
    ) -> DefensePlan:
        teams = self._generate_teams(
            match_format,
            goal=goal or GenerationGoal(),
            reference_cost=reference_cost,
            max_attempts=max_attempts,
        )
        if mask is None:
            mask = tuple((0, 0, 0, 0, 0) for _ in range(match_format.n_teams))
        return DefensePlan(format=match_format, teams=teams, mask=mask, source=source)

    def generate_attack_candidates(
        self,
        match_format: MatchFormat,
        *,
        count: int,
        goal: GenerationGoal | None = None,
        reference_cost: float | None = None,
    ) -> list[AttackPlan]:
        candidates: list[AttackPlan] = []
        seen: set[str] = set()
        attempts = 0
        while len(candidates) < count and attempts < count * 50:
            attempts += 1
            try:
                candidate = self.generate_attack_plan(
                    match_format,
                    goal=goal,
                    reference_cost=reference_cost,
                    max_attempts=64,
                )
            except ValueError:
                continue
            digest = candidate.hash()
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append(candidate)
        return candidates

    def _generate_teams(
        self,
        match_format: MatchFormat,
        *,
        goal: GenerationGoal,
        reference_cost: float | None,
        max_attempts: int,
    ) -> tuple[Team, ...]:
        for _ in range(max_attempts):
            used_heroes: set[int] = set()
            used_equips: set[int] = set()
            teams: list[Team] = []
            total_cost = 0.0
            total_slots = match_format.n_teams * match_format.team_size
            failed = False
            for _team_idx in range(match_format.n_teams):
                slots: list[Loadout] = []
                for slot_idx in range(match_format.team_size):
                    remaining_after = match_format.team_size - slot_idx - 1
                    remaining_total_after = total_slots - (len(teams) * match_format.team_size + slot_idx + 1)
                    mask = self.engine.legal_action_mask(
                        self.loadout_pool,
                        current_team_slots=tuple(slots),
                        remaining_team_slots_after_candidate=remaining_after,
                        used_hero_ids=frozenset(used_heroes),
                        used_unique_equip_ids=frozenset(used_equips),
                        use_future_feasibility=self.use_future_feasibility,
                    )
                    legal = [loadout for loadout, allowed in zip(self.loadout_pool, mask) if allowed]
                    if reference_cost is not None and goal.target_power_ratio < 1.0:
                        budget = goal.target_power_ratio * reference_cost
                        legal = [
                            loadout
                            for loadout in legal
                            if self._budget_future_feasible(
                                loadout,
                                total_cost=total_cost,
                                budget=budget,
                                remaining_total_slots_after_candidate=remaining_total_after,
                                used_hero_ids=used_heroes,
                                used_unique_equip_ids=used_equips,
                            )
                        ]
                        legal = self._budget_choice_pool(
                            legal,
                            total_cost=total_cost,
                            budget=budget,
                            remaining_total_slots_after_candidate=remaining_total_after,
                        )
                    if not legal:
                        failed = True
                        break
                    chosen = self.rng.choice(legal)
                    slots.append(chosen)
                    used_heroes.add(chosen.hero_id)
                    if chosen.unique_equip_id is not None:
                        used_equips.add(chosen.unique_equip_id)
                    total_cost += chosen.cost
                if failed:
                    break
                teams.append(Team(tuple(slots)))
            if failed:
                continue
            result = tuple(teams)
            if not self.engine._check_roster(result):
                return result
        raise ValueError("could not generate a legal plan from the provided loadout pool")

    def _budget_future_feasible(
        self,
        candidate: Loadout,
        *,
        total_cost: float,
        budget: float,
        remaining_total_slots_after_candidate: int,
        used_hero_ids: set[int],
        used_unique_equip_ids: set[int],
    ) -> bool:
        new_total = total_cost + candidate.cost
        if new_total > budget:
            return False
        if remaining_total_slots_after_candidate <= 0:
            return True
        heroes = set(used_hero_ids)
        equips = set(used_unique_equip_ids)
        heroes.add(candidate.hero_id)
        if candidate.unique_equip_id is not None:
            equips.add(candidate.unique_equip_id)
        cheapest_remaining = 0.0
        count = 0
        for loadout in self._cost_sorted_loadout_pool:
            if loadout.hero_id in heroes:
                continue
            if loadout.unique_equip_id is not None and loadout.unique_equip_id in equips:
                continue
            heroes.add(loadout.hero_id)
            if loadout.unique_equip_id is not None:
                equips.add(loadout.unique_equip_id)
            cheapest_remaining += loadout.cost
            count += 1
            if count >= remaining_total_slots_after_candidate:
                return new_total + cheapest_remaining <= budget
        return False

    def _budget_choice_pool(
        self,
        legal: list[Loadout],
        *,
        total_cost: float,
        budget: float,
        remaining_total_slots_after_candidate: int,
    ) -> list[Loadout]:
        if not legal:
            return legal
        remaining_slots_including_candidate = remaining_total_slots_after_candidate + 1
        if remaining_slots_including_candidate <= 1:
            return legal
        target_per_slot = max((budget - total_cost) / remaining_slots_including_candidate, 0.0)
        ceiling = target_per_slot * 1.35
        affordable = [loadout for loadout in legal if loadout.cost <= ceiling]
        ordered = sorted(
            affordable or legal,
            key=lambda item: (item.cost, item.standing_rank, item.hero_id, item.unique_equip_id or -1),
        )
        return ordered[: max(1, min(len(ordered), max(4, len(ordered) // 3)))]
