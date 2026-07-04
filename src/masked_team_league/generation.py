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
    def __init__(self, loadout_pool: tuple[Loadout, ...], *, seed: int = 0) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.rng = random.Random(seed)
        self.engine = ConstraintEngine(self.loadout_pool)

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
            failed = False
            for _team_idx in range(match_format.n_teams):
                slots: list[Loadout] = []
                for slot_idx in range(match_format.team_size):
                    remaining_after = match_format.team_size - slot_idx - 1
                    pool = list(self.loadout_pool)
                    self.rng.shuffle(pool)
                    legal = [
                        loadout
                        for loadout in pool
                        if self.engine.future_feasible(
                            loadout,
                            current_team_slots=tuple(slots),
                            remaining_team_slots_after_candidate=remaining_after,
                            used_hero_ids=frozenset(used_heroes),
                            used_unique_equip_ids=frozenset(used_equips),
                            pool=self.loadout_pool,
                        )
                    ]
                    if reference_cost is not None and goal.target_power_ratio < 1.0:
                        budget = goal.target_power_ratio * reference_cost
                        legal = [loadout for loadout in legal if total_cost + loadout.cost <= budget]
                    if not legal:
                        failed = True
                        break
                    chosen = legal[0]
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
