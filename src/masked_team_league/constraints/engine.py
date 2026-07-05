from __future__ import annotations

from typing import Iterable

from ..domain import AttackPlan, DefensePlan, Loadout, Team
from .action_masks import ActionMaskMixin
from .completion import CompletionMixin
from .diagnostics import LegalReport


class ConstraintEngine(ActionMaskMixin, CompletionMixin):
    def __init__(self, loadout_pool: Iterable[Loadout] = ()) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self._sorted_loadout_pool = tuple(sorted(self.loadout_pool, key=lambda item: item.standing_rank))
        self._sorted_pool_cache: dict[tuple[Loadout, ...], tuple[Loadout, ...]] = {
            self.loadout_pool: self._sorted_loadout_pool
        }

    def check_team(self, team: Team) -> LegalReport:
        reasons: list[str] = []
        hero_ids = [loadout.hero_id for loadout in team.slots]
        if len(set(hero_ids)) != len(hero_ids):
            reasons.append("duplicate hero in team")
        equip_ids = [loadout.unique_equip_id for loadout in team.slots if loadout.unique_equip_id is not None]
        if len(set(equip_ids)) != len(equip_ids):
            reasons.append("duplicate unique equipment in team")
        ranks = [loadout.standing_rank for loadout in team.slots]
        if any(left >= right for left, right in zip(ranks, ranks[1:])):
            reasons.append("standing_rank must be strictly increasing")
        for loadout in team.slots:
            if loadout.unique_equip_id is not None and loadout.unique_equip_star not in (3, 4, 5):
                reasons.append("unique equipment star must be 3, 4, or 5")
        return LegalReport(not reasons, tuple(reasons))

    def check_attack(self, plan: AttackPlan) -> LegalReport:
        reasons = self._check_roster(plan.teams)
        if len(plan.teams) != plan.format.n_teams:
            reasons.append("attack team count mismatch")
        return LegalReport(not reasons, tuple(reasons))

    def check_defense(self, plan: DefensePlan) -> LegalReport:
        reasons = self._check_roster(plan.teams)
        if len(plan.teams) != plan.format.n_teams:
            reasons.append("defense team count mismatch")
        reasons.extend(self._check_mask(plan))
        return LegalReport(not reasons, tuple(reasons))

    def is_legal_attack(self, plan: AttackPlan) -> bool:
        return self.check_attack(plan).legal

    def is_legal_defense(self, plan: DefensePlan) -> bool:
        return self.check_defense(plan).legal

    def _check_roster(self, teams: tuple[Team, ...]) -> list[str]:
        reasons: list[str] = []
        all_heroes: list[int] = []
        all_equips: list[int] = []
        for index, team in enumerate(teams, start=1):
            team_report = self.check_team(team)
            reasons.extend(f"team {index}: {reason}" for reason in team_report.reasons)
            all_heroes.extend(loadout.hero_id for loadout in team.slots)
            all_equips.extend(loadout.unique_equip_id for loadout in team.slots if loadout.unique_equip_id is not None)
        if len(set(all_heroes)) != len(all_heroes):
            reasons.append("duplicate hero across roster")
        if len(set(all_equips)) != len(all_equips):
            reasons.append("duplicate unique equipment across roster")
        return reasons

    def _check_mask(self, plan: DefensePlan) -> list[str]:
        reasons: list[str] = []
        hidden_total = 0
        for team_idx, mask_row in enumerate(plan.mask, start=1):
            row_hidden = sum(1 for value in mask_row if value)
            hidden_total += row_hidden
            if row_hidden > plan.format.max_hidden_per_team:
                reasons.append(f"team {team_idx}: mask exceeds per-team limit")
            if any(value not in (0, 1) for value in mask_row):
                reasons.append(f"team {team_idx}: mask entries must be 0 or 1")
        if hidden_total > plan.format.max_hidden_total:
            reasons.append("mask exceeds global hidden limit")
        return reasons
