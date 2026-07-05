from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing import Iterable

from .models import AttackPlan, DefensePlan, Loadout, Observation, Slot, Team


LEGAL_DIAGNOSTIC_SCHEMA_VERSION = "legality_diagnostics.v1"


@dataclass(frozen=True)
class LegalDiagnostic:
    code: str
    message: str
    path: tuple[str, ...] = ()
    severity: str = "error"
    details: tuple[tuple[str, str], ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "severity": self.severity,
            "details": {key: value for key, value in self.details},
        }


@dataclass(frozen=True)
class LegalReport:
    legal: bool
    reasons: tuple[str, ...] = ()
    diagnostics: tuple[LegalDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics) or _diagnostics_from_reasons(self.reasons))

    @classmethod
    def ok(cls) -> "LegalReport":
        return cls(True, ())

    @classmethod
    def fail(cls, *reasons: str) -> "LegalReport":
        return cls(False, tuple(reasons))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEGAL_DIAGNOSTIC_SCHEMA_VERSION,
            "legal": self.legal,
            "reasons": list(self.reasons),
            "diagnostics": [diagnostic.to_json_dict() for diagnostic in self.diagnostics],
        }


class ConstraintEngine:
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

    def position_bounds(self, observation: Observation, slot: Slot) -> tuple[float, float]:
        team_idx, slot_idx = slot
        row = observation.slots[team_idx - 1]
        lower = -math.inf
        upper = math.inf
        for idx in range(slot_idx - 2, -1, -1):
            visible = row[idx]
            if not visible.is_hidden and visible.standing_rank is not None:
                lower = visible.standing_rank
                break
        for idx in range(slot_idx, observation.format.team_size):
            visible = row[idx]
            if not visible.is_hidden and visible.standing_rank is not None:
                upper = visible.standing_rank
                break
        return lower, upper

    def build_domains(self, observation: Observation) -> dict[Slot, list[Loadout]]:
        domains: dict[Slot, list[Loadout]] = {}
        for slot in observation.hidden_slots:
            lower, upper = self.position_bounds(observation, slot)
            candidates = []
            for loadout in self.loadout_pool:
                if loadout.hero_id in observation.visible_heroes:
                    continue
                if loadout.unique_equip_id is not None and loadout.unique_equip_id in observation.visible_unique_equip_ids:
                    continue
                if not (lower < loadout.standing_rank < upper):
                    continue
                candidates.append(loadout)
            candidates.sort(key=lambda item: (item.standing_rank, item.hero_id, item.unique_equip_id or -1))
            domains[slot] = candidates
        return domains

    def enumerate_completions(self, observation: Observation, max_k: int = 100) -> list[tuple[Team, ...]]:
        domains = self.build_domains(observation)
        assignments: dict[Slot, Loadout] = {}
        results: list[tuple[Team, ...]] = []
        visible_loadouts = self._visible_loadouts(observation)
        used_heroes = {loadout.hero_id for loadout in visible_loadouts.values()}
        used_equips = {
            loadout.unique_equip_id
            for loadout in visible_loadouts.values()
            if loadout.unique_equip_id is not None
        }

        def backtrack(current_domains: dict[Slot, list[Loadout]]) -> None:
            if len(results) >= max_k:
                return
            if len(assignments) == len(observation.hidden_slots):
                teams = self._materialize_roster(observation, visible_loadouts, assignments)
                if self._check_roster(teams):
                    return
                results.append(teams)
                return
            slot = min(
                (item for item in observation.hidden_slots if item not in assignments),
                key=lambda item: len(current_domains.get(item, ())),
            )
            for loadout in current_domains.get(slot, ()):
                if loadout.hero_id in used_heroes:
                    continue
                if loadout.unique_equip_id is not None and loadout.unique_equip_id in used_equips:
                    continue
                assignments[slot] = loadout
                used_heroes.add(loadout.hero_id)
                if loadout.unique_equip_id is not None:
                    used_equips.add(loadout.unique_equip_id)
                next_domains = self._forward_check(current_domains, slot, loadout, observation)
                if all(next_domains.get(open_slot) for open_slot in observation.hidden_slots if open_slot not in assignments):
                    backtrack(next_domains)
                if loadout.unique_equip_id is not None:
                    used_equips.remove(loadout.unique_equip_id)
                used_heroes.remove(loadout.hero_id)
                assignments.pop(slot)

        if any(not domain for domain in domains.values()):
            return []
        backtrack(domains)
        return results

    def beam_complete(self, observation: Observation, beam_size: int = 32, max_k: int = 100) -> list[tuple[Team, ...]]:
        return self.enumerate_completions(observation, max_k=min(max_k, max(beam_size, 1)))

    def future_feasible(
        self,
        candidate: Loadout,
        *,
        current_team_slots: tuple[Loadout, ...],
        remaining_team_slots_after_candidate: int,
        used_hero_ids: frozenset[int],
        used_unique_equip_ids: frozenset[int],
        pool: tuple[Loadout, ...] | None = None,
    ) -> bool:
        if candidate.hero_id in used_hero_ids:
            return False
        if candidate.unique_equip_id is not None and candidate.unique_equip_id in used_unique_equip_ids:
            return False
        if current_team_slots and candidate.standing_rank <= current_team_slots[-1].standing_rank:
            return False
        if remaining_team_slots_after_candidate <= 0:
            return True
        candidate_pool = self.loadout_pool if pool is None else pool
        return self._future_feasible_with_sorted_pool(
            candidate,
            current_team_slots=current_team_slots,
            remaining_team_slots_after_candidate=remaining_team_slots_after_candidate,
            used_hero_ids=used_hero_ids,
            used_unique_equip_ids=used_unique_equip_ids,
            sorted_pool=self._sorted_candidate_pool(candidate_pool),
        )

    def _future_feasible_with_sorted_pool(
        self,
        candidate: Loadout,
        *,
        current_team_slots: tuple[Loadout, ...],
        remaining_team_slots_after_candidate: int,
        used_hero_ids: frozenset[int],
        used_unique_equip_ids: frozenset[int],
        sorted_pool: tuple[Loadout, ...],
    ) -> bool:
        if candidate.hero_id in used_hero_ids:
            return False
        if candidate.unique_equip_id is not None and candidate.unique_equip_id in used_unique_equip_ids:
            return False
        if current_team_slots and candidate.standing_rank <= current_team_slots[-1].standing_rank:
            return False
        if remaining_team_slots_after_candidate <= 0:
            return True
        heroes = set(used_hero_ids)
        equips = set(used_unique_equip_ids)
        heroes.add(candidate.hero_id)
        if candidate.unique_equip_id is not None:
            equips.add(candidate.unique_equip_id)
        count = 0
        last_rank = candidate.standing_rank
        for loadout in sorted_pool:
            if loadout.standing_rank <= last_rank:
                continue
            if loadout.hero_id in heroes:
                continue
            if loadout.unique_equip_id is not None and loadout.unique_equip_id in equips:
                continue
            heroes.add(loadout.hero_id)
            if loadout.unique_equip_id is not None:
                equips.add(loadout.unique_equip_id)
            last_rank = loadout.standing_rank
            count += 1
            if count >= remaining_team_slots_after_candidate:
                return True
        return False

    def _sorted_candidate_pool(self, candidate_pool: tuple[Loadout, ...]) -> tuple[Loadout, ...]:
        cached = self._sorted_pool_cache.get(candidate_pool)
        if cached is not None:
            return cached
        sorted_pool = tuple(sorted(candidate_pool, key=lambda item: item.standing_rank))
        self._sorted_pool_cache[candidate_pool] = sorted_pool
        return sorted_pool

    def legal_action_mask(
        self,
        candidate_pool: tuple[Loadout, ...],
        *,
        current_team_slots: tuple[Loadout, ...],
        remaining_team_slots_after_candidate: int,
        used_hero_ids: frozenset[int],
        used_unique_equip_ids: frozenset[int],
        use_future_feasibility: bool = True,
    ) -> tuple[bool, ...]:
        if use_future_feasibility:
            sorted_pool = self._sorted_candidate_pool(candidate_pool)
            return tuple(
                self._future_feasible_with_sorted_pool(
                    loadout,
                    current_team_slots=current_team_slots,
                    remaining_team_slots_after_candidate=remaining_team_slots_after_candidate,
                    used_hero_ids=used_hero_ids,
                    used_unique_equip_ids=used_unique_equip_ids,
                    sorted_pool=sorted_pool,
                )
                for loadout in candidate_pool
            )
        return tuple(
            self._immediate_feasible(
                loadout,
                current_team_slots=current_team_slots,
                used_hero_ids=used_hero_ids,
                used_unique_equip_ids=used_unique_equip_ids,
            )
            for loadout in candidate_pool
        )

    def _immediate_feasible(
        self,
        candidate: Loadout,
        *,
        current_team_slots: tuple[Loadout, ...],
        used_hero_ids: frozenset[int],
        used_unique_equip_ids: frozenset[int],
    ) -> bool:
        if candidate.hero_id in used_hero_ids:
            return False
        if candidate.unique_equip_id is not None and candidate.unique_equip_id in used_unique_equip_ids:
            return False
        if current_team_slots and candidate.standing_rank <= current_team_slots[-1].standing_rank:
            return False
        return True

    def _forward_check(
        self,
        domains: dict[Slot, list[Loadout]],
        assigned_slot: Slot,
        assigned_loadout: Loadout,
        observation: Observation,
    ) -> dict[Slot, list[Loadout]]:
        result: dict[Slot, list[Loadout]] = {}
        for slot, domain in domains.items():
            if slot == assigned_slot:
                result[slot] = domain
                continue
            lower, upper = self.position_bounds(observation, slot)
            if slot[0] == assigned_slot[0] and slot[1] > assigned_slot[1]:
                lower = max(lower, assigned_loadout.standing_rank)
            if slot[0] == assigned_slot[0] and slot[1] < assigned_slot[1]:
                upper = min(upper, assigned_loadout.standing_rank)
            result[slot] = [
                loadout
                for loadout in domain
                if loadout.hero_id != assigned_loadout.hero_id
                and (
                    assigned_loadout.unique_equip_id is None
                    or loadout.unique_equip_id != assigned_loadout.unique_equip_id
                )
                and lower < loadout.standing_rank < upper
            ]
        return result

    def _visible_loadouts(self, observation: Observation) -> dict[Slot, Loadout]:
        loadouts: dict[Slot, Loadout] = {}
        for team_idx, row in enumerate(observation.slots, start=1):
            for slot_idx, visible in enumerate(row, start=1):
                if visible.is_hidden:
                    continue
                if visible.loadout is None:
                    raise ValueError("visible slots must carry the full loadout")
                loadouts[(team_idx, slot_idx)] = visible.loadout
        return loadouts

    def _materialize_roster(
        self,
        observation: Observation,
        visible_loadouts: dict[Slot, Loadout],
        assignments: dict[Slot, Loadout],
    ) -> tuple[Team, ...]:
        teams: list[Team] = []
        for team_idx in range(1, observation.format.n_teams + 1):
            slots: list[Loadout] = []
            for slot_idx in range(1, observation.format.team_size + 1):
                slot = (team_idx, slot_idx)
                loadout = visible_loadouts.get(slot) or assignments[slot]
                slots.append(loadout)
            teams.append(Team(tuple(slots)))
        return tuple(teams)


def _diagnostics_from_reasons(reasons: tuple[str, ...]) -> tuple[LegalDiagnostic, ...]:
    return tuple(_diagnostic_from_reason(reason) for reason in reasons)


def _diagnostic_from_reason(reason: str) -> LegalDiagnostic:
    subject = reason
    path: tuple[str, ...] = ()
    if reason.startswith("team ") and ": " in reason:
        team_prefix, subject = reason.split(": ", 1)
        team_token = team_prefix.split()[1]
        path = ("teams", f"team_{team_token}")
    code = _diagnostic_code(subject)
    if code.startswith("MASK_"):
        path = _mask_path(path)
    elif code == "FORMAT_TEAM_COUNT":
        path = ("format", "n_teams")
    return LegalDiagnostic(
        code=code,
        message=reason,
        path=path,
        severity="error",
        details=(("raw_reason", reason),),
    )


def _diagnostic_code(subject: str) -> str:
    if "duplicate hero" in subject:
        return "DUPLICATE_HERO"
    if "duplicate unique equipment" in subject:
        return "DUPLICATE_UNIQUE_EQUIP"
    if "standing_rank" in subject:
        return "STANDING_ORDER"
    if "unique equipment star" in subject:
        return "UNIQUE_EQUIP_STAR"
    if "team count mismatch" in subject:
        return "FORMAT_TEAM_COUNT"
    if "mask exceeds per-team limit" in subject:
        return "MASK_PER_TEAM_LIMIT"
    if "mask entries" in subject:
        return "MASK_VALUE"
    if "mask exceeds global hidden limit" in subject:
        return "MASK_GLOBAL_LIMIT"
    return "LEGALITY_VIOLATION"


def _mask_path(path: tuple[str, ...]) -> tuple[str, ...]:
    if len(path) == 2 and path[0] == "teams":
        return ("mask", path[1])
    return ("mask",)
