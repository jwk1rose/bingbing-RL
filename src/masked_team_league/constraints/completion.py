from __future__ import annotations

import math

from ..domain import Loadout, Observation, Slot, Team


class CompletionMixin:
    """隐藏槽位候选域与可行补全，对应 tex §363-470。"""

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
            # 对应 tex §403-410：隐藏槽位补全是约束满足问题。
            # 这里使用 MRV，优先选择候选域最小的槽位，减少 AllDifferent 回溯。
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
                # 对应 tex §410：每次赋值后传播站位、英雄唯一和唯一装备 ID 约束。
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
