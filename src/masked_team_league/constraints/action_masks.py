from __future__ import annotations

from ..domain import Loadout


class ActionMaskMixin:
    """生成端 legal action mask，对应 tex §677-705。"""

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
        # 对应 tex §702：action mask 要保证选择当前 token 后仍能补完整个 team。
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
