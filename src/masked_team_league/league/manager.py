from __future__ import annotations

from dataclasses import dataclass, replace

from ..domain import AttackPlan, DefensePlan, canonical_hash


@dataclass(frozen=True)
class StrategyRecord:
    strategy_id: str
    role: str
    side: str
    plan_hash: str
    source: str
    created_iteration: int
    strength: float
    diversity_cluster: str = "default"
    resource_cost: float = 0.0
    underdog_gap: float = 0.0
    active: bool = True
    retired_reason: str | None = None


@dataclass(frozen=True)
class PayoffEntry:
    attack_id: str
    defense_id: str
    attack_success: float
    games: int


class LeagueManager:
    def __init__(self) -> None:
        self.attack_pool: dict[str, tuple[AttackPlan, StrategyRecord]] = {}
        self.defense_pool: dict[str, tuple[DefensePlan, StrategyRecord]] = {}
        self.payoffs: dict[tuple[str, str], PayoffEntry] = {}
        self.iteration = 0

    def add_attack(
        self,
        plan: AttackPlan,
        *,
        role: str,
        source: str,
        strength: float,
        diversity_cluster: str | None = None,
        resource_cost: float | None = None,
        underdog_gap: float = 0.0,
    ) -> StrategyRecord:
        strategy_id = f"atk-{len(self.attack_pool) + 1:06d}"
        record = StrategyRecord(
            strategy_id,
            role,
            "attack",
            plan.hash(),
            source,
            self.iteration,
            strength,
            diversity_cluster or _plan_diversity_cluster(plan),
            _plan_resource_cost(plan) if resource_cost is None else float(resource_cost),
            float(underdog_gap),
        )
        self.attack_pool[strategy_id] = (plan, record)
        return record

    def add_defense(
        self,
        plan: DefensePlan,
        *,
        role: str,
        source: str,
        strength: float,
        diversity_cluster: str | None = None,
        resource_cost: float | None = None,
        underdog_gap: float = 0.0,
    ) -> StrategyRecord:
        strategy_id = f"def-{len(self.defense_pool) + 1:06d}"
        record = StrategyRecord(
            strategy_id,
            role,
            "defense",
            plan.hash(),
            source,
            self.iteration,
            strength,
            diversity_cluster or _plan_diversity_cluster(plan),
            _plan_resource_cost(plan) if resource_cost is None else float(resource_cost),
            float(underdog_gap),
        )
        self.defense_pool[strategy_id] = (plan, record)
        return record

    def record_payoff(self, attack_id: str, defense_id: str, *, attack_success: float, games: int) -> None:
        self.payoffs[(attack_id, defense_id)] = PayoffEntry(attack_id, defense_id, attack_success, games)

    def meta_distribution(self, side: str) -> tuple[tuple[str, float], ...]:
        pool = self._active_pool(side)
        if not pool:
            return ()
        strengths = [(strategy_id, max(record.strength, 1e-6)) for strategy_id, (_plan, record) in pool.items()]
        total = sum(strength for _strategy_id, strength in strengths)
        return tuple((strategy_id, strength / total) for strategy_id, strength in strengths)

    def mixed_meta_distribution(self, side: str) -> tuple[tuple[str, float], ...]:
        pool = self._active_pool(side)
        if not pool:
            return ()
        all_ids = tuple(pool.keys())
        components: list[tuple[float, tuple[tuple[str, float], ...]]] = [
            (0.4, self.meta_distribution(side)),
            (0.3, self._uniform_distribution(self._historical_ids(pool) or all_ids)),
            (0.2, self._uniform_distribution(self._diverse_ids(pool))),
            (0.1, self._uniform_distribution(all_ids)),
        ]
        weights: dict[str, float] = {strategy_id: 0.0 for strategy_id in all_ids}
        for component_weight, distribution in components:
            for strategy_id, weight in distribution:
                weights[strategy_id] = weights.get(strategy_id, 0.0) + component_weight * weight
        total = sum(weights.values())
        if total <= 0.0:
            return self._uniform_distribution(all_ids)
        return tuple((strategy_id, weights[strategy_id] / total) for strategy_id in all_ids)

    def mixed_meta_plans(self, side: str, *, limit: int | None = None) -> tuple[tuple[AttackPlan | DefensePlan, float], ...]:
        pool = self._active_pool(side)
        distribution = sorted(self.mixed_meta_distribution(side), key=lambda item: item[1], reverse=True)
        if limit is not None:
            distribution = distribution[:limit]
        total = sum(weight for _strategy_id, weight in distribution)
        if total <= 0.0:
            return ()
        return tuple((pool[strategy_id][0], weight / total) for strategy_id, weight in distribution)

    def strongest_plans(
        self,
        side: str,
        *,
        limit: int,
        active_only: bool = True,
    ) -> tuple[tuple[AttackPlan | DefensePlan, float], ...]:
        pool = self._active_pool(side) if active_only else self._pool(side)
        ordered = sorted(pool.values(), key=lambda item: item[1].strength, reverse=True)
        return tuple((plan, record.strength) for plan, record in ordered[:limit])

    def hardest_defense_plans(
        self,
        *,
        limit: int,
        active_only: bool = True,
    ) -> tuple[tuple[DefensePlan, float], ...]:
        pool = self._active_pool("defense") if active_only else self.defense_pool
        ordered = sorted(pool.values(), key=lambda item: item[1].strength)
        return tuple((plan, 1.0 - record.strength) for plan, record in ordered[:limit])

    def apply_retention(
        self,
        side: str,
        *,
        max_active: int,
        historical_keep: int = 4,
    ) -> tuple[StrategyRecord, ...]:
        if max_active <= 0:
            raise ValueError("max_active must be positive")
        pool = self._pool(side)
        selected: list[str] = []

        def add(strategy_id: str) -> None:
            if strategy_id not in selected and len(selected) < max_active:
                selected.append(strategy_id)

        historical = sorted(
            self._historical_ids(pool),
            key=lambda strategy_id: pool[strategy_id][1].strength,
            reverse=True,
        )
        for strategy_id in historical[: max(0, historical_keep)]:
            add(strategy_id)
        for strategy_id in self._diverse_ids(pool):
            add(strategy_id)
        for strategy_id, (_plan, _record) in sorted(pool.items(), key=lambda item: item[1][1].strength, reverse=True):
            add(strategy_id)

        selected_set = set(selected)
        for strategy_id, (plan, record) in list(pool.items()):
            active = strategy_id in selected_set
            pool[strategy_id] = (
                plan,
                replace(record, active=active, retired_reason=None if active else "retention"),
            )
        return tuple(record for _plan, record in pool.values())

    def next_iteration(self) -> int:
        self.iteration += 1
        return self.iteration

    def _pool(self, side: str) -> dict[str, tuple[AttackPlan | DefensePlan, StrategyRecord]]:
        if side == "attack":
            return self.attack_pool
        if side == "defense":
            return self.defense_pool
        raise ValueError("side must be 'attack' or 'defense'")

    def _active_pool(self, side: str) -> dict[str, tuple[AttackPlan | DefensePlan, StrategyRecord]]:
        return {strategy_id: item for strategy_id, item in self._pool(side).items() if item[1].active}

    def _historical_ids(self, pool: dict[str, tuple[AttackPlan | DefensePlan, StrategyRecord]]) -> tuple[str, ...]:
        historical = [
            strategy_id
            for strategy_id, (_plan, record) in pool.items()
            if record.role == "historical" or record.created_iteration < max(self.iteration, 1) - 1
        ]
        return tuple(historical)

    def _diverse_ids(self, pool: dict[str, tuple[AttackPlan | DefensePlan, StrategyRecord]]) -> tuple[str, ...]:
        best_by_cluster: dict[str, tuple[str, float]] = {}
        for strategy_id, (_plan, record) in pool.items():
            existing = best_by_cluster.get(record.diversity_cluster)
            if existing is None or record.strength > existing[1]:
                best_by_cluster[record.diversity_cluster] = (strategy_id, record.strength)
        return tuple(strategy_id for strategy_id, _strength in best_by_cluster.values())

    def _uniform_distribution(self, strategy_ids: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
        if not strategy_ids:
            return ()
        weight = 1.0 / len(strategy_ids)
        return tuple((strategy_id, weight) for strategy_id in strategy_ids)


def _plan_diversity_cluster(plan: AttackPlan | DefensePlan) -> str:
    hero_ids = sorted(loadout.hero_id for team in plan.teams for loadout in team.slots)
    return "heroes-" + canonical_hash((plan.format.n_teams, hero_ids))[:12]


def _plan_resource_cost(plan: AttackPlan | DefensePlan) -> float:
    return sum(team.total_cost for team in plan.teams)
