from __future__ import annotations

from dataclasses import dataclass

from .models import AttackPlan, DefensePlan


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

    def add_attack(self, plan: AttackPlan, *, role: str, source: str, strength: float) -> StrategyRecord:
        strategy_id = f"atk-{len(self.attack_pool) + 1:06d}"
        record = StrategyRecord(strategy_id, role, "attack", plan.hash(), source, self.iteration, strength)
        self.attack_pool[strategy_id] = (plan, record)
        return record

    def add_defense(self, plan: DefensePlan, *, role: str, source: str, strength: float) -> StrategyRecord:
        strategy_id = f"def-{len(self.defense_pool) + 1:06d}"
        record = StrategyRecord(strategy_id, role, "defense", plan.hash(), source, self.iteration, strength)
        self.defense_pool[strategy_id] = (plan, record)
        return record

    def record_payoff(self, attack_id: str, defense_id: str, *, attack_success: float, games: int) -> None:
        self.payoffs[(attack_id, defense_id)] = PayoffEntry(attack_id, defense_id, attack_success, games)

    def meta_distribution(self, side: str) -> tuple[tuple[str, float], ...]:
        pool = self.attack_pool if side == "attack" else self.defense_pool
        if not pool:
            return ()
        strengths = [(strategy_id, max(record.strength, 1e-6)) for strategy_id, (_plan, record) in pool.items()]
        total = sum(strength for _strategy_id, strength in strengths)
        return tuple((strategy_id, strength / total) for strategy_id, strength in strengths)

    def next_iteration(self) -> int:
        self.iteration += 1
        return self.iteration
