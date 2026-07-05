from __future__ import annotations

from dataclasses import dataclass

from ..belief import BeliefOutput
from ..domain import AttackPlan, Loadout, MatchFormat, Observation
from .legal_generator import GenerationGoal, LegalPlanGenerator


@dataclass(frozen=True)
class AttackNetInput:
    format: MatchFormat
    observation: Observation
    belief_topk: BeliefOutput
    attack_loadout_pool: tuple[Loadout, ...]
    selected_loadouts: tuple[Loadout, ...]
    used_hero_ids: frozenset[int]
    used_unique_equip_ids: frozenset[int]
    current_slot: tuple[int, int]
    goal: GenerationGoal


@dataclass(frozen=True)
class AttackNetOutput:
    candidates: tuple[AttackPlan, ...]
    log_probs: tuple[float, ...]
    proposal_scores: tuple[float, ...]
    value_estimates: tuple[float, ...]
    legality_flags: tuple[bool, ...]


class LegalMaskedAttackGenerator:
    """Action-mask based proposal generator used before neural distillation."""

    def __init__(self, loadout_pool: tuple[Loadout, ...], *, seed: int = 0) -> None:
        self.generator = LegalPlanGenerator(loadout_pool, seed=seed)

    def generate(self, input_data: AttackNetInput, *, count: int = 8) -> AttackNetOutput:
        candidates = self.generator.generate_attack_candidates(input_data.format, count=count, goal=input_data.goal)
        return AttackNetOutput(
            candidates=tuple(candidates),
            log_probs=tuple(0.0 for _ in candidates),
            proposal_scores=tuple(0.0 for _ in candidates),
            value_estimates=tuple(0.0 for _ in candidates),
            legality_flags=tuple(True for _ in candidates),
        )
