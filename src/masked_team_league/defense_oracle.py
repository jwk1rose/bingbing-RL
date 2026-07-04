from __future__ import annotations

from dataclasses import dataclass

from .attack_oracle import AttackOracle, AttackOracleConfig
from .constraints import ConstraintEngine
from .generation import GenerationGoal, LegalPlanGenerator
from .mask import MaskSearcher
from .models import AttackPlan, DefensePlan, MatchFormat, ResultMetadata, Team, observe_defense


@dataclass(frozen=True)
class DefenseOracleConfig:
    roster_candidates: int = 32
    masks_per_roster: int = 4
    max_masks_per_roster: int | None = 256


@dataclass(frozen=True)
class DefenseOracleOutput:
    best_defense: DefensePlan | None
    backup_defenses: tuple[DefensePlan, ...]
    estimated_attack_success: float
    ambiguity_score: float
    worst_case_attack: AttackPlan | None
    explanation: dict[str, str]
    metadata: ResultMetadata


class DefenseOracle:
    def __init__(
        self,
        *,
        loadout_pool,
        constraint_engine: ConstraintEngine | None = None,
        attack_oracle: AttackOracle | None = None,
        seed: int = 0,
        config: DefenseOracleConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.constraint_engine = constraint_engine or ConstraintEngine(self.loadout_pool)
        self.generator = LegalPlanGenerator(self.loadout_pool, seed=seed)
        self.mask_searcher = MaskSearcher(self.constraint_engine)
        self.attack_oracle = attack_oracle or AttackOracle(
            loadout_pool=self.loadout_pool,
            constraint_engine=self.constraint_engine,
            seed=seed,
            config=AttackOracleConfig(candidate_count=64, diversity_keep=16, final_keep=3),
        )
        self.config = config or DefenseOracleConfig()
        self.seed = seed

    def search(
        self,
        match_format: MatchFormat,
        *,
        attack_meta: tuple[tuple[AttackPlan, float], ...] = (),
        goal: GenerationGoal | None = None,
        metadata: ResultMetadata | None = None,
    ) -> DefenseOracleOutput:
        del attack_meta
        candidates: list[tuple[DefensePlan, float, float, AttackPlan | None, dict[str, str]]] = []
        seen: set[str] = set()
        for _ in range(self.config.roster_candidates):
            try:
                roster = self.generator.generate_defense_plan(match_format, goal=goal).teams
            except ValueError:
                continue
            roster_hash = _roster_hash(roster)
            if roster_hash in seen:
                continue
            seen.add(roster_hash)
            for mask, mask_score, mask_stats in self.mask_searcher.search(
                match_format,
                roster,
                keep=self.config.masks_per_roster,
                max_masks=self.config.max_masks_per_roster,
            ):
                defense = DefensePlan(format=match_format, teams=roster, mask=mask, source="defense_oracle")
                if not self.constraint_engine.is_legal_defense(defense):
                    continue
                observation = observe_defense(defense)
                attack_output = self.attack_oracle.search(observation, goal=goal)
                attack_success = attack_output.simulated_scores[0] if attack_output.simulated_scores else 1.0
                ambiguity = attack_output.belief.entropy
                defense_score = (1.0 - attack_success) + 0.05 * ambiguity + 0.02 * mask_score
                worst_attack = attack_output.ranked_attacks[0] if attack_output.ranked_attacks else None
                candidates.append(
                    (
                        defense,
                        defense_score,
                        attack_success,
                        worst_attack,
                        {
                            "mask_score": f"{mask_score:.4f}",
                            "belief_entropy": f"{ambiguity:.4f}",
                            "domain_log": f"{mask_stats.get('domain_log', 0.0):.4f}",
                        },
                    )
                )
        if not candidates:
            return DefenseOracleOutput(
                best_defense=None,
                backup_defenses=(),
                estimated_attack_success=1.0,
                ambiguity_score=0.0,
                worst_case_attack=None,
                explanation={"failure": "no legal defense candidates"},
                metadata=metadata or ResultMetadata(random_seed=self.seed),
            )
        candidates.sort(key=lambda item: item[1], reverse=True)
        best, score, attack_success, worst_attack, details = candidates[0]
        return DefenseOracleOutput(
            best_defense=best,
            backup_defenses=tuple(item[0] for item in candidates[1:5]),
            estimated_attack_success=attack_success,
            ambiguity_score=float(details["belief_entropy"]),
            worst_case_attack=worst_attack,
            explanation={
                "defense_score": f"{score:.4f}",
                "estimated_attack_success": f"{attack_success:.4f}",
                **details,
            },
            metadata=metadata or ResultMetadata(random_seed=self.seed),
        )


def _roster_hash(roster: tuple[Team, ...]) -> str:
    from .models import canonical_hash

    return canonical_hash(roster)
