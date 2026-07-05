from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .attack_oracle import AttackOracle, AttackOracleConfig
from .belief import BeliefOutput
from .constraints import ConstraintEngine
from .evaluation import plan_cost
from .generation import GenerationGoal, LegalPlanGenerator
from .mask import MaskSearcher, MaskSlotScoreProvider
from .models import AttackPlan, DefensePlan, MatchFormat, ResultMetadata, Team, observe_defense
from .output_contracts import failure_diagnostics, jsonable

DefenseRosterSource = Callable[..., Sequence[tuple[Team, ...] | DefensePlan]]


@dataclass(frozen=True)
class DefenseOracleConfig:
    roster_candidates: int = 32
    masks_per_roster: int = 4
    max_masks_per_roster: int | None = 256
    underdog_residual_weight: float = 0.0
    use_future_feasibility_mask: bool = True


@dataclass(frozen=True)
class DefenseOracleOutput:
    best_defense: DefensePlan | None
    backup_defenses: tuple[DefensePlan, ...]
    estimated_attack_success: float
    ambiguity_score: float
    worst_case_attack: AttackPlan | None
    explanation: dict[str, str]
    risk_report: dict[str, Any]
    metadata: ResultMetadata

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "defense_oracle_output.v1",
            "module": "DefenseOracle",
            "metadata": jsonable(self.metadata),
            "best_defense_hash": None if self.best_defense is None else self.best_defense.hash(),
            "best_defense": None if self.best_defense is None else jsonable(self.best_defense),
            "backup_defense_hashes": [plan.hash() for plan in self.backup_defenses],
            "backup_defenses": jsonable(self.backup_defenses),
            "estimated_attack_success": float(self.estimated_attack_success),
            "ambiguity_score": float(self.ambiguity_score),
            "worst_case_attack_hash": None if self.worst_case_attack is None else self.worst_case_attack.hash(),
            "worst_case_attack": None if self.worst_case_attack is None else jsonable(self.worst_case_attack),
            "explanation": jsonable(self.explanation),
            "risk_report": jsonable(self.risk_report),
            "diagnostics": failure_diagnostics(self.risk_report),
        }


class DefenseOracle:
    def __init__(
        self,
        *,
        loadout_pool,
        constraint_engine: ConstraintEngine | None = None,
        attack_oracle: AttackOracle | None = None,
        roster_sources: Sequence[DefenseRosterSource] = (),
        mask_slot_score_provider: MaskSlotScoreProvider | None = None,
        learned_mask_score_weight: float = 0.05,
        seed: int = 0,
        config: DefenseOracleConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.constraint_engine = constraint_engine or ConstraintEngine(self.loadout_pool)
        self.config = config or DefenseOracleConfig()
        self.generator = LegalPlanGenerator(
            self.loadout_pool,
            seed=seed,
            use_future_feasibility=self.config.use_future_feasibility_mask,
        )
        self.mask_searcher = MaskSearcher(
            self.constraint_engine,
            slot_score_provider=mask_slot_score_provider,
            learned_score_weight=learned_mask_score_weight,
        )
        self.attack_oracle = attack_oracle or AttackOracle(
            loadout_pool=self.loadout_pool,
            constraint_engine=self.constraint_engine,
            seed=seed,
            config=AttackOracleConfig(candidate_count=64, diversity_keep=16, final_keep=3),
        )
        self.roster_sources = tuple(roster_sources)
        self.seed = seed

    def search(
        self,
        match_format: MatchFormat,
        *,
        attack_meta: tuple[tuple[AttackPlan, float], ...] = (),
        goal: GenerationGoal | None = None,
        metadata: ResultMetadata | None = None,
    ) -> DefenseOracleOutput:
        normalized_attack_meta = _normalize_attack_meta(attack_meta)
        reference_cost = _average_attack_meta_cost(normalized_attack_meta)
        candidates: list[tuple[DefensePlan, float, float, float, AttackPlan | None, dict[str, Any], dict[str, Any]]] = []
        seen: set[str] = set()
        roster_candidates = list(
            self._external_rosters(
                match_format,
                attack_meta=normalized_attack_meta,
                goal=goal,
                reference_cost=reference_cost,
            )
        )
        attempts = 0
        while len(roster_candidates) < self.config.roster_candidates and attempts < max(self.config.roster_candidates * 10, 10):
            attempts += 1
            try:
                roster_candidates.append(
                    self.generator.generate_defense_plan(
                        match_format,
                        goal=goal,
                        reference_cost=reference_cost,
                    ).teams
                )
            except ValueError:
                continue
        for roster in roster_candidates[: self.config.roster_candidates]:
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
                meta_attack_success = self._score_attack_meta(roster, normalized_attack_meta)
                ambiguity = attack_output.belief.entropy
                anti_meta_score = 1.0 - meta_attack_success if normalized_attack_meta else 0.0
                reference_cost_value = 0.0 if reference_cost is None else float(reference_cost)
                underdog_gap = _defense_underdog_gap(roster, reference_cost)
                underdog_bonus = (
                    self.config.underdog_residual_weight * underdog_gap
                    if goal is not None and goal.target_power_ratio < 1.0
                    else 0.0
                )
                defense_score = (
                    (1.0 - attack_success)
                    + 0.25 * anti_meta_score
                    + 0.05 * ambiguity
                    + 0.02 * mask_score
                    + underdog_bonus
                )
                worst_attack = attack_output.ranked_attacks[0] if attack_output.ranked_attacks else None
                candidates.append(
                    (
                        defense,
                        defense_score,
                        attack_success,
                        meta_attack_success,
                        worst_attack,
                        {
                            "mask_score": f"{mask_score:.4f}",
                            "belief_entropy": f"{ambiguity:.4f}",
                            "domain_log": f"{mask_stats.get('domain_log', 0.0):.4f}",
                            "hidden_count": f"{mask_stats.get('hidden_count', 0.0):.0f}",
                            "learned_mask_score": f"{mask_stats.get('learned_mask_score', 0.0):.4f}",
                            "top_hidden_slots": _top_hidden_slots_summary(mask_stats.get("hidden_slot_explanations", ())),
                            "mask_explanation": _mask_explanation(mask_stats),
                            "attack_meta_count": str(len(normalized_attack_meta)),
                            "meta_attack_success": f"{meta_attack_success:.4f}",
                            "defense_cost": f"{_roster_cost(roster):.2f}",
                            "reference_attack_cost": f"{reference_cost_value:.2f}",
                            "underdog_defense_gap": f"{underdog_gap:.4f}",
                            "underdog_residual_bonus": f"{underdog_bonus:.4f}",
                            "roster_sources": str(len(self.roster_sources)),
                        },
                        dict(attack_output.risk_report),
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
                risk_report={"failure": "no legal defense candidates"},
                metadata=metadata or ResultMetadata(random_seed=self.seed),
            )
        candidates.sort(key=lambda item: item[1], reverse=True)
        best, score, attack_success, meta_attack_success, worst_attack, details, attack_risk = candidates[0]
        backup_rows = candidates[1:5]
        return DefenseOracleOutput(
            best_defense=best,
            backup_defenses=tuple(item[0] for item in backup_rows),
            estimated_attack_success=attack_success,
            ambiguity_score=float(details["belief_entropy"]),
            worst_case_attack=worst_attack,
            explanation={
                "defense_score": f"{score:.4f}",
                "estimated_attack_success": f"{attack_success:.4f}",
                **details,
            },
            risk_report={
                "best_defense_hash": best.hash(),
                "estimated_break_rate": float(attack_success),
                "estimated_survival_rate": float(1.0 - attack_success),
                "meta_attack_success": float(meta_attack_success),
                "defense_cost": float(details.get("defense_cost", 0.0)),
                "reference_attack_cost": float(details.get("reference_attack_cost", 0.0)),
                "underdog_defense_gap": float(details.get("underdog_defense_gap", 0.0)),
                "underdog_residual_bonus": float(details.get("underdog_residual_bonus", 0.0)),
                "worst_case_attack_hash": None if worst_attack is None else worst_attack.hash(),
                "backup_defense_count": len(backup_rows),
                "backup_defense_hashes": [item[0].hash() for item in backup_rows],
                "backup_break_rates": [float(item[2]) for item in backup_rows],
                "backup_survival_rates": [float(1.0 - item[2]) for item in backup_rows],
                "hidden_count": int(sum(sum(row) for row in best.mask)),
                "learned_mask_score": float(details.get("learned_mask_score", 0.0)),
                "mask_explanation": details.get("mask_explanation", {}),
                "counter_attack_risk_report": attack_risk,
            },
            metadata=metadata or ResultMetadata(random_seed=self.seed),
        )

    def _score_attack_meta(
        self,
        roster: tuple[Team, ...],
        attack_meta: tuple[tuple[AttackPlan, float], ...],
    ) -> float:
        if not attack_meta:
            return 0.0
        belief = BeliefOutput(
            candidates=(roster,),
            weights=(1.0,),
            entropy=0.0,
            feasible_count_estimate=1,
            top1_top2_gap=1.0,
            domain_stats=(),
        )
        return sum(weight * self.attack_oracle._score_candidate(attack, belief) for attack, weight in attack_meta)

    def _external_rosters(
        self,
        match_format: MatchFormat,
        *,
        attack_meta: tuple[tuple[AttackPlan, float], ...],
        goal: GenerationGoal | None,
        reference_cost: float | None,
    ) -> tuple[tuple[Team, ...], ...]:
        rosters: list[tuple[Team, ...]] = []
        for source in self.roster_sources:
            produced = source(
                match_format=match_format,
                attack_meta=attack_meta,
                goal=goal,
                reference_cost=reference_cost,
                loadout_pool=self.loadout_pool,
                constraint_engine=self.constraint_engine,
            )
            for item in produced:
                roster = item.teams if isinstance(item, DefensePlan) else tuple(item)
                if len(roster) != match_format.n_teams:
                    continue
                rosters.append(roster)
        return tuple(rosters)


def _roster_hash(roster: tuple[Team, ...]) -> str:
    from .models import canonical_hash

    return canonical_hash(roster)


def _mask_explanation(mask_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_count": int(float(mask_stats.get("hidden_count", 0.0) or 0.0)),
        "domain_log": float(mask_stats.get("domain_log", 0.0) or 0.0),
        "leakage_penalty": float(mask_stats.get("leakage_penalty", 0.0) or 0.0),
        "learned_mask_score": float(mask_stats.get("learned_mask_score", 0.0) or 0.0),
        "learned_score_weight": float(mask_stats.get("learned_score_weight", 0.0) or 0.0),
        "learned_slot_scores": mask_stats.get("learned_slot_scores", ()),
        "hidden_slot_explanations": list(mask_stats.get("hidden_slot_explanations", ())),
        "top_learned_slots": list(mask_stats.get("top_learned_slots", ())),
    }


def _top_hidden_slots_summary(rows: Any) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ""
    parts: list[str] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        parts.append(
            f"t{int(row.get('team_index', 0))}s{int(row.get('slot_index', 0))}:"
            f"{float(row.get('learned_slot_score', 0.0)):.3f}"
        )
    return ",".join(parts)


def _normalize_attack_meta(attack_meta: tuple[tuple[AttackPlan, float], ...]) -> tuple[tuple[AttackPlan, float], ...]:
    positive = tuple((attack, max(float(weight), 0.0)) for attack, weight in attack_meta)
    total = sum(weight for _attack, weight in positive)
    if total <= 0.0:
        return ()
    return tuple((attack, weight / total) for attack, weight in positive if weight > 0.0)


def _average_attack_meta_cost(attack_meta: tuple[tuple[AttackPlan, float], ...]) -> float | None:
    if not attack_meta:
        return None
    return sum(plan_cost(attack) * weight for attack, weight in attack_meta)


def _roster_cost(roster: tuple[Team, ...]) -> float:
    return sum(team.total_cost for team in roster)


def _defense_underdog_gap(roster: tuple[Team, ...], reference_attack_cost: float | None) -> float:
    if reference_attack_cost is None:
        return 0.0
    return max(0.0, (float(reference_attack_cost) - _roster_cost(roster)) / max(float(reference_attack_cost), 1e-9))
