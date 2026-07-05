from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .belief import BeliefEngine, BeliefOutput
from .cache import MatchupCacheKey, SimulationCache, SurrogateSimulator
from .constraints import ConstraintEngine
from .evaluation import diversity_select, match_win_probability, plan_cost
from .generation import GenerationGoal, LegalPlanGenerator
from .hyperband import HalvingStage, HalvingTrace, successive_halving
from .models import AttackPlan, DefensePlan, Observation, ResultMetadata, Team
from .output_contracts import failure_diagnostics, jsonable
from .surrogate import HeuristicSurrogateScorer, SurrogateScorer

AttackCandidateSource = Callable[..., Sequence[AttackPlan]]


@dataclass(frozen=True)
class AttackOracleConfig:
    candidate_count: int = 256
    diversity_keep: int = 64
    final_keep: int = 5
    lcb_beta: float = 1.0
    diversity_weight: float = 0.05
    underdog_residual_weight: float = 0.0
    use_future_feasibility_mask: bool = True
    halving_stages: tuple[HalvingStage, ...] = (
        HalvingStage(games_each=3, keep=24),
        HalvingStage(games_each=10, keep=8),
        HalvingStage(games_each=30, keep=5),
    )


@dataclass(frozen=True)
class AttackOracleOutput:
    ranked_attacks: tuple[AttackPlan, ...]
    predicted_scores: tuple[float, ...]
    simulated_scores: tuple[float, ...]
    belief: BeliefOutput
    traces: tuple[HalvingTrace[AttackPlan], ...]
    explanation: dict[str, str]
    risk_report: dict[str, Any]
    metadata: ResultMetadata

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "attack_oracle_output.v1",
            "module": "AttackOracle",
            "metadata": jsonable(self.metadata),
            "ranked_attack_hashes": [plan.hash() for plan in self.ranked_attacks],
            "ranked_attacks": jsonable(self.ranked_attacks),
            "predicted_scores": [float(value) for value in self.predicted_scores],
            "simulated_scores": [float(value) for value in self.simulated_scores],
            "belief_summary": {
                "entropy": float(self.belief.entropy),
                "feasible_count_estimate": int(self.belief.feasible_count_estimate),
                "top1_top2_gap": float(self.belief.top1_top2_gap),
                "domain_stats": jsonable(self.belief.domain_stats),
            },
            "halving_traces": [
                {
                    "stage_index": trace.stage_index,
                    "games_each": trace.games_each,
                    "kept_hashes": [plan.hash() for plan in trace.kept],
                    "scores": jsonable(trace.scores),
                }
                for trace in self.traces
            ],
            "explanation": jsonable(self.explanation),
            "risk_report": jsonable(self.risk_report),
            "diagnostics": failure_diagnostics(self.risk_report),
        }


class AttackOracle:
    def __init__(
        self,
        *,
        loadout_pool,
        constraint_engine: ConstraintEngine | None = None,
        surrogate: SurrogateScorer | None = None,
        cache: SimulationCache | None = None,
        candidate_sources: Sequence[AttackCandidateSource] = (),
        belief_engine: BeliefEngine | None = None,
        seed: int = 0,
        config: AttackOracleConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.constraint_engine = constraint_engine or ConstraintEngine(self.loadout_pool)
        self.belief_engine = belief_engine or BeliefEngine(self.constraint_engine)
        self.config = config or AttackOracleConfig()
        self.surrogate = surrogate or HeuristicSurrogateScorer()
        self.cache = cache or SimulationCache()
        self.candidate_sources = tuple(candidate_sources)
        self.simulator = SurrogateSimulator(self.surrogate)
        self.generator = LegalPlanGenerator(
            self.loadout_pool,
            seed=seed,
            use_future_feasibility=self.config.use_future_feasibility_mask,
        )
        self.seed = seed

    def search(
        self,
        target: DefensePlan | Observation,
        *,
        goal: GenerationGoal | None = None,
        metadata: ResultMetadata | None = None,
    ) -> AttackOracleOutput:
        belief = self._belief_from_target(target)
        if not belief.candidates:
            return AttackOracleOutput(
                ranked_attacks=(),
                predicted_scores=(),
                simulated_scores=(),
                belief=belief,
                traces=(),
                explanation={"failure": "no legal belief candidates"},
                risk_report=_failure_report(
                    code="NO_LEGAL_BELIEF_CANDIDATES",
                    stage="belief",
                    message="no legal belief candidates",
                    belief=belief,
                    extra={
                        "belief_feasible_count": belief.feasible_count_estimate,
                        "belief_entropy": belief.entropy,
                        "domain_stats": list(belief.domain_stats),
                    },
                ),
                metadata=metadata or ResultMetadata(random_seed=self.seed),
            )
        match_format = target.format
        reference_cost = _average_roster_cost(belief.candidates, belief.weights)
        candidates = self.generator.generate_attack_candidates(
            match_format,
            count=self.config.candidate_count,
            goal=goal,
            reference_cost=reference_cost,
        )
        candidates.extend(self._external_candidates(target, match_format, belief, goal, reference_cost))
        legal_candidates = self._legal_unique_candidates(candidates)
        fallback_used = False
        if not legal_candidates and goal is not None and goal.target_power_ratio < 1.0:
            fallback_used = True
            fallback_goal = GenerationGoal(
                target_power_ratio=1.0,
                explore_beta=goal.explore_beta,
                diversity_weight=goal.diversity_weight,
            )
            candidates = self.generator.generate_attack_candidates(
                match_format,
                count=self.config.candidate_count,
                goal=fallback_goal,
                reference_cost=reference_cost,
            )
            candidates.extend(self._external_candidates(target, match_format, belief, fallback_goal, reference_cost))
            legal_candidates = self._legal_unique_candidates(candidates)
        if not legal_candidates:
            return AttackOracleOutput(
                ranked_attacks=(),
                predicted_scores=(),
                simulated_scores=(),
                belief=belief,
                traces=(),
                explanation={"failure": "no legal attack candidates"},
                risk_report=_failure_report(
                    code="NO_LEGAL_ATTACK_CANDIDATES",
                    stage="candidate_generation",
                    message="no legal attack candidates",
                    belief=belief,
                    extra={
                        "generated_candidate_count": len(candidates),
                        "legal_candidate_count": 0,
                        "external_candidate_source_count": len(self.candidate_sources),
                        "fallback_used": fallback_used,
                        "target_power_ratio": None if goal is None else goal.target_power_ratio,
                        "reference_cost": reference_cost,
                    },
                ),
                metadata=metadata or ResultMetadata(random_seed=self.seed),
            )
        scored = [
            (candidate, self._objective_score(self._score_candidate(candidate, belief), candidate, reference_cost, goal))
            for candidate in legal_candidates
        ]
        selected = diversity_select(
            scored,
            keep=min(self.config.diversity_keep, len(scored)),
            diversity_weight=self.config.diversity_weight,
        )
        selected_plans = [item.item for item in selected]
        final_plans, traces = successive_halving(
            selected_plans,
            stages=self.config.halving_stages,
            evaluate=lambda plan, games: self._objective_score(
                self._simulate_candidate(plan, belief, games),
                plan,
                reference_cost,
                goal,
            ),
            key=lambda plan: plan.hash(),
        )
        final_scores = [self._simulate_candidate(plan, belief, self.config.halving_stages[-1].games_each) for plan in final_plans]
        predicted_scores = [self._score_candidate(plan, belief) for plan in final_plans]
        final_objective_scores = [
            self._objective_score(score, plan, reference_cost, goal)
            for plan, score in zip(final_plans, final_scores)
        ]
        ranked = sorted(
            zip(final_plans, predicted_scores, final_scores, final_objective_scores),
            key=lambda item: item[3],
            reverse=True,
        )
        ranked = ranked[: self.config.final_keep]
        return AttackOracleOutput(
            ranked_attacks=tuple(plan for plan, _pred, _sim, _objective in ranked),
            predicted_scores=tuple(pred for _plan, pred, _sim, _objective in ranked),
            simulated_scores=tuple(sim for _plan, _pred, sim, _objective in ranked),
            belief=belief,
            traces=traces,
            explanation=self._explain(ranked, belief, reference_cost, goal),
            risk_report=self._risk_report(ranked, belief, reference_cost, goal),
            metadata=metadata or ResultMetadata(random_seed=self.seed),
        )

    def _external_candidates(
        self,
        target: DefensePlan | Observation,
        match_format,
        belief: BeliefOutput,
        goal: GenerationGoal | None,
        reference_cost: float,
    ) -> list[AttackPlan]:
        candidates: list[AttackPlan] = []
        for source in self.candidate_sources:
            produced = source(
                target=target,
                match_format=match_format,
                belief=belief,
                goal=goal,
                reference_cost=reference_cost,
                loadout_pool=self.loadout_pool,
                constraint_engine=self.constraint_engine,
            )
            candidates.extend(produced)
        return candidates

    def _legal_unique_candidates(self, candidates: Sequence[AttackPlan]) -> list[AttackPlan]:
        legal: list[AttackPlan] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not self.constraint_engine.is_legal_attack(candidate):
                continue
            digest = candidate.hash()
            if digest in seen:
                continue
            seen.add(digest)
            legal.append(candidate)
        return legal

    def _belief_from_target(self, target: DefensePlan | Observation) -> BeliefOutput:
        if isinstance(target, DefensePlan):
            return BeliefOutput(
                candidates=(target.teams,),
                weights=(1.0,),
                entropy=0.0,
                feasible_count_estimate=1,
                top1_top2_gap=1.0,
                domain_stats=(),
            )
        return self.belief_engine.build(target)

    def _score_candidate(self, plan: AttackPlan, belief: BeliefOutput) -> float:
        total = 0.0
        for roster, weight in zip(belief.candidates, belief.weights):
            probabilities = []
            for attack_team, defense_team in zip(plan.teams, roster):
                prediction = self.surrogate.predict(attack_team, defense_team)
                probabilities.append(prediction.conservative(beta=self.config.lcb_beta))
            total += weight * match_win_probability(probabilities, plan.format.win_required)
        return total

    def _simulate_candidate(self, plan: AttackPlan, belief: BeliefOutput, games_each: int) -> float:
        total = 0.0
        for roster, weight in zip(belief.candidates, belief.weights):
            probabilities = list(self._lane_win_rates_for_roster(plan, roster, games_each))
            total += weight * match_win_probability(probabilities, plan.format.win_required)
        return total

    def _objective_score(
        self,
        match_win: float,
        plan: AttackPlan,
        reference_cost: float,
        goal: GenerationGoal | None,
    ) -> float:
        gap = _attack_underdog_gap(plan, reference_cost)
        if goal is None or goal.target_power_ratio >= 1.0:
            return float(match_win)
        return float(match_win) + self.config.underdog_residual_weight * gap

    def _lane_win_rates_for_roster(self, plan: AttackPlan, roster: tuple[Team, ...], games_each: int) -> tuple[float, ...]:
        probabilities = []
        for attack_team, defense_team in zip(plan.teams, roster):
            key = MatchupCacheKey.from_teams(attack_team, defense_team)
            result = self.cache.get_or_run(key, lambda a=attack_team, d=defense_team: self.simulator.run(a, d, games=games_each))
            probabilities.append(result.win_rate)
        return tuple(float(value) for value in probabilities)

    def _risk_report(
        self,
        ranked: list[tuple[AttackPlan, float, float, float]],
        belief: BeliefOutput,
        reference_cost: float,
        goal: GenerationGoal | None,
    ) -> dict[str, Any]:
        if not ranked:
            return _failure_report(
                code="NO_ATTACK_SURVIVED_HALVING",
                stage="successive_halving",
                message="no attack survived successive halving",
                belief=belief,
            )
        best, predicted, simulated, objective = ranked[0]
        games_each = self.config.halving_stages[-1].games_each
        belief_cases: list[dict[str, Any]] = []
        expected_lanes = [0.0 for _team in best.teams]
        for index, (roster, weight) in enumerate(zip(belief.candidates, belief.weights)):
            lane_rates = self._lane_win_rates_for_roster(best, roster, games_each)
            match_win = match_win_probability(lane_rates, best.format.win_required)
            for lane_idx, value in enumerate(lane_rates):
                expected_lanes[lane_idx] += float(weight) * float(value)
            belief_cases.append(
                {
                    "belief_index": index,
                    "weight": float(weight),
                    "lane_win_rates": list(lane_rates),
                    "match_win": float(match_win),
                }
            )
        worst_case = min(belief_cases, key=lambda item: float(item["match_win"]), default=None)
        backup_match_wins = [float(sim) for _plan, _pred, sim, _objective in ranked[1:]]
        backup_hashes = [plan.hash() for plan, _pred, _sim, _objective in ranked[1:]]
        underdog_gap = _attack_underdog_gap(best, reference_cost)
        underdog_bonus = self.config.underdog_residual_weight * underdog_gap if goal is not None and goal.target_power_ratio < 1.0 else 0.0
        return {
            "best_attack_hash": best.hash(),
            "expected_match_win": float(simulated),
            "predicted_match_win": float(predicted),
            "objective_score": float(objective),
            "underdog_gap": float(underdog_gap),
            "underdog_residual_bonus": float(underdog_bonus),
            "reference_defense_cost": float(reference_cost),
            "attack_cost": float(plan_cost(best)),
            "expected_lane_win_rates": [float(value) for value in expected_lanes],
            "worst_case_match_win": 0.0 if worst_case is None else float(worst_case["match_win"]),
            "worst_case_lane_win_rates": [] if worst_case is None else list(worst_case["lane_win_rates"]),
            "worst_case_belief_index": None if worst_case is None else int(worst_case["belief_index"]),
            "worst_case_belief_weight": 0.0 if worst_case is None else float(worst_case["weight"]),
            "backup_attack_count": len(ranked) - 1,
            "backup_attack_hashes": backup_hashes,
            "backup_match_wins": backup_match_wins,
            "belief_case_count": len(belief_cases),
            "belief_case_match_wins": [float(item["match_win"]) for item in belief_cases],
        }

    def _explain(
        self,
        ranked: list[tuple[AttackPlan, float, float, float]],
        belief: BeliefOutput,
        reference_cost: float,
        goal: GenerationGoal | None,
    ) -> dict[str, str]:
        if not ranked:
            return {"failure": "no attack survived successive halving"}
        best, predicted, simulated, objective = ranked[0]
        underdog_gap = _attack_underdog_gap(best, reference_cost)
        underdog_bonus = self.config.underdog_residual_weight * underdog_gap if goal is not None and goal.target_power_ratio < 1.0 else 0.0
        return {
            "belief_candidates": str(belief.feasible_count_estimate),
            "belief_entropy": f"{belief.entropy:.4f}",
            "predicted_match_win": f"{predicted:.4f}",
            "simulated_match_win": f"{simulated:.4f}",
            "objective_score": f"{objective:.4f}",
            "attack_cost": f"{plan_cost(best):.2f}",
            "reference_defense_cost": f"{reference_cost:.2f}",
            "underdog_gap": f"{underdog_gap:.4f}",
            "underdog_residual_bonus": f"{underdog_bonus:.4f}",
            "candidate_sources": str(len(self.candidate_sources)),
        }


def _average_roster_cost(candidates: tuple[tuple[Team, ...], ...], weights: tuple[float, ...]) -> float:
    total = 0.0
    for roster, weight in zip(candidates, weights):
        total += weight * sum(team.total_cost for team in roster)
    return total


def _attack_underdog_gap(plan: AttackPlan, reference_cost: float) -> float:
    return max(0.0, (float(reference_cost) - plan_cost(plan)) / max(float(reference_cost), 1e-9))


def _failure_report(
    *,
    code: str,
    stage: str,
    message: str,
    belief: BeliefOutput,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "failure": message,
        "failure_code": code,
        "failure_stage": stage,
        "belief_feasible_count": int(belief.feasible_count_estimate),
        "belief_entropy": float(belief.entropy),
        "belief_top1_top2_gap": float(belief.top1_top2_gap),
        "domain_stats": list(belief.domain_stats),
        "failure_context": {
            "belief_feasible_count": int(belief.feasible_count_estimate),
            "belief_entropy": float(belief.entropy),
        },
    }
    if extra:
        report.update(extra)
        report["failure_context"] = {**report["failure_context"], **extra}
    return report
