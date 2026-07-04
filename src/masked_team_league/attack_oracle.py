from __future__ import annotations

from dataclasses import dataclass

from .belief import BeliefEngine, BeliefOutput
from .cache import MatchupCacheKey, SimulationCache, SurrogateSimulator
from .constraints import ConstraintEngine
from .evaluation import diversity_select, match_win_probability, plan_cost
from .generation import GenerationGoal, LegalPlanGenerator
from .hyperband import HalvingStage, HalvingTrace, successive_halving
from .models import AttackPlan, DefensePlan, Observation, ResultMetadata, Team
from .surrogate import HeuristicSurrogateScorer, SurrogateScorer


@dataclass(frozen=True)
class AttackOracleConfig:
    candidate_count: int = 256
    diversity_keep: int = 64
    final_keep: int = 5
    lcb_beta: float = 1.0
    diversity_weight: float = 0.05
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
    metadata: ResultMetadata


class AttackOracle:
    def __init__(
        self,
        *,
        loadout_pool,
        constraint_engine: ConstraintEngine | None = None,
        surrogate: SurrogateScorer | None = None,
        cache: SimulationCache | None = None,
        seed: int = 0,
        config: AttackOracleConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.constraint_engine = constraint_engine or ConstraintEngine(self.loadout_pool)
        self.belief_engine = BeliefEngine(self.constraint_engine)
        self.surrogate = surrogate or HeuristicSurrogateScorer()
        self.cache = cache or SimulationCache()
        self.simulator = SurrogateSimulator(self.surrogate)
        self.generator = LegalPlanGenerator(self.loadout_pool, seed=seed)
        self.config = config or AttackOracleConfig()
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
        legal_candidates = [candidate for candidate in candidates if self.constraint_engine.is_legal_attack(candidate)]
        scored = [(candidate, self._score_candidate(candidate, belief)) for candidate in legal_candidates]
        selected = diversity_select(
            scored,
            keep=min(self.config.diversity_keep, len(scored)),
            diversity_weight=self.config.diversity_weight,
        )
        selected_plans = [item.item for item in selected]
        final_plans, traces = successive_halving(
            selected_plans,
            stages=self.config.halving_stages,
            evaluate=lambda plan, games: self._simulate_candidate(plan, belief, games),
            key=lambda plan: plan.hash(),
        )
        final_scores = [self._simulate_candidate(plan, belief, self.config.halving_stages[-1].games_each) for plan in final_plans]
        predicted_scores = [self._score_candidate(plan, belief) for plan in final_plans]
        ranked = sorted(zip(final_plans, predicted_scores, final_scores), key=lambda item: item[2], reverse=True)
        ranked = ranked[: self.config.final_keep]
        return AttackOracleOutput(
            ranked_attacks=tuple(plan for plan, _pred, _sim in ranked),
            predicted_scores=tuple(pred for _plan, pred, _sim in ranked),
            simulated_scores=tuple(sim for _plan, _pred, sim in ranked),
            belief=belief,
            traces=traces,
            explanation=self._explain(ranked, belief, reference_cost),
            metadata=metadata or ResultMetadata(random_seed=self.seed),
        )

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
            probabilities = []
            for attack_team, defense_team in zip(plan.teams, roster):
                key = MatchupCacheKey.from_teams(attack_team, defense_team)
                result = self.cache.get_or_run(key, lambda a=attack_team, d=defense_team: self.simulator.run(a, d, games=games_each))
                probabilities.append(result.win_rate)
            total += weight * match_win_probability(probabilities, plan.format.win_required)
        return total

    def _explain(
        self,
        ranked: list[tuple[AttackPlan, float, float]],
        belief: BeliefOutput,
        reference_cost: float,
    ) -> dict[str, str]:
        if not ranked:
            return {"failure": "no attack survived successive halving"}
        best, predicted, simulated = ranked[0]
        return {
            "belief_candidates": str(belief.feasible_count_estimate),
            "belief_entropy": f"{belief.entropy:.4f}",
            "predicted_match_win": f"{predicted:.4f}",
            "simulated_match_win": f"{simulated:.4f}",
            "attack_cost": f"{plan_cost(best):.2f}",
            "reference_defense_cost": f"{reference_cost:.2f}",
            "underdog_gap": f"{(reference_cost - plan_cost(best)) / max(reference_cost, 1e-9):.4f}",
        }


def _average_roster_cost(candidates: tuple[tuple[Team, ...], ...], weights: tuple[float, ...]) -> float:
    total = 0.0
    for roster, weight in zip(candidates, weights):
        total += weight * sum(team.total_cost for team in roster)
    return total
