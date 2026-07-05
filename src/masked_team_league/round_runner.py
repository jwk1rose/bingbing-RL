from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import json
from pathlib import Path
from typing import Any

from .active import ActivePerceptionScheduler, Query
from .attack_oracle import AttackOracle, AttackOracleConfig
from .belief import BeliefEngine
from .belief_ranker import load_belief_ranker_checkpoint
from .constraints import ConstraintEngine
from .data_tables import (
    LeagueStrategyTableRecord,
    LoadoutTableRecord,
    ObservationTableRecord,
    PlanMatchTableRecord,
    SingleMatchupTableRecord,
    write_table_jsonl,
)
from .defense_oracle import DefenseOracle, DefenseOracleConfig
from .generation import GenerationGoal
from .hyperband import HalvingStage
from .league import LeagueManager
from .model_selection import select_best_checkpoint
from .models import AttackPlan, DefensePlan, MatchFormat, ResultMetadata, canonical_hash, observe_defense
from .proposal_training import load_attack_proposal_candidate_source, load_defense_proposal_candidate_source
from .real_calibration import RealMetaDB
from .real_oracle import OracleBatchEvaluator, OracleEvaluationRecord
from .run_metadata import RunArtifactRef, RunMetadataManifest, hash_generation_config, write_run_metadata_manifest
from .surrogate import HeuristicSurrogateScorer


@dataclass(frozen=True)
class LeagueRoundConfig:
    teams: int = 3
    defenses: int = 20
    attacks_per_defense: int = 200
    oracle_top_k: int = 20
    seed: int = 0
    round_id: str = "round_0001"
    defense_roster_candidates: int = 8
    defense_masks_per_roster: int = 2
    defense_max_masks_per_roster: int | None = 128
    attack_roles: tuple[str, ...] = ("main", "exploiter", "underdog")
    defense_roles: tuple[str, ...] = ("main", "exploiter", "underdog")
    underdog_power_ratio: float = 0.9
    underdog_residual_weight: float = 0.25
    active_sim_keep: int = 32
    active_real_keep: int = 0
    attack_pool_max_active: int | None = None
    defense_pool_max_active: int | None = None
    historical_keep: int = 4
    attack_proposal_checkpoint: str | Path | None = None
    attack_proposal_beam_size: int = 8
    attack_proposal_device: str | None = None
    defense_proposal_checkpoint: str | Path | None = None
    defense_proposal_beam_size: int = 8
    defense_proposal_device: str | None = None
    belief_ranker_checkpoint: str | Path | None = None
    belief_ranker_registry: str | Path | None = None
    belief_ranker_metric: str = "holdout_top1_accuracy"
    belief_ranker_metric_mode: str = "max"
    belief_ranker_dataset_hash: str | None = None
    belief_ranker_weight: float = 1.0
    belief_ranker_device: str | None = None
    real_meta_db_jsonl: str | Path | None = None
    use_position_features: bool = True
    use_equipment_star_features: bool = True
    use_future_feasibility_mask: bool = True
    use_real_calibration: bool = True


@dataclass(frozen=True)
class LeagueRoundSummary:
    round_id: str
    out_dir: str
    defenses: int
    candidates: int
    oracle_pairs: int
    oracle_requests: int
    best_attack_success: float
    worst_defense_break_rate: float


class LeagueRoundRunner:
    def __init__(
        self,
        *,
        loadout_pool,
        evaluator: OracleBatchEvaluator,
        league: LeagueManager | None = None,
        config: LeagueRoundConfig | None = None,
    ) -> None:
        self.loadout_pool = tuple(loadout_pool)
        self.evaluator = evaluator
        self.league = league or LeagueManager()
        self.config = config or LeagueRoundConfig()
        self.engine = ConstraintEngine(self.loadout_pool)

    def run(self, out_dir: Path) -> LeagueRoundSummary:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.league.next_iteration()
        match_format = MatchFormat(self.config.teams)
        attack_candidate_sources = _attack_candidate_sources(self.config)
        defense_roster_sources = _defense_roster_sources(self.config)
        belief_engine, belief_ranker_checkpoint = _belief_engine_from_config(self.config, self.engine)
        surrogate = _surrogate_from_config(self.config)
        attack_oracle = AttackOracle(
            loadout_pool=self.loadout_pool,
            constraint_engine=self.engine,
            surrogate=surrogate,
            candidate_sources=attack_candidate_sources,
            belief_engine=belief_engine,
            seed=self.config.seed + 101,
            config=_attack_config(self.config),
        )
        defense_oracle = DefenseOracle(
            loadout_pool=self.loadout_pool,
            constraint_engine=self.engine,
            attack_oracle=AttackOracle(
                loadout_pool=self.loadout_pool,
                constraint_engine=self.engine,
                surrogate=surrogate,
                candidate_sources=attack_candidate_sources,
                belief_engine=belief_engine,
                seed=self.config.seed + 211,
                config=_defense_attack_config(self.config),
            ),
            roster_sources=defense_roster_sources,
            seed=self.config.seed + 303,
            config=DefenseOracleConfig(
                roster_candidates=self.config.defense_roster_candidates,
                masks_per_roster=self.config.defense_masks_per_roster,
                max_masks_per_roster=self.config.defense_max_masks_per_roster,
                underdog_residual_weight=self.config.underdog_residual_weight,
                use_future_feasibility_mask=self.config.use_future_feasibility_mask,
            ),
        )
        defenses: list[tuple[str, DefensePlan]] = []
        defense_details: dict[str, dict[str, Any]] = {}
        seen_defenses: set[str] = set()
        attempts = 0
        while len(defenses) < self.config.defenses and attempts < max(self.config.defenses * 10, 10):
            attempts += 1
            for defense_role in self.config.defense_roles:
                output = defense_oracle.search(
                    match_format,
                    attack_meta=self._attack_meta_for_defense_role(defense_role),
                    goal=self._goal_for_role(defense_role),
                )
                if output.best_defense is None:
                    continue
                for defense in (output.best_defense, *output.backup_defenses):
                    if len(defenses) >= self.config.defenses:
                        break
                    defense = replace(defense, source=f"defense_oracle:{defense_role}")
                    if not self.engine.is_legal_defense(defense):
                        continue
                    digest = canonical_hash((defense.teams, defense.mask))
                    if digest in seen_defenses:
                        continue
                    seen_defenses.add(digest)
                    defense_id = f"def-{len(defenses) + 1:05d}"
                    defenses.append((defense_id, defense))
                    defense_details[defense_id] = {
                        "role": defense_role,
                        "source": defense.source,
                        "estimated_attack_success": output.estimated_attack_success,
                        "ambiguity_score": output.ambiguity_score,
                        "hidden_count": sum(sum(row) for row in defense.mask),
                        "defense_oracle_explanation": output.explanation,
                        "defense_risk_report": output.risk_report,
                    }
                if len(defenses) >= self.config.defenses:
                    break

        candidate_rows: list[dict[str, Any]] = []
        pairs: list[tuple[str, AttackPlan, str, DefensePlan, str]] = []
        attack_by_hash_role: dict[tuple[str, str], str] = {}
        attack_counter = 0
        for defense_id, defense in defenses:
            observation = observe_defense(defense)
            for attack_role in self.config.attack_roles:
                output = attack_oracle.search(observation, goal=self._goal_for_role(attack_role))
                for rank, attack in enumerate(output.ranked_attacks[: self.config.oracle_top_k], start=1):
                    attack_hash = attack.hash()
                    attack_key = (attack_hash, attack_role)
                    attack_id = attack_by_hash_role.get(attack_key)
                    if attack_id is None:
                        attack_counter += 1
                        attack_id = f"atk-{attack_counter:05d}"
                        attack_by_hash_role[attack_key] = attack_id
                    predicted = output.predicted_scores[rank - 1] if rank - 1 < len(output.predicted_scores) else None
                    simulated = output.simulated_scores[rank - 1] if rank - 1 < len(output.simulated_scores) else None
                    candidate_rows.append(
                        {
                            "round_id": self.config.round_id,
                            "defense_id": defense_id,
                            "attack_id": attack_id,
                            "attack_role": attack_role,
                            "defense_role": defense_details.get(defense_id, {}).get("role"),
                            "rank": rank,
                            "attack_hash": attack_hash,
                            "attack_plan": attack,
                            "defense_hash": defense.hash(),
                            "target_kind": "mask_observation",
                            "belief_candidates": output.belief.feasible_count_estimate,
                            "belief_entropy": output.belief.entropy,
                            "belief_top1_top2_gap": output.belief.top1_top2_gap,
                            "belief_domain_stats": output.belief.domain_stats,
                            "predicted_score": predicted,
                            "surrogate_score": simulated,
                            "candidate_sources": int(output.explanation.get("candidate_sources", "0")),
                            "belief_ranker_applied": int(dict(output.belief.domain_stats).get("ranker_applied", 0.0)),
                            "belief_ranker_checkpoint": belief_ranker_checkpoint,
                            "attack_risk_report": output.risk_report,
                        }
                    )
                    pairs.append((attack_id, attack, defense_id, defense, attack_role))

        records = self.evaluator.evaluate_pairs(
            [(attack_id, attack, defense_id, defense) for attack_id, attack, defense_id, defense, _attack_role in pairs],
            job_prefix=self.config.round_id,
            base_seed=self.config.seed,
            metadata={"round_id": self.config.round_id},
        )
        attack_records, defense_records = self._update_league(pairs, records, defense_details)
        self._apply_retention()
        active_query_rows = self._schedule_active_queries(candidate_rows)
        self._write_artifacts(out_dir, candidate_rows, pairs, records, attack_records, defense_records, active_query_rows)
        best_attack_success = max((record.attack_success for record in records), default=0.0)
        defense_break_rates = _defense_break_rates(records)
        worst_defense_break_rate = max(defense_break_rates.values(), default=0.0)
        summary = LeagueRoundSummary(
            round_id=self.config.round_id,
            out_dir=str(out_dir),
            defenses=len(defenses),
            candidates=len(candidate_rows),
            oracle_pairs=len(records),
            oracle_requests=sum(len(record.oracle_request_ids) for record in records),
            best_attack_success=best_attack_success,
            worst_defense_break_rate=worst_defense_break_rate,
        )
        (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._write_run_metadata(out_dir, summary)
        return summary

    def _update_league(
        self,
        pairs: list[tuple[str, AttackPlan, str, DefensePlan, str]],
        records: list[OracleEvaluationRecord],
        defense_details: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pair_by_ids = {
            (attack_id, defense_id): (attack, defense, attack_role)
            for attack_id, attack, defense_id, defense, attack_role in pairs
        }
        defense_strength = {defense_id: 1.0 - rate for defense_id, rate in _defense_break_rates(records).items()}
        attack_rows: list[dict[str, Any]] = []
        defense_rows: list[dict[str, Any]] = []
        seen_attacks: set[str] = set()
        seen_defenses: set[str] = set()
        for record in records:
            attack, defense, attack_role = pair_by_ids[(record.attack_id, record.defense_id)]
            if record.attack_id not in seen_attacks:
                seen_attacks.add(record.attack_id)
                league_record = self.league.add_attack(
                    attack,
                    role=attack_role,
                    source=f"attack_oracle:{attack_role}",
                    strength=record.attack_success,
                )
                attack_rows.append(
                    {
                        "attack_id": record.attack_id,
                        "league_id": league_record.strategy_id,
                        "role": attack_role,
                        "attack_hash": record.attack_hash,
                        "strength": record.attack_success,
                    }
                )
            if record.defense_id not in seen_defenses:
                seen_defenses.add(record.defense_id)
                strength = defense_strength.get(record.defense_id, 0.0)
                details = defense_details.get(record.defense_id, {})
                defense_role = str(details.get("role") or "main")
                league_record = self.league.add_defense(
                    defense,
                    role=defense_role,
                    source=str(details.get("source") or defense.source),
                    strength=strength,
                )
                defense_rows.append(
                    {
                        "defense_id": record.defense_id,
                        "league_id": league_record.strategy_id,
                        "role": defense_role,
                        "defense_hash": record.defense_hash,
                        "defense_plan": defense,
                        "source": details.get("source") or defense.source,
                        "defense_role": defense_role,
                        "strength": strength,
                        "break_rate": 1.0 - strength,
                        "estimated_attack_success": details.get("estimated_attack_success"),
                        "ambiguity_score": details.get("ambiguity_score"),
                        "hidden_count": details.get("hidden_count", sum(sum(row) for row in defense.mask)),
                        "defense_oracle_explanation": details.get("defense_oracle_explanation", {}),
                        "defense_risk_report": details.get("defense_risk_report", {}),
                    }
                )
            self.league.record_payoff(
                record.attack_id,
                record.defense_id,
                attack_success=record.attack_success,
                games=len(record.round_win_rates),
            )
        return attack_rows, defense_rows

    def _attack_meta_for_defense_role(self, role: str) -> tuple[tuple[AttackPlan, float], ...]:
        if role == "exploiter":
            strongest = self.league.strongest_plans("attack", limit=1)
            return tuple((plan, weight) for plan, weight in strongest)
        return self.league.mixed_meta_plans("attack", limit=16)

    def _apply_retention(self) -> None:
        if self.config.attack_pool_max_active is not None:
            self.league.apply_retention(
                "attack",
                max_active=self.config.attack_pool_max_active,
                historical_keep=self.config.historical_keep,
            )
        if self.config.defense_pool_max_active is not None:
            self.league.apply_retention(
                "defense",
                max_active=self.config.defense_pool_max_active,
                historical_keep=self.config.historical_keep,
            )

    def _goal_for_role(self, role: str) -> GenerationGoal:
        if role == "underdog":
            return GenerationGoal(target_power_ratio=self.config.underdog_power_ratio, diversity_weight=0.1)
        if role == "exploiter":
            return GenerationGoal(target_power_ratio=1.0, explore_beta=0.2, diversity_weight=0.05)
        return GenerationGoal(target_power_ratio=1.0, explore_beta=0.0, diversity_weight=0.05)

    def _schedule_active_queries(self, candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queries: list[Query] = []
        candidate_by_query_id: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(candidate_rows, start=1):
            attack_role = str(row.get("attack_role") or "main")
            entropy = float(row.get("belief_entropy") or 0.0)
            gap = float(row.get("belief_top1_top2_gap") or 0.0)
            decision_impact = 1.0 / (1.0 + max(gap, 0.0))
            query_type = "underdog" if attack_role == "underdog" else "mask_observation"
            query_id = f"{self.config.round_id}-q{index:06d}"
            candidate_by_query_id[query_id] = row
            queries.append(
                Query(
                    query_id=query_id,
                    query_type=query_type,
                    info_gain=entropy,
                    decision_impact=decision_impact,
                    meta_frequency=1.0 if attack_role == "main" else 0.6 if attack_role == "exploiter" else 0.3,
                    novelty=0.5 if attack_role in {"exploiter", "underdog"} else 0.1,
                    underdog_potential=1.0 if attack_role == "underdog" else 0.0,
                    cost=float(self.config.teams),
                )
            )
        sim_keep = min(self.config.active_sim_keep, len(queries))
        real_keep = min(self.config.active_real_keep, max(0, len(queries) - sim_keep))
        scheduled = ActivePerceptionScheduler().schedule(tuple(queries), sim_keep=sim_keep, real_keep=real_keep)
        score_by_id = dict(scheduled.scores)
        rows: list[dict[str, Any]] = []
        for queue_name, queue in (("sim", scheduled.sim_queue), ("real", scheduled.real_query_queue)):
            for query in queue:
                candidate = candidate_by_query_id.get(query.query_id, {})
                rows.append(
                    {
                        "queue": queue_name,
                        "query_id": query.query_id,
                        "query_type": query.query_type,
                        "attack_id": candidate.get("attack_id"),
                        "defense_id": candidate.get("defense_id"),
                        "attack_role": candidate.get("attack_role"),
                        "defense_role": candidate.get("defense_role"),
                        "rank": candidate.get("rank"),
                        "predicted_score": candidate.get("predicted_score"),
                        "surrogate_score": candidate.get("surrogate_score"),
                        "belief_entropy": candidate.get("belief_entropy"),
                        "belief_top1_top2_gap": candidate.get("belief_top1_top2_gap"),
                        "score": score_by_id.get(query.query_id, 0.0),
                        "info_gain": query.info_gain,
                        "decision_impact": query.decision_impact,
                        "meta_frequency": query.meta_frequency,
                        "novelty": query.novelty,
                        "underdog_potential": query.underdog_potential,
                        "cost": query.cost,
                    }
                )
        return rows

    def _write_artifacts(
        self,
        out_dir: Path,
        candidate_rows: list[dict[str, Any]],
        pairs: list[tuple[str, AttackPlan, str, DefensePlan, str]],
        records: list[OracleEvaluationRecord],
        attack_records: list[dict[str, Any]],
        defense_records: list[dict[str, Any]],
        active_query_rows: list[dict[str, Any]],
    ) -> None:
        _write_jsonl(out_dir / "candidates.jsonl", candidate_rows)
        _write_jsonl(out_dir / "oracle_requests.jsonl", [request for record in records for request in record.requests])
        _write_jsonl(out_dir / "oracle_results.jsonl", [result for record in records for result in record.results])
        _write_jsonl(out_dir / "oracle_pairs.jsonl", [_oracle_pair_row(record) for record in records])
        _write_jsonl(out_dir / "scored_attacks.jsonl", attack_records)
        _write_jsonl(out_dir / "scored_defenses.jsonl", defense_records)
        _write_jsonl(out_dir / "active_queries.jsonl", active_query_rows)
        self._write_core_tables(out_dir, candidate_rows, pairs, records)
        state = {
            "iteration": self.league.iteration,
            "attack_pool": [_jsonable(record) for _plan, record in self.league.attack_pool.values()],
            "defense_pool": [_jsonable(record) for _plan, record in self.league.defense_pool.values()],
            "payoffs": [_jsonable(entry) for entry in self.league.payoffs.values()],
        }
        (out_dir / "league_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_core_tables(
        self,
        out_dir: Path,
        candidate_rows: list[dict[str, Any]],
        pairs: list[tuple[str, AttackPlan, str, DefensePlan, str]],
        records: list[OracleEvaluationRecord],
    ) -> None:
        table_dir = out_dir / "tables"
        observations = []
        seen_observations: set[str] = set()
        defense_by_id = {defense_id: defense for _attack_id, _attack, defense_id, defense, _role in pairs}
        for row in candidate_rows:
            defense = defense_by_id.get(str(row.get("defense_id") or ""))
            if defense is None:
                continue
            observation = observe_defense(defense)
            if observation.hash() in seen_observations:
                continue
            seen_observations.add(observation.hash())
            observations.append(
                ObservationTableRecord.from_observation(
                    observation,
                    belief_candidate_count=int(row.get("belief_candidates", 0) or 0),
                    belief_entropy=float(row.get("belief_entropy", 0.0) or 0.0),
                )
            )
        plan_rows = []
        single_rows = []
        pair_lookup = {
            (attack_id, defense_id): (attack, defense)
            for attack_id, attack, defense_id, defense, _attack_role in pairs
        }
        for record in records:
            attack, defense = pair_lookup[(record.attack_id, record.defense_id)]
            plan_rows.append(
                PlanMatchTableRecord.from_plan_match(
                    attack,
                    defense,
                    sim_or_real="sim",
                    num_games=len(record.round_win_rates),
                    round_win_rates=record.round_win_rates,
                    simulator_version=self.evaluator.simulator_version,
                    model_version=_round_model_version(self.config),
                )
            )
            for lane_idx, win_rate in enumerate(record.round_win_rates):
                games = 1
                single_rows.append(
                    SingleMatchupTableRecord.from_matchup(
                        attack.teams[lane_idx],
                        defense.teams[lane_idx],
                        sim_or_real="sim",
                        num_games=games,
                        wins=int(round(float(win_rate) * games)),
                        mean_duration=0.0,
                        mean_margin=0.0,
                        simulator_version=self.evaluator.simulator_version,
                        model_version=_round_model_version(self.config),
                    )
                )
        strategy_rows = [
            LeagueStrategyTableRecord.from_strategy(record)
            for _plan, record in tuple(self.league.attack_pool.values()) + tuple(self.league.defense_pool.values())
        ]
        write_table_jsonl(
            table_dir / "loadouts.jsonl",
            (LoadoutTableRecord.from_loadout(loadout, data_version="round_loadout_pool", season="unknown") for loadout in self.loadout_pool),
        )
        write_table_jsonl(table_dir / "observations.jsonl", observations)
        write_table_jsonl(table_dir / "single_matchups.jsonl", single_rows)
        write_table_jsonl(table_dir / "plan_matches.jsonl", plan_rows)
        write_table_jsonl(table_dir / "league_strategies.jsonl", strategy_rows)

    def _write_run_metadata(self, out_dir: Path, summary: LeagueRoundSummary) -> None:
        output_names = (
            "summary.json",
            "candidates.jsonl",
            "oracle_requests.jsonl",
            "oracle_results.jsonl",
            "oracle_pairs.jsonl",
            "scored_attacks.jsonl",
            "scored_defenses.jsonl",
            "active_queries.jsonl",
            "league_state.json",
            "tables/loadouts.jsonl",
            "tables/observations.jsonl",
            "tables/single_matchups.jsonl",
            "tables/plan_matches.jsonl",
            "tables/league_strategies.jsonl",
        )
        input_paths = [
            self.config.attack_proposal_checkpoint,
            self.config.defense_proposal_checkpoint,
            self.config.belief_ranker_checkpoint,
            self.config.belief_ranker_registry,
        ]
        metadata = ResultMetadata(
            model_version=_round_model_version(self.config),
            data_version=f"loadouts:{canonical_hash(self.loadout_pool)}",
            simulator_version=self.evaluator.simulator_version,
            league_iteration=self.league.iteration,
            random_seed=self.config.seed,
            generation_config_hash=hash_generation_config(_jsonable(self.config)),
            calibration_version="none",
        )
        manifest = RunMetadataManifest.from_result_metadata(
            run_id=self.config.round_id,
            metadata=metadata,
            created_at=0.0,
            code_version="local",
            input_artifacts=_artifact_refs(input_paths, role="input"),
            output_artifacts=_artifact_refs((out_dir / name for name in output_names), role="output"),
            metrics={
                "defenses": summary.defenses,
                "candidates": summary.candidates,
                "oracle_pairs": summary.oracle_pairs,
                "oracle_requests": summary.oracle_requests,
                "best_attack_success": summary.best_attack_success,
                "worst_defense_break_rate": summary.worst_defense_break_rate,
            },
            extra={"runner": "LeagueRoundRunner"},
        )
        write_run_metadata_manifest(manifest, out_dir / "run_metadata.json")


def _attack_config(config: LeagueRoundConfig) -> AttackOracleConfig:
    diversity_keep = min(config.attacks_per_defense, max(config.oracle_top_k * 4, config.oracle_top_k))
    first_keep = min(diversity_keep, max(config.oracle_top_k * 2, config.oracle_top_k))
    return AttackOracleConfig(
        candidate_count=config.attacks_per_defense,
        diversity_keep=diversity_keep,
        final_keep=config.oracle_top_k,
        underdog_residual_weight=config.underdog_residual_weight,
        use_future_feasibility_mask=config.use_future_feasibility_mask,
        halving_stages=(HalvingStage(games_each=2, keep=first_keep), HalvingStage(games_each=5, keep=config.oracle_top_k)),
    )


def _attack_candidate_sources(config: LeagueRoundConfig):
    if config.attack_proposal_checkpoint is None:
        return ()
    return (
        load_attack_proposal_candidate_source(
            config.attack_proposal_checkpoint,
            beam_size=config.attack_proposal_beam_size,
            use_future_feasibility=config.use_future_feasibility_mask,
            device=config.attack_proposal_device,
        ),
    )


def _defense_roster_sources(config: LeagueRoundConfig):
    if config.defense_proposal_checkpoint is None:
        return ()
    return (
        load_defense_proposal_candidate_source(
            config.defense_proposal_checkpoint,
            beam_size=config.defense_proposal_beam_size,
            use_future_feasibility=config.use_future_feasibility_mask,
            device=config.defense_proposal_device,
        ),
    )


def _belief_engine_from_config(config: LeagueRoundConfig, engine: ConstraintEngine) -> tuple[BeliefEngine, str | None]:
    checkpoint_path = config.belief_ranker_checkpoint
    if checkpoint_path is None and config.belief_ranker_registry is not None:
        record = select_best_checkpoint(
            config.belief_ranker_registry,
            metric=config.belief_ranker_metric,
            mode=config.belief_ranker_metric_mode,
            model_type="belief_ranker",
            dataset_hash=config.belief_ranker_dataset_hash,
        )
        checkpoint_path = record.model_path
    real_meta_db = None
    if config.use_real_calibration and config.real_meta_db_jsonl is not None:
        real_meta_db = RealMetaDB.load(config.real_meta_db_jsonl)
    if checkpoint_path is None:
        return (
            BeliefEngine(
                engine,
                real_meta_db=real_meta_db,
                use_equipment_star_features=config.use_equipment_star_features,
                use_position_features=config.use_position_features,
            ),
            None,
        )
    ranker = load_belief_ranker_checkpoint(checkpoint_path, device=config.belief_ranker_device)
    return (
        BeliefEngine(
            engine,
            real_meta_db=real_meta_db,
            ranker=ranker,
            ranker_weight=config.belief_ranker_weight,
            use_equipment_star_features=config.use_equipment_star_features,
            use_position_features=config.use_position_features,
        ),
        str(checkpoint_path),
    )


def _surrogate_from_config(config: LeagueRoundConfig) -> HeuristicSurrogateScorer:
    return HeuristicSurrogateScorer(
        use_equipment_star_features=config.use_equipment_star_features,
        use_position_features=config.use_position_features,
    )


def _defense_attack_config(config: LeagueRoundConfig) -> AttackOracleConfig:
    keep = max(1, min(8, config.oracle_top_k))
    first_keep = max(keep, min(12, max(config.attacks_per_defense // 4, keep)))
    return AttackOracleConfig(
        candidate_count=max(keep, min(64, config.attacks_per_defense)),
        diversity_keep=first_keep,
        final_keep=keep,
        underdog_residual_weight=config.underdog_residual_weight,
        use_future_feasibility_mask=config.use_future_feasibility_mask,
        halving_stages=(HalvingStage(games_each=2, keep=first_keep), HalvingStage(games_each=4, keep=keep)),
    )


def _defense_break_rates(records: list[OracleEvaluationRecord]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.defense_id, []).append(record.attack_success)
    return {defense_id: max(values) for defense_id, values in grouped.items() if values}


def _oracle_pair_row(record: OracleEvaluationRecord) -> dict[str, Any]:
    return {
        "attack_id": record.attack_id,
        "defense_id": record.defense_id,
        "attack_hash": record.attack_hash,
        "defense_hash": record.defense_hash,
        "attack_success": record.attack_success,
        "round_win_rates": record.round_win_rates,
        "oracle_request_ids": record.oracle_request_ids,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _artifact_refs(paths: Any, *, role: str) -> tuple[RunArtifactRef, ...]:
    refs: list[RunArtifactRef] = []
    for item in paths:
        if item is None:
            continue
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        refs.append(RunArtifactRef.from_path(path, kind=path.suffix.lstrip(".") or "file", role=role))
    return tuple(refs)


def _round_model_version(config: LeagueRoundConfig) -> str:
    values = tuple(
        str(path)
        for path in (
            config.attack_proposal_checkpoint,
            config.defense_proposal_checkpoint,
            config.belief_ranker_checkpoint,
        )
        if path is not None
    )
    return "|".join(values) if values else "none"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
