from __future__ import annotations

import json
from pathlib import Path

import pytest

from masked_team_league.league.active_perception import ActivePerceptionScheduler, Query
from masked_team_league.oracles.attack import AttackOracle, AttackOracleConfig
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.oracles.defense import DefenseOracle, DefenseOracleConfig
from masked_team_league.generation import GenerationGoal, LegalPlanGenerator
from masked_team_league.scoring import HalvingStage
from masked_team_league.league import LeagueManager
from masked_team_league.oracles.mask_search import MaskSearcher
from masked_team_league.domain import AttackPlan, Team, DefensePlan, canonical_hash, observe_defense
from masked_team_league.generation.legal_masked import AttackNetInput, LegalMaskedAttackGenerator
from masked_team_league.belief import BeliefEngine, BeliefOutput
from masked_team_league.real_platform.calibration import RealMetaDB, RealMetaRecord
from masked_team_league.scoring import SurrogatePrediction


def test_attack_oracle_returns_legal_reproducible_attacks(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = LegalPlanGenerator(loadouts, seed=3).generate_defense_plan(fmt3)
    config = AttackOracleConfig(
        candidate_count=32,
        diversity_keep=12,
        final_keep=3,
        halving_stages=(HalvingStage(3, 6), HalvingStage(8, 3)),
    )
    oracle_a = AttackOracle(loadout_pool=loadouts, constraint_engine=engine, seed=11, config=config)
    oracle_b = AttackOracle(loadout_pool=loadouts, constraint_engine=engine, seed=11, config=config)
    out_a = oracle_a.search(defense)
    out_b = oracle_b.search(defense)
    assert out_a.ranked_attacks
    assert [plan.hash() for plan in out_a.ranked_attacks] == [plan.hash() for plan in out_b.ranked_attacks]
    assert all(engine.is_legal_attack(plan) for plan in out_a.ranked_attacks)
    assert "belief_candidates" in out_a.explanation


def test_appendix_g_attack_oracle_full_defense_returns_legal_top_five(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = LegalPlanGenerator(loadouts, seed=3111).generate_defense_plan(fmt3)
    oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=3112,
        config=AttackOracleConfig(
            candidate_count=40,
            diversity_keep=16,
            final_keep=5,
            halving_stages=(HalvingStage(2, 12), HalvingStage(4, 5)),
        ),
    )

    output = oracle.search(defense)

    assert len(output.ranked_attacks) == 5
    assert all(engine.is_legal_attack(plan) for plan in output.ranked_attacks)


def test_appendix_g_mask_observation_belief_candidates_match_visible_slots(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = LegalPlanGenerator(loadouts, seed=3121).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 1, 0, 1, 0), (1, 0, 0, 0, 0)),
    )
    observation = observe_defense(defense)
    oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=3122,
        config=AttackOracleConfig(
            candidate_count=20,
            diversity_keep=8,
            final_keep=2,
            halving_stages=(HalvingStage(2, 4), HalvingStage(4, 2)),
        ),
    )

    output = oracle.search(observation)

    assert output.belief.candidates
    assert all(_roster_matches_observation(roster, observation) for roster in output.belief.candidates)


def test_attack_oracle_risk_report_includes_per_lane_worst_case_and_backups(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = LegalPlanGenerator(loadouts, seed=36).generate_defense_plan(fmt3)
    oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=37,
        config=AttackOracleConfig(
            candidate_count=24,
            diversity_keep=8,
            final_keep=2,
            halving_stages=(HalvingStage(2, 4), HalvingStage(4, 2)),
        ),
    )

    output = oracle.search(defense)
    report = output.risk_report

    assert report["best_attack_hash"] == output.ranked_attacks[0].hash()
    assert len(report["expected_lane_win_rates"]) == fmt3.n_teams
    assert len(report["worst_case_lane_win_rates"]) == fmt3.n_teams
    assert 0.0 <= report["worst_case_match_win"] <= report["expected_match_win"] <= 1.0
    assert report["backup_attack_count"] == len(output.ranked_attacks) - 1
    assert len(report["backup_match_wins"]) == len(output.ranked_attacks) - 1


def test_attack_oracle_underdog_residual_prefers_lower_cost_equal_win_candidate(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = DefensePlan(
        fmt3,
        (Team(loadouts[20:25]), Team(loadouts[25:30]), Team(loadouts[30:35])),
        ((0, 0, 0, 0, 0),) * 3,
        "target",
    )
    low_cost_attack = AttackPlan(fmt3, (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15])), "low")
    high_cost_attack = AttackPlan(fmt3, (Team(loadouts[25:30]), Team(loadouts[30:35]), Team(loadouts[35:40])), "high")

    def source(**_kwargs):
        return (high_cost_attack, low_cost_attack)

    oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        surrogate=_ConstantSurrogate(),
        candidate_sources=(source,),
        seed=502,
        config=AttackOracleConfig(
            candidate_count=0,
            diversity_keep=2,
            final_keep=2,
            diversity_weight=0.0,
            underdog_residual_weight=1.0,
            halving_stages=(HalvingStage(3, 2),),
        ),
    )

    output = oracle.search(defense, goal=GenerationGoal(target_power_ratio=0.9))

    assert output.ranked_attacks[0].hash() == low_cost_attack.hash()
    assert output.risk_report["underdog_gap"] > 0.0
    assert output.risk_report["underdog_residual_bonus"] > 0.0
    assert output.explanation["underdog_residual_bonus"] != "0.0000"


def test_attack_oracle_annotates_no_legal_belief_candidate_failure(loadouts, fmt3):
    class EmptyBeliefEngine:
        def build(self, _observation, *, max_k=64):
            return BeliefOutput(
                candidates=(),
                weights=(),
                entropy=0.0,
                feasible_count_estimate=0,
                top1_top2_gap=0.0,
                domain_stats=(("domain_1:1", 0.0),),
            )

    defense = LegalPlanGenerator(loadouts, seed=3001).generate_defense_plan(
        fmt3,
        mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    oracle = AttackOracle(
        loadout_pool=loadouts,
        belief_engine=EmptyBeliefEngine(),
        seed=3002,
        config=AttackOracleConfig(candidate_count=8, diversity_keep=4, final_keep=1, halving_stages=(HalvingStage(1, 1),)),
    )

    output = oracle.search(observe_defense(defense))
    payload = output.to_json_dict()

    assert output.risk_report["failure_code"] == "NO_LEGAL_BELIEF_CANDIDATES"
    assert output.risk_report["failure_stage"] == "belief"
    assert output.risk_report["belief_feasible_count"] == 0
    assert output.risk_report["domain_stats"] == [("domain_1:1", 0.0)]
    assert payload["diagnostics"][0]["code"] == "NO_LEGAL_BELIEF_CANDIDATES"
    assert payload["diagnostics"][0]["stage"] == "belief"


def test_attack_oracle_annotates_no_legal_attack_candidate_failure(loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=3003).generate_defense_plan(fmt3)
    oracle = AttackOracle(
        loadout_pool=loadouts,
        seed=3004,
        config=AttackOracleConfig(candidate_count=0, diversity_keep=4, final_keep=1, halving_stages=(HalvingStage(1, 1),)),
    )

    output = oracle.search(defense)
    payload = output.to_json_dict()

    assert output.risk_report["failure_code"] == "NO_LEGAL_ATTACK_CANDIDATES"
    assert output.risk_report["failure_stage"] == "candidate_generation"
    assert output.risk_report["generated_candidate_count"] == 0
    assert output.risk_report["legal_candidate_count"] == 0
    assert output.risk_report["external_candidate_source_count"] == 0
    assert payload["diagnostics"][0]["code"] == "NO_LEGAL_ATTACK_CANDIDATES"
    assert payload["diagnostics"][0]["stage"] == "candidate_generation"


def test_attack_oracle_accepts_external_candidate_source(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    defense = LegalPlanGenerator(loadouts, seed=33).generate_defense_plan(fmt3)
    proposal_attack = LegalPlanGenerator(loadouts, seed=34).generate_attack_plan(fmt3, source="attack_proposal")
    seen = {}

    def candidate_source(**kwargs):
        seen.update(kwargs)
        return (proposal_attack,)

    oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        candidate_sources=(candidate_source,),
        seed=35,
        config=AttackOracleConfig(
            candidate_count=0,
            diversity_keep=4,
            final_keep=1,
            halving_stages=(HalvingStage(1, 1),),
        ),
    )

    output = oracle.search(defense)

    assert output.ranked_attacks == (proposal_attack,)
    assert seen["target"] == defense
    assert seen["match_format"] == fmt3
    assert seen["belief"].feasible_count_estimate == 1
    assert "candidate_sources" in output.explanation


def test_attack_oracle_accepts_mask_observation(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "test")
    observation = observe_defense(defense)
    oracle = AttackOracle(
        loadout_pool=loadouts,
        seed=13,
        config=AttackOracleConfig(candidate_count=24, diversity_keep=8, final_keep=2, halving_stages=(HalvingStage(3, 4),)),
    )
    output = oracle.search(observation)
    assert output.ranked_attacks
    assert output.belief.feasible_count_estimate > 0
    assert float(output.explanation["belief_entropy"]) >= 0.0


def test_belief_uses_real_meta_frequency_and_recency(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "real-meta")
    observation = observe_defense(defense)
    attack_plan = AttackPlan(fmt3, teams, "unit")
    db = RealMetaDB()
    for offset in range(3):
        db.add(
            RealMetaRecord.from_match(
                observation=observation,
                full_defense_if_available=defense,
                attack_plan=attack_plan,
                lane_results=(1.0, 0.0, 1.0),
                match_result=1.0,
                rank_segment="top",
                server="unit",
                season="S28",
                timestamp=1000.0 + offset,
            )
        )

    belief = BeliefEngine(ConstraintEngine(loadouts), real_meta_db=db, now=1005.0).build(observation, max_k=16)

    assert belief.candidates[0] == defense.teams
    assert ("real_candidate_count", 1.0) in belief.domain_stats
    assert ("real_record_count", 3.0) in belief.domain_stats


def test_belief_uses_similar_real_meta_when_exact_observation_is_missing(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    record_defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "real-similar")
    query_defense = DefensePlan(fmt3, teams, ((0, 1, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "query")
    record_observation = observe_defense(record_defense)
    query_observation = observe_defense(query_defense)
    attack_plan = AttackPlan(fmt3, teams, "unit")
    db = RealMetaDB()
    db.add(
        RealMetaRecord.from_match(
            observation=record_observation,
            full_defense_if_available=record_defense,
            attack_plan=attack_plan,
            lane_results=(0.0, 1.0, 0.0),
            match_result=0.0,
            rank_segment="top",
            server="unit",
            season="S28",
            timestamp=1000.0,
        )
    )

    belief = BeliefEngine(
        ConstraintEngine(loadouts),
        real_meta_db=db,
        now=1005.0,
        real_frequency_alpha=6.0,
        similarity_eta=2.0,
    ).build(query_observation, max_k=16)
    stats = dict(belief.domain_stats)

    assert record_observation.hash() != query_observation.hash()
    assert belief.candidates[0] == query_defense.teams
    assert stats["real_exact_record_count"] == 0.0
    assert stats["real_similar_record_count"] == 1.0
    assert stats["real_similarity_mean"] > 0.5
    assert stats["real_match_result_mean"] == 0.0


def test_belief_uses_compatible_defense_pool_candidates(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "defense-pool")
    observation = observe_defense(defense)

    belief = BeliefEngine(ConstraintEngine(loadouts), defense_pool=(defense,)).build(observation, max_k=16)

    assert belief.candidates[0] == defense.teams
    assert ("defense_pool_candidate_count", 1.0) in belief.domain_stats
    assert ("defense_pool_record_count", 1.0) in belief.domain_stats


def test_belief_ranker_can_prioritize_candidate_completion(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "ranker")
    observation = observe_defense(defense)
    target_hash = canonical_hash(defense.teams)
    seen_features = []

    def ranker(_observation, roster, features):
        seen_features.append(features)
        return 5.0 if canonical_hash(roster) == target_hash else -5.0

    belief = BeliefEngine(ConstraintEngine(loadouts), ranker=ranker, ranker_weight=1000.0).build(observation, max_k=16)

    assert belief.candidates[0] == defense.teams
    assert seen_features
    assert {"roster_strength", "real_frequency", "pool_frequency", "compatible_visible_ratio"} <= set(seen_features[0])
    assert ("ranker_applied", 1.0) in belief.domain_stats


def test_belief_domain_stats_include_aggregate_domain_and_weight_metrics(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 1, 0, 0, 1), (0, 0, 0, 0, 0)), "domain-stats")
    observation = observe_defense(defense)

    belief = BeliefEngine(ConstraintEngine(loadouts)).build(observation, max_k=16)
    stats = dict(belief.domain_stats)

    assert stats["hidden_slot_count"] == 4.0
    assert stats["domain_count_min"] <= stats["domain_count_mean"] <= stats["domain_count_max"]
    assert stats["domain_count_entropy"] >= 0.0
    assert stats["candidate_count"] == float(belief.feasible_count_estimate)
    assert stats["top1_weight"] >= stats["top2_weight"] >= 0.0
    assert 0.0 <= stats["weight_entropy_normalized"] <= 1.0


def test_defense_oracle_returns_legal_defense(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=5,
        config=AttackOracleConfig(candidate_count=16, diversity_keep=6, final_keep=1, halving_stages=(HalvingStage(2, 3),)),
    )
    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        seed=5,
        config=DefenseOracleConfig(roster_candidates=4, masks_per_roster=2, max_masks_per_roster=16),
    )
    output = defense_oracle.search(fmt3)
    assert output.best_defense is not None
    assert engine.is_legal_defense(output.best_defense)
    assert "estimated_attack_success" in output.explanation
    assert output.risk_report["best_defense_hash"] == output.best_defense.hash()
    assert output.risk_report["backup_defense_count"] == len(output.backup_defenses)
    assert 0.0 <= output.risk_report["estimated_break_rate"] <= 1.0
    assert "worst_case_attack_hash" in output.risk_report


def test_appendix_g_defense_oracle_reports_counter_attack_success(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=3131,
        config=AttackOracleConfig(candidate_count=16, diversity_keep=6, final_keep=1, halving_stages=(HalvingStage(2, 3),)),
    )
    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        seed=3132,
        config=DefenseOracleConfig(roster_candidates=3, masks_per_roster=1, max_masks_per_roster=8),
    )

    output = defense_oracle.search(fmt3)

    assert output.best_defense is not None
    assert "estimated_attack_success" in output.explanation
    assert 0.0 <= float(output.explanation["estimated_attack_success"]) <= 1.0
    assert output.risk_report["estimated_break_rate"] == pytest.approx(
        float(output.explanation["estimated_attack_success"]),
        abs=1e-4,
    )
    assert output.risk_report["worst_case_attack_hash"]


def test_defense_oracle_underdog_residual_prefers_lower_cost_equal_survival_roster(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_meta_plan = AttackPlan(
        fmt3,
        (Team(loadouts[25:30]), Team(loadouts[30:35]), Team(loadouts[35:40])),
        "meta",
    )
    low_cost_defense = DefensePlan(
        fmt3,
        (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15])),
        ((0, 0, 0, 0, 0),) * 3,
        "low",
    )
    high_cost_defense = DefensePlan(
        fmt3,
        (Team(loadouts[25:30]), Team(loadouts[30:35]), Team(loadouts[35:40])),
        ((0, 0, 0, 0, 0),) * 3,
        "high",
    )

    def roster_source(**_kwargs):
        return (high_cost_defense, low_cost_defense)

    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        surrogate=_ConstantSurrogate(),
        seed=602,
        config=AttackOracleConfig(
            candidate_count=1,
            diversity_keep=1,
            final_keep=1,
            halving_stages=(HalvingStage(3, 1),),
        ),
    )
    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        roster_sources=(roster_source,),
        seed=603,
            config=DefenseOracleConfig(
                roster_candidates=2,
                masks_per_roster=1,
                max_masks_per_roster=1,
                underdog_residual_weight=2.0,
            ),
        )

    output = defense_oracle.search(
        fmt3,
        attack_meta=((attack_meta_plan, 1.0),),
        goal=GenerationGoal(target_power_ratio=0.9),
    )

    assert output.best_defense is not None
    assert output.best_defense.teams == low_cost_defense.teams
    assert output.risk_report["hidden_count"] > 0
    assert output.risk_report["underdog_defense_gap"] > 0.0
    assert output.risk_report["underdog_residual_bonus"] > 0.0
    assert output.explanation["underdog_residual_bonus"] != "0.0000"


def test_mask_searcher_uses_learned_slot_score_provider(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    roster = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))

    def learned_scores(_roster, _match_format):
        return (
            (100.0, 0.0, 0.0, 0.0, 90.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )

    searcher = MaskSearcher(engine, slot_score_provider=learned_scores, learned_score_weight=1.0)
    best_mask, _score, stats = searcher.search(fmt3, roster, keep=1, max_masks=None)[0]

    assert best_mask[0][0] == 1
    assert best_mask[0][4] == 1
    assert stats["learned_mask_score"] == 190.0


def test_mask_searcher_explains_learned_hidden_slots(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    roster = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))

    def learned_scores(_roster, _match_format):
        return (
            (100.0, 0.0, 0.0, 0.0, 90.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )

    searcher = MaskSearcher(engine, slot_score_provider=learned_scores, learned_score_weight=1.0)
    best_mask, _score, stats = searcher.search(fmt3, roster, keep=1, max_masks=None)[0]

    assert best_mask[0][0] == 1
    assert stats["learned_slot_scores"][0][0] == 100.0
    assert stats["hidden_slot_explanations"][0]["team_index"] == 0
    assert stats["hidden_slot_explanations"][0]["slot_index"] == 0
    assert stats["hidden_slot_explanations"][0]["hero_id"] == loadouts[0].hero_id
    assert stats["hidden_slot_explanations"][0]["unique_equip_id"] == loadouts[0].unique_equip_id
    assert stats["hidden_slot_explanations"][0]["learned_slot_score"] == 100.0
    assert stats["top_learned_slots"][0]["slot_index"] == 0


def test_mask_searcher_max_masks_prioritizes_hidden_candidates(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    roster = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))

    searcher = MaskSearcher(engine)
    best_mask, _score, stats = searcher.search(fmt3, roster, keep=1, max_masks=1)[0]

    assert sum(sum(row) for row in best_mask) == min(
        fmt3.max_hidden_total,
        fmt3.n_teams * fmt3.max_hidden_per_team,
    )
    assert stats["hidden_count"] == float(sum(sum(row) for row in best_mask))


def test_defense_oracle_exposes_learned_mask_score(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=55,
        config=AttackOracleConfig(candidate_count=8, diversity_keep=4, final_keep=1, halving_stages=(HalvingStage(1, 3),)),
    )

    def learned_scores(_roster, _match_format):
        return (
            (100.0, 0.0, 0.0, 0.0, 90.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )

    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        mask_slot_score_provider=learned_scores,
        seed=55,
        config=DefenseOracleConfig(roster_candidates=1, masks_per_roster=1, max_masks_per_roster=None),
    )

    output = defense_oracle.search(fmt3)

    assert output.best_defense is not None
    assert output.best_defense.mask[0][0] == 1
    assert output.best_defense.mask[0][4] == 1
    assert float(output.explanation["learned_mask_score"]) > 0.0
    assert output.risk_report["learned_mask_score"] == float(output.explanation["learned_mask_score"])


def test_defense_oracle_risk_report_includes_mask_explanation(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=56,
        config=AttackOracleConfig(candidate_count=8, diversity_keep=4, final_keep=1, halving_stages=(HalvingStage(1, 3),)),
    )

    def learned_scores(_roster, _match_format):
        return (
            (100.0, 0.0, 0.0, 0.0, 90.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )

    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        mask_slot_score_provider=learned_scores,
        seed=56,
        config=DefenseOracleConfig(roster_candidates=1, masks_per_roster=1, max_masks_per_roster=None),
    )

    output = defense_oracle.search(fmt3)
    mask_explanation = output.risk_report["mask_explanation"]

    assert mask_explanation["learned_score_weight"] > 0.0
    assert mask_explanation["hidden_slot_explanations"][0]["learned_slot_score"] == 100.0
    assert mask_explanation["hidden_slot_explanations"][0]["hidden"] is True
    assert output.explanation["top_hidden_slots"].startswith("t0s0:")


def test_active_scheduler_and_league_pool(loadouts, fmt3):
    scheduler = ActivePerceptionScheduler()
    queries = (
        Query("q1", "matchup", 0.1, 0.1, 0.1, 0.1, 0.1, 1.0),
        Query("q2", "underdog", 0.5, 0.2, 0.1, 0.4, 0.8, 1.0),
    )
    scheduled = scheduler.schedule(queries, sim_keep=1)
    assert scheduled.sim_queue[0].query_id == "q2"

    generator = LegalPlanGenerator(loadouts, seed=8)
    attack = generator.generate_attack_plan(fmt3)
    defense = generator.generate_defense_plan(fmt3)
    league = LeagueManager()
    atk_record = league.add_attack(attack, role="main", source="test", strength=0.7)
    def_record = league.add_defense(defense, role="main", source="test", strength=0.3)
    league.record_payoff(atk_record.strategy_id, def_record.strategy_id, attack_success=0.6, games=10)
    assert league.meta_distribution("attack")[0][1] == 1.0


def test_appendix_g_active_scheduler_prioritizes_high_info_high_impact_query():
    scheduler = ActivePerceptionScheduler()
    queries = (
        Query("cheap-low-impact", "sim", 0.05, 0.05, 0.8, 0.0, 0.0, 0.0),
        Query("uncertain-top-decision", "sim", 0.9, 0.8, 0.2, 0.0, 0.0, 0.1),
        Query("novel-but-low-impact", "sim", 0.1, 0.05, 0.1, 1.0, 0.0, 0.1),
    )

    scheduled = scheduler.schedule(queries, sim_keep=1, real_keep=1)

    assert scheduled.sim_queue[0].query_id == "uncertain-top-decision"
    assert dict(scheduled.scores)["uncertain-top-decision"] > dict(scheduled.scores)["novel-but-low-impact"]


def test_league_manager_mixed_meta_distribution_uses_roles(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=18)
    league = LeagueManager()
    records = []
    for role, strength in (
        ("main", 0.80),
        ("exploiter", 0.65),
        ("historical", 0.40),
        ("underdog", 0.55),
    ):
        records.append(league.add_attack(generator.generate_attack_plan(fmt3, source=role), role=role, source="test", strength=strength))

    mixed = dict(league.mixed_meta_distribution("attack"))

    assert set(mixed) == {record.strategy_id for record in records}
    assert abs(sum(mixed.values()) - 1.0) < 1e-9
    assert mixed[records[2].strategy_id] > mixed[records[3].strategy_id]


def test_league_manager_assigns_diversity_clusters_and_applies_historical_retention(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=24)
    league = LeagueManager()
    records = []
    for index, strength in enumerate((0.10, 0.90, 0.20, 0.80, 0.30), start=1):
        league.iteration = index
        role = "historical" if index == 1 else "main"
        records.append(league.add_attack(generator.generate_attack_plan(fmt3, source=f"p{index}"), role=role, source="test", strength=strength))
    league.iteration = 8

    retained = league.apply_retention("attack", max_active=3, historical_keep=1)
    active_ids = {record.strategy_id for record in retained if record.active}
    mixed_ids = {strategy_id for strategy_id, _weight in league.mixed_meta_distribution("attack")}

    assert len(league.attack_pool) == 5
    assert len(active_ids) <= 3
    assert records[0].strategy_id in active_ids
    assert records[1].strategy_id in active_ids
    assert any(not record.active and record.retired_reason == "retention" for _plan, record in league.attack_pool.values())
    assert len({record.diversity_cluster for _plan, record in league.attack_pool.values()}) > 1
    assert mixed_ids <= active_ids


def test_appendix_g_league_ten_rounds_keep_attack_and_defense_clusters_diverse(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=3141)
    league = LeagueManager()

    for iteration in range(10):
        league.next_iteration()
        attack = generator.generate_attack_plan(fmt3, source=f"attack-round-{iteration}")
        defense = generator.generate_defense_plan(fmt3, source=f"defense-round-{iteration}")
        attack_record = league.add_attack(attack, role="main", source="appendix_g", strength=0.5 + iteration * 0.01)
        defense_record = league.add_defense(defense, role="main", source="appendix_g", strength=0.5 - iteration * 0.01)
        league.record_payoff(
            attack_record.strategy_id,
            defense_record.strategy_id,
            attack_success=0.45 + iteration * 0.01,
            games=3,
        )

    attack_clusters = {record.diversity_cluster for _plan, record in league.attack_pool.values()}
    defense_clusters = {record.diversity_cluster for _plan, record in league.defense_pool.values()}

    assert len(league.attack_pool) == 10
    assert len(league.defense_pool) == 10
    assert len(attack_clusters) > 1
    assert len(defense_clusters) > 1


def test_league_manager_exploiter_targets_use_payoff_strength(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=25)
    league = LeagueManager()
    weak_attack = league.add_attack(generator.generate_attack_plan(fmt3, source="weak"), role="historical", source="test", strength=0.20)
    strong_attack = league.add_attack(generator.generate_attack_plan(fmt3, source="strong"), role="main", source="test", strength=0.90)
    hard_defense = league.add_defense(generator.generate_defense_plan(fmt3, source="hard"), role="main", source="test", strength=0.10)
    soft_defense = league.add_defense(generator.generate_defense_plan(fmt3, source="soft"), role="main", source="test", strength=0.80)

    strongest_attack = league.strongest_plans("attack", limit=1)
    hardest_defense = league.hardest_defense_plans(limit=1)

    assert strongest_attack[0][0].hash() == league.attack_pool[strong_attack.strategy_id][0].hash()
    assert strongest_attack[0][1] == strong_attack.strength
    assert hardest_defense[0][0].hash() == league.defense_pool[hard_defense.strategy_id][0].hash()
    assert hardest_defense[0][1] == 1.0 - hard_defense.strength
    assert weak_attack.strategy_id in league.attack_pool
    assert soft_defense.strategy_id in league.defense_pool


def test_legal_masked_attack_generator_outputs_legal(loadouts, fmt3):
    defense = LegalPlanGenerator(loadouts, seed=2).generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    observation = observe_defense(defense)
    engine = ConstraintEngine(loadouts)
    belief = BeliefEngine(engine).build(observation)
    net = LegalMaskedAttackGenerator(loadouts, seed=2)
    output = net.generate(
        AttackNetInput(
            format=fmt3,
            observation=observation,
            belief_topk=belief,
            attack_loadout_pool=loadouts,
            selected_loadouts=(),
            used_hero_ids=frozenset(),
            used_unique_equip_ids=frozenset(),
            current_slot=(1, 1),
            goal=__import__("masked_team_league.generation", fromlist=["GenerationGoal"]).GenerationGoal(),
        ),
        count=4,
    )
    assert output.candidates
    assert all(engine.is_legal_attack(candidate) for candidate in output.candidates)
    assert all(output.legality_flags)


def test_defense_oracle_uses_attack_meta_in_explanation(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=21,
        config=AttackOracleConfig(candidate_count=16, diversity_keep=6, final_keep=1, halving_stages=(HalvingStage(2, 3),)),
    )
    attack_meta = ((LegalPlanGenerator(loadouts, seed=22).generate_attack_plan(fmt3, source="meta"), 1.0),)
    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        seed=23,
        config=DefenseOracleConfig(roster_candidates=3, masks_per_roster=1, max_masks_per_roster=8),
    )

    output = defense_oracle.search(fmt3, attack_meta=attack_meta)

    assert output.best_defense is not None
    assert output.explanation["attack_meta_count"] == "1"
    assert "meta_attack_success" in output.explanation


def test_oracle_outputs_have_stable_json_contracts(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    generator = LegalPlanGenerator(loadouts, seed=51)
    defense = generator.generate_defense_plan(fmt3, mask=((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    attack_oracle = AttackOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        seed=52,
        config=AttackOracleConfig(candidate_count=8, diversity_keep=4, final_keep=1, halving_stages=(HalvingStage(2, 2),)),
    )
    attack_output = attack_oracle.search(observe_defense(defense))
    defense_oracle = DefenseOracle(
        loadout_pool=loadouts,
        constraint_engine=engine,
        attack_oracle=attack_oracle,
        seed=53,
        config=DefenseOracleConfig(roster_candidates=2, masks_per_roster=1, max_masks_per_roster=4),
    )
    defense_output = defense_oracle.search(fmt3)

    attack_payload = attack_output.to_json_dict()
    defense_payload = defense_output.to_json_dict()

    assert attack_payload["schema_version"] == "attack_oracle_output.v1"
    assert attack_payload["module"] == "AttackOracle"
    assert attack_payload["metadata"]["random_seed"] == 52
    assert attack_payload["belief_summary"]["feasible_count_estimate"] >= 1
    assert "risk_report" in attack_payload
    assert defense_payload["schema_version"] == "defense_oracle_output.v1"
    assert defense_payload["module"] == "DefenseOracle"
    assert defense_payload["metadata"]["random_seed"] == 53
    assert "risk_report" in defense_payload
    json.dumps(attack_payload, sort_keys=True)
    json.dumps(defense_payload, sort_keys=True)


def test_module_output_contracts_doc_lists_oracle_schema_versions():
    text = Path("docs/module_output_contracts.md").read_text(encoding="utf-8")

    assert "attack_oracle_output.v1" in text
    assert "defense_oracle_output.v1" in text
    assert "metadata" in text
    assert "risk_report" in text
    assert "underdog_residual_bonus" in text


class _ConstantSurrogate:
    def predict(self, _attack, _defense):
        return SurrogatePrediction(
            win_prob=0.5,
            uncertainty=0.0,
            margin=0.0,
            duration=60.0,
            counter_residual=0.0,
        )


def _roster_matches_observation(roster, observation) -> bool:
    for team_idx, row in enumerate(observation.slots, start=1):
        for slot_idx, visible in enumerate(row, start=1):
            if visible.is_hidden:
                continue
            loadout = roster[team_idx - 1].slots[slot_idx - 1]
            if visible.hero_id != loadout.hero_id:
                return False
            if visible.unique_equip_id != loadout.unique_equip_id:
                return False
            if visible.unique_equip_star != loadout.unique_equip_star:
                return False
    return True
