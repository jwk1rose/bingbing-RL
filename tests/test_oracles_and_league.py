from __future__ import annotations

from masked_team_league.active import ActivePerceptionScheduler, Query
from masked_team_league.attack_oracle import AttackOracle, AttackOracleConfig
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.defense_oracle import DefenseOracle, DefenseOracleConfig
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.hyperband import HalvingStage
from masked_team_league.league import LeagueManager
from masked_team_league.models import Team, DefensePlan, observe_defense
from masked_team_league.networks import AttackNetInput, LegalMaskedAttackGenerator
from masked_team_league.belief import BeliefEngine


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
