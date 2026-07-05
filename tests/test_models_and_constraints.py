from __future__ import annotations

import builtins

import pytest

import masked_team_league.scoring.cache as cache_module
import masked_team_league.domain.hashing as model_module
from masked_team_league.scoring import MatchupCacheKey
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.generation import GenerationGoal, LegalPlanGenerator
from masked_team_league.domain import AttackPlan, DefensePlan, HeroRecord, Loadout, MatchFormat, Team, observe_defense


def test_loadout_requires_unique_equip_star(heroes):
    with pytest.raises(ValueError):
        Loadout.from_hero(heroes[0], unique_equip_star=None)


def test_unique_equipment_id_conflicts_even_when_star_differs(loadouts, fmt3):
    duplicate_equip = Loadout(
        hero_id=99,
        unique_equip_id=loadouts[0].unique_equip_id,
        unique_equip_star=3 if loadouts[0].unique_equip_star != 3 else 4,
        final_power=1200.0,
        standing_rank=99.0,
        standing_bucket="back",
    )
    team = Team((loadouts[0], loadouts[1], loadouts[2], loadouts[3], duplicate_equip))
    defense = DefensePlan(fmt3, (team, Team(loadouts[5:10]), Team(loadouts[10:15])), ((0, 0, 0, 0, 0),) * 3, "test")
    report = ConstraintEngine(loadouts + (duplicate_equip,)).check_defense(defense)
    assert not report.legal
    assert any("unique equipment" in reason for reason in report.reasons)


def test_position_order_is_strict(loadouts):
    team = Team((loadouts[4], loadouts[3], loadouts[2], loadouts[1], loadouts[0]))
    report = ConstraintEngine(loadouts).check_team(team)
    assert not report.legal
    assert any("standing_rank" in reason for reason in report.reasons)


def test_mask_limits_are_enforced(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    too_many_per_team = ((1, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    defense = DefensePlan(fmt3, teams, too_many_per_team, "test")
    report = ConstraintEngine(loadouts).check_defense(defense)
    assert not report.legal
    assert any("per-team" in reason for reason in report.reasons)


def test_observation_tracks_visible_and_hidden_slots(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    mask = ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    defense = DefensePlan(fmt3, teams, mask, "test")
    observation = observe_defense(defense)
    assert observation.hidden_slots == ((1, 1), (1, 5), (3, 2))
    assert loadouts[1].hero_id in observation.visible_heroes
    assert loadouts[0].hero_id not in observation.visible_heroes


def test_generator_produces_many_legal_plans(loadouts, fmt3):
    generator = LegalPlanGenerator(loadouts, seed=7)
    engine = ConstraintEngine(loadouts)
    for _ in range(20):
        attack = generator.generate_attack_plan(fmt3)
        assert engine.is_legal_attack(attack)


def test_legal_action_mask_can_ablate_future_feasibility(loadouts):
    engine = ConstraintEngine(loadouts)
    high_rank_candidate_index = len(loadouts) - 1

    strict = engine.legal_action_mask(
        loadouts,
        current_team_slots=(),
        remaining_team_slots_after_candidate=4,
        used_hero_ids=frozenset(),
        used_unique_equip_ids=frozenset(),
    )
    ablated = engine.legal_action_mask(
        loadouts,
        current_team_slots=(),
        remaining_team_slots_after_candidate=4,
        used_hero_ids=frozenset(),
        used_unique_equip_ids=frozenset(),
        use_future_feasibility=False,
    )

    assert not strict[high_rank_candidate_index]
    assert ablated[high_rank_candidate_index]


def test_legal_action_mask_reuses_sorted_pool(loadouts):
    engine = ConstraintEngine(loadouts)
    sorted_calls = 0
    real_sorted = builtins.sorted

    def counted_sorted(*args, **kwargs):
        nonlocal sorted_calls
        sorted_calls += 1
        return real_sorted(*args, **kwargs)

    original_sorted = builtins.sorted
    builtins.sorted = counted_sorted
    try:
        for _ in range(3):
            mask = engine.legal_action_mask(
                loadouts,
                current_team_slots=(),
                remaining_team_slots_after_candidate=4,
                used_hero_ids=frozenset(),
                used_unique_equip_ids=frozenset(),
            )
            assert any(mask)
    finally:
        builtins.sorted = original_sorted

    assert sorted_calls == 0


def test_team_and_plan_hashes_cache_canonical_hash(loadouts, fmt3, monkeypatch):
    calls = 0
    real_hash = model_module.canonical_hash

    def counted_hash(value):
        nonlocal calls
        calls += 1
        return real_hash(value)

    monkeypatch.setattr(model_module, "canonical_hash", counted_hash)
    team = Team(loadouts[0:5])
    attack = AttackPlan(fmt3, (team, Team(loadouts[5:10]), Team(loadouts[10:15])), "test")
    defense = DefensePlan(fmt3, attack.teams, ((0, 0, 0, 0, 0),) * 3, "test")

    assert team.hash() == team.hash()
    assert attack.hash() == attack.hash()
    assert defense.hash() == defense.hash()

    assert calls == 3


def test_matchup_cache_key_hash_caches_canonical_hash(loadouts, monkeypatch):
    calls = 0
    real_hash = cache_module.canonical_hash

    def counted_hash(value):
        nonlocal calls
        calls += 1
        return real_hash(value)

    attack = Team(loadouts[0:5])
    defense = Team(loadouts[5:10])
    key = MatchupCacheKey.from_teams(attack, defense)
    monkeypatch.setattr(cache_module, "canonical_hash", counted_hash)

    assert key.hash() == key.hash()

    assert calls == 1


def test_budgeted_attack_candidate_generation_avoids_random_retry_spins(monkeypatch):
    heroes = tuple(
        HeroRecord.from_mapping(
            hero_id=idx,
            name=f"budget-hero-{idx}",
            standing_rank=float(idx),
            standing_bucket="test",
            base_power=1000.0 + idx * 30.0,
            default_unique_equip_id=1000 + idx,
        )
        for idx in range(1, 41)
    )
    loadouts = tuple(
        Loadout.from_hero(
            hero,
            unique_equip_star=3 + hero.hero_id % 3,
            final_power=hero.base_power + (3 + hero.hero_id % 3) * 20.0,
        )
        for hero in heroes
    )
    fmt = MatchFormat(3)
    cheapest_legal_roster_cost = sum(loadout.cost for loadout in sorted(loadouts, key=lambda item: item.cost)[:15])
    reference_cost = cheapest_legal_roster_cost * 1.2 / 0.9
    mask_calls = 0
    real_mask = ConstraintEngine.legal_action_mask

    def counted_mask(self, *args, **kwargs):
        nonlocal mask_calls
        mask_calls += 1
        return real_mask(self, *args, **kwargs)

    monkeypatch.setattr(ConstraintEngine, "legal_action_mask", counted_mask)

    candidates = LegalPlanGenerator(loadouts, seed=7).generate_attack_candidates(
        fmt,
        count=3,
        goal=GenerationGoal(target_power_ratio=0.9),
        reference_cost=reference_cost,
    )

    assert len(candidates) == 3
    assert mask_calls < 1_000


def test_twenty_illegal_inputs_are_rejected(loadouts, fmt3):
    engine = ConstraintEngine(loadouts)
    illegal = []
    base_teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    illegal.append(DefensePlan(fmt3, (Team((loadouts[0], loadouts[0], *loadouts[2:5])), base_teams[1], base_teams[2]), ((0, 0, 0, 0, 0),) * 3, "bad"))
    illegal.append(DefensePlan(fmt3, (Team((loadouts[4], loadouts[3], loadouts[2], loadouts[1], loadouts[0])), base_teams[1], base_teams[2]), ((0, 0, 0, 0, 0),) * 3, "bad"))
    for idx in range(18):
        mask = ((1, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)) if idx % 2 == 0 else ((2, 1, 0, 0, 0),) * 3
        illegal.append(DefensePlan(fmt3, base_teams, mask, "bad"))
    assert len(illegal) == 20
    assert all(not engine.is_legal_defense(plan) for plan in illegal)
