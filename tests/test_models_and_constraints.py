from __future__ import annotations

import pytest

from masked_team_league.constraints import ConstraintEngine
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.models import DefensePlan, Loadout, Team, observe_defense


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
