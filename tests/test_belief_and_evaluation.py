from __future__ import annotations

from masked_team_league.belief import BeliefEngine
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.evaluation import match_win_probability
from masked_team_league.models import DefensePlan, Team, observe_defense


def test_match_win_probability_bo3_and_bo5():
    assert abs(match_win_probability([0.5, 0.5, 0.5], 2) - 0.5) < 1e-9
    assert abs(match_win_probability([1.0, 1.0, 0.0, 0.0, 0.0], 3) - 0.0001) < 1e-3
    assert match_win_probability([0.9, 0.9, 0.9, 0.1, 0.1], 3) > 0.7


def test_hidden_domains_exclude_visible_heroes_and_equips(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "test")
    observation = observe_defense(defense)
    engine = ConstraintEngine(loadouts)
    domains = engine.build_domains(observation)
    for domain in domains.values():
        assert all(loadout.hero_id not in observation.visible_heroes for loadout in domain)
        assert all(
            loadout.unique_equip_id is None or loadout.unique_equip_id not in observation.visible_unique_equip_ids
            for loadout in domain
        )


def test_belief_completions_are_legal_and_include_visible_slots(loadouts, fmt3):
    teams = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, teams, ((1, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "test")
    observation = observe_defense(defense)
    engine = ConstraintEngine(loadouts)
    belief = BeliefEngine(engine).build(observation, max_k=16)
    assert belief.feasible_count_estimate > 0
    assert abs(sum(belief.weights) - 1.0) < 1e-9
    for roster in belief.candidates:
        visible_slot = roster[0].slots[1]
        assert visible_slot.hero_id == loadouts[1].hero_id
        reconstructed = DefensePlan(fmt3, roster, ((0, 0, 0, 0, 0),) * 3, "completion")
        assert engine.is_legal_defense(reconstructed)
