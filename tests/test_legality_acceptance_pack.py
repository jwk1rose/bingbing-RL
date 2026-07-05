from __future__ import annotations

import json
from pathlib import Path

from masked_team_league.constraints import ConstraintEngine
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.models import AttackPlan, DefensePlan, Loadout, MatchFormat, Team


def test_acceptance_pack_accepts_twenty_legal_attacks_and_twenty_legal_defenses(loadouts):
    engine = ConstraintEngine(loadouts)
    legal_attacks: list[AttackPlan] = []
    legal_defenses: list[DefensePlan] = []
    for index in range(20):
        fmt = MatchFormat(n_teams=3 if index < 10 else 5)
        generator = LegalPlanGenerator(loadouts, seed=1000 + index)
        legal_attacks.append(generator.generate_attack_plan(fmt, source="acceptance_pack"))
        legal_defenses.append(
            generator.generate_defense_plan(
                fmt,
                mask=_legal_mask(fmt, seed=index),
                source="acceptance_pack",
            )
        )

    assert len(legal_attacks) == 20
    assert len(legal_defenses) == 20
    assert all(engine.is_legal_attack(plan) for plan in legal_attacks)
    assert all(engine.is_legal_defense(plan) for plan in legal_defenses)


def test_acceptance_pack_rejects_twenty_illegal_inputs_with_reasons(loadouts):
    engine = ConstraintEngine(loadouts)
    illegal_defenses = _illegal_defense_cases(loadouts)

    assert len(illegal_defenses) == 20
    for label, plan in illegal_defenses:
        report = engine.check_defense(plan)
        assert not report.legal, label
        assert report.reasons, label


def test_rejection_diagnostics_are_structured_and_json_serializable(loadouts):
    engine = ConstraintEngine(loadouts)
    fmt3 = MatchFormat(n_teams=3)
    base3 = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    defense = DefensePlan(fmt3, base3, ((2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "illegal")

    report = engine.check_defense(defense)
    payload = report.to_json_dict()
    diagnostics = payload["diagnostics"]

    assert payload["schema_version"] == "legality_diagnostics.v1"
    assert payload["legal"] is False
    assert report.reasons
    assert diagnostics
    assert diagnostics[0]["code"] == "MASK_VALUE"
    assert diagnostics[0]["path"] == ["mask", "team_1"]
    assert diagnostics[0]["severity"] == "error"
    assert diagnostics[0]["message"] in report.reasons
    json.dumps(payload, sort_keys=True)


def test_legality_diagnostics_schema_doc_lists_required_fields():
    text = Path("docs/legality_diagnostics_schema.md").read_text(encoding="utf-8")

    assert "legality_diagnostics.v1" in text
    assert "schema_version" in text
    assert "diagnostics" in text
    assert "DUPLICATE_HERO" in text
    assert "MASK_VALUE" in text


def _legal_mask(match_format: MatchFormat, *, seed: int) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    hidden_total = 0
    for team_idx in range(match_format.n_teams):
        hidden_count = min((seed + team_idx) % 3, match_format.max_hidden_per_team)
        row = [0] * match_format.team_size
        for offset in range(hidden_count):
            if hidden_total >= match_format.max_hidden_total:
                break
            row[(seed + team_idx + offset * 2) % match_format.team_size] = 1
            hidden_total += 1
        rows.append(tuple(row))
    return tuple(rows)


def _illegal_defense_cases(loadouts: tuple[Loadout, ...]) -> list[tuple[str, DefensePlan]]:
    fmt3 = MatchFormat(n_teams=3)
    fmt5_global_mask = MatchFormat(n_teams=5, max_hidden_per_team=5, max_hidden_total=10)
    base3 = (Team(loadouts[0:5]), Team(loadouts[5:10]), Team(loadouts[10:15]))
    base5 = (
        Team(loadouts[0:5]),
        Team(loadouts[5:10]),
        Team(loadouts[10:15]),
        Team(loadouts[15:20]),
        Team(loadouts[20:25]),
    )
    cases: list[tuple[str, DefensePlan]] = []
    zero_mask3 = ((0, 0, 0, 0, 0),) * 3
    for team_idx in range(5):
        teams = list(base3)
        duplicate = loadouts[team_idx]
        teams[team_idx % 3] = Team((duplicate, duplicate, loadouts[20], loadouts[21], loadouts[22]))
        cases.append((f"duplicate hero in team {team_idx}", DefensePlan(fmt3, tuple(teams), zero_mask3, "illegal")))
    for case_idx in range(5):
        teams = list(base3)
        teams[(case_idx % 2) + 1] = Team((loadouts[0], *loadouts[16 + case_idx : 20 + case_idx]))
        cases.append((f"duplicate hero across roster {case_idx}", DefensePlan(fmt3, tuple(teams), zero_mask3, "illegal")))
    for case_idx in range(4):
        clone = Loadout(
            hero_id=1000 + case_idx,
            unique_equip_id=loadouts[0].unique_equip_id,
            unique_equip_star=loadouts[0].unique_equip_star,
            final_power=loadouts[14 + case_idx].final_power,
            standing_rank=15.5 + case_idx,
            standing_bucket="back",
        )
        cases.append(
            (
                f"duplicate unique equipment across roster {case_idx}",
                DefensePlan(fmt3, (base3[0], base3[1], Team((*loadouts[10:14], clone))), zero_mask3, "illegal"),
            )
        )
    for case_idx in range(3):
        bad_team = Team((loadouts[case_idx], loadouts[case_idx + 2], loadouts[case_idx + 1], loadouts[case_idx + 3], loadouts[case_idx + 4]))
        cases.append((f"standing order violation {case_idx}", DefensePlan(fmt3, (bad_team, base3[1], base3[2]), zero_mask3, "illegal")))
    cases.append(
        (
            "mask exceeds per-team hidden limit",
            DefensePlan(fmt3, base3, ((1, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "illegal"),
        )
    )
    cases.append(
        (
            "mask uses non-binary entries",
            DefensePlan(fmt3, base3, ((2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "illegal"),
        )
    )
    cases.append(
        (
            "mask exceeds global hidden limit",
            DefensePlan(
                fmt5_global_mask,
                base5,
                ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
                "illegal",
            ),
        )
    )
    return cases
