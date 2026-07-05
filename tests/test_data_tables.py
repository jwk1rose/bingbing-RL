from __future__ import annotations

import json
from pathlib import Path

from masked_team_league.data_tables import (
    CORE_TABLE_SCHEMA_VERSION,
    LeagueStrategyTableRecord,
    LoadoutTableRecord,
    ObservationTableRecord,
    PlanMatchTableRecord,
    SingleMatchupTableRecord,
    load_table_jsonl,
    write_table_jsonl,
)
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.league import LeagueManager
from masked_team_league.models import MatchFormat, observe_defense


def test_core_table_records_round_trip_jsonl(tmp_path, loadouts):
    fmt = MatchFormat(3)
    generator = LegalPlanGenerator(loadouts, seed=4101)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)
    observation = observe_defense(defense)
    league = LeagueManager()
    attack_record = league.add_attack(attack, role="main", source="unit", strength=0.7)
    rows = (
        LoadoutTableRecord.from_loadout(loadouts[0], data_version="unit-data", season="S28"),
        ObservationTableRecord.from_observation(
            observation,
            real_frequency=0.25,
            belief_candidate_count=3,
            belief_entropy=0.5,
        ),
        SingleMatchupTableRecord.from_matchup(
            attack.teams[0],
            defense.teams[0],
            sim_or_real="sim",
            num_games=3,
            wins=2,
            mean_duration=123.0,
            mean_margin=1.5,
            simulator_version="unit-sim",
            model_version="unit-model",
        ),
        PlanMatchTableRecord.from_plan_match(
            attack,
            defense,
            sim_or_real="sim",
            num_games=3,
            round_win_rates=(1.0, 0.0, 1.0),
            simulator_version="unit-sim",
            model_version="unit-model",
        ),
        LeagueStrategyTableRecord.from_strategy(attack_record),
    )
    path = tmp_path / "table.jsonl"

    write_table_jsonl(path, rows)
    payloads = load_table_jsonl(path)

    assert len(payloads) == len(rows)
    assert {row["schema_version"] for row in payloads} == {CORE_TABLE_SCHEMA_VERSION}
    assert payloads[0]["table"] == "LoadoutTable"
    assert payloads[1]["table"] == "ObservationTable"
    assert payloads[2]["table"] == "SingleMatchupTable"
    assert payloads[3]["table"] == "PlanMatchTable"
    assert payloads[4]["table"] == "LeagueStrategyTable"
    json.dumps(payloads, sort_keys=True)


def test_core_tables_schema_doc_lists_all_persistent_tables():
    text = Path("docs/core_tables_schema.md").read_text(encoding="utf-8")

    assert "core_tables.v1" in text
    assert "LoadoutTable" in text
    assert "SingleMatchupTable" in text
    assert "PlanMatchTable" in text
    assert "ObservationTable" in text
    assert "LeagueStrategyTable" in text
