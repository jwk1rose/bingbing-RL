from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from masked_team_league.cache import SimulationCache
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.models import MatchFormat
from masked_team_league.real_oracle import OracleBatchEvaluator
from masked_team_league.resources import load_hero_resource_bundle
from masked_team_league.round_runner import LeagueRoundConfig, LeagueRoundRunner


def _write_heroes(path: Path, count: int = 40) -> None:
    heroes = []
    for hero_id in range(1, count + 1):
        heroes.append(
            {
                "id": hero_id,
                "displayName": f"英雄{hero_id}",
                "level": 100,
                "stars": 5,
                "rank": 23,
                "equipIds": [6000 + hero_id, 6100 + hero_id],
                "stats": {"GS": 10000 + hero_id * 100},
                "positionType": "front" if hero_id <= 12 else "mid" if hero_id <= 26 else "back",
            }
        )
    path.write_text(json.dumps({"heroes": heroes}, ensure_ascii=False), encoding="utf-8")


class _FakeOracleClient:
    def __init__(self) -> None:
        self.submitted_requests: list[dict[str, object]] = []

    def submit_and_wait(self, requests, *, metadata=None):
        self.submitted_requests.extend(requests)
        return {"job_id": "job-unit", "status": "completed"}

    def read_results(self, job_id):
        rows = []
        for index, request in enumerate(self.submitted_requests):
            rows.append(
                {
                    "request_id": request["request_id"],
                    "status": "completed",
                    "battle_result": 0 if index % 3 != 1 else 1,
                }
            )
        return rows


def test_oracle_batch_evaluator_batches_plan_pairs_and_populates_cache(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    fmt = MatchFormat(3)
    generator = LegalPlanGenerator(resources.loadouts, seed=1)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)
    cache = SimulationCache()
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=cache)

    records = evaluator.evaluate_pairs(
        [("atk-1", attack, "def-1", defense)],
        job_prefix="unit",
        base_seed=123,
        metadata={"kind": "unit"},
    )

    assert len(records) == 1
    assert records[0].attack_id == "atk-1"
    assert records[0].defense_id == "def-1"
    assert records[0].round_win_rates == (1.0, 0.0, 1.0)
    assert records[0].attack_success > 0.99
    assert len(client.submitted_requests) == 3
    assert len(cache) == 3

    second = evaluator.evaluate_pairs(
        [("atk-1", attack, "def-1", defense)],
        job_prefix="unit-repeat",
        base_seed=456,
    )

    assert second[0].round_win_rates == records[0].round_win_rates
    assert len(client.submitted_requests) == 3


def test_league_round_runner_writes_artifacts_and_updates_league(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "round_0001"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=3,
            defenses=2,
            attacks_per_defense=16,
            oracle_top_k=2,
            seed=7,
        ),
    )

    summary = runner.run(out_dir)

    assert summary.defenses == 2
    assert summary.oracle_pairs == 4
    assert summary.oracle_requests == 12
    assert (out_dir / "candidates.jsonl").exists()
    assert (out_dir / "oracle_results.jsonl").exists()
    assert (out_dir / "scored_attacks.jsonl").exists()
    assert (out_dir / "scored_defenses.jsonl").exists()
    state = json.loads((out_dir / "league_state.json").read_text(encoding="utf-8"))
    assert len(state["attack_pool"]) == 4
    assert len(state["defense_pool"]) == 2
    assert len(state["payoffs"]) == 4


def test_run_league_round_script_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_league_round.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--defenses" in result.stdout
