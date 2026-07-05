from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path

import pytest

from masked_team_league.league.active_feedback import dispatch_active_real_queries
from masked_team_league.scoring import SimulationCache
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.domain import MatchFormat
from masked_team_league.real_platform.oracle import OracleBatchEvaluator
from masked_team_league.real_platform.resources import load_hero_resource_bundle


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
        self.metadata: dict[str, object] | None = None

    def submit_and_wait(self, requests, *, metadata=None):
        self.submitted_requests.extend(requests)
        self.metadata = metadata
        return {"job_id": "job-real-query", "status": "completed"}

    def read_results(self, job_id):
        return [
            {
                "request_id": request["request_id"],
                "status": "completed",
                "battle_result": 0 if index % 3 != 1 else 1,
            }
            for index, request in enumerate(self.submitted_requests)
        ]


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def test_dispatch_active_real_queries_writes_feedback_and_teacher_jsonl(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    generator = LegalPlanGenerator(resources.loadouts, seed=701)
    fmt = MatchFormat(n_teams=3)
    defense = generator.generate_defense_plan(fmt, source="defense_oracle:main")
    attack = generator.generate_attack_plan(fmt, source="attack_oracle:underdog")
    round_dir = tmp_path / "round_0001"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "round_id": "round_0001",
                "attack_id": "atk-00001",
                "defense_id": "def-00001",
                "attack_role": "underdog",
                "rank": 1,
                "belief_top1_top2_gap": 0.25,
                "target_defense_id": "def-00001",
                "target_defense_hash": defense.hash(),
                "target_defense_strength": 0.80,
                "target_baseline_break_rate": 0.20,
                "role_weight": 1.25,
                "attack_plan": asdict(attack),
            }
        ],
    )
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "round_id": "round_0001",
                "defense_id": "def-00001",
                "defense_role": "main",
                "defense_plan": asdict(defense),
                "ambiguity_score": 1.5,
                "defense_risk_report": {"meta_attack_success": 0.20},
            }
        ],
    )
    _write_jsonl(
        round_dir / "active_queries.jsonl",
        [
            {"queue": "sim", "query_id": "q-sim", "attack_id": "atk-00001", "defense_id": "def-00001"},
            {
                "queue": "real",
                "query_id": "q-real",
                "query_type": "underdog",
                "attack_id": "atk-00001",
                "defense_id": "def-00001",
                "score": 3.0,
            },
            {
                "queue": "real",
                "query_id": "q-missing",
                "query_type": "mask_observation",
                "attack_id": "atk-missing",
                "defense_id": "def-00001",
                "score": 2.0,
            },
        ],
    )
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    out_dir = tmp_path / "real_query_feedback"

    summary = dispatch_active_real_queries(
        round_dir,
        out_dir,
        evaluator=evaluator,
        job_prefix="real_query_round_0001",
        base_seed=900,
    )

    assert summary.queued_queries == 2
    assert summary.dispatchable_queries == 1
    assert summary.skipped_queries == 1
    assert summary.dispatched_pairs == 1
    assert summary.oracle_requests == 3
    assert summary.oracle_result_errors == 0
    assert summary.completion_rate == 1.0
    assert summary.teacher_feedback_complete is True
    assert len(client.submitted_requests) == 3
    assert client.metadata["kind"] == "masked_team_league_real_query"
    pairs = [json.loads(line) for line in (out_dir / "real_query_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    attack_teacher = [json.loads(line) for line in (out_dir / "attack_teacher.jsonl").read_text(encoding="utf-8").splitlines()]
    defense_teacher = [json.loads(line) for line in (out_dir / "defense_teacher.jsonl").read_text(encoding="utf-8").splitlines()]
    validation = json.loads((out_dir / "validation_report.json").read_text(encoding="utf-8"))

    assert pairs[0]["query_id"] == "q-real"
    assert validation["real_query_queue_validated"] is True
    assert validation["schema_version"] == "active_real_query_dispatch_validation.v1"
    assert validation["module"] == "ActiveRealQueryDispatch"
    assert validation["skipped_queries"] == 1
    assert validation["oracle_result_errors"] == 0
    assert validation["teacher_feedback_complete"] is True
    assert pairs[0]["attack_success"] > 0.99
    assert attack_teacher[0]["source"] == "active_real_query"
    assert attack_teacher[0]["query_id"] == "q-real"
    assert attack_teacher[0]["attack_success"] == pytest.approx(pairs[0]["attack_success"])
    assert attack_teacher[0]["target_defense_id"] == "def-00001"
    assert attack_teacher[0]["target_defense_hash"] == defense.hash()
    assert attack_teacher[0]["target_defense_strength"] == 0.80
    assert attack_teacher[0]["target_baseline_break_rate"] == 0.20
    assert attack_teacher[0]["exploiter_residual_target"] == pytest.approx(pairs[0]["attack_success"] - 0.20)
    assert attack_teacher[0]["role_weight"] == 1.25
    assert defense_teacher[0]["source"] == "active_real_query"
    assert defense_teacher[0]["break_rate"] == pytest.approx(pairs[0]["attack_success"])
    assert defense_teacher[0]["survival_rate"] == pytest.approx(1.0 - pairs[0]["attack_success"])
    assert defense_teacher[0]["anti_meta_residual_target"] == pytest.approx(
        1.0 - pairs[0]["attack_success"] - 0.2
    )


def test_dispatch_active_real_queries_script_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.dispatch_active_real_queries", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-dir" in result.stdout
    assert "--backend" in result.stdout
