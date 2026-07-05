from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path

from masked_team_league.scoring import SimulationCache
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.domain import MatchFormat
from masked_team_league.real_platform.oracle import OracleBatchEvaluator
from masked_team_league.real_platform.resources import load_hero_resource_bundle
from masked_team_league.league.round_runner import LeagueRoundConfig, LeagueRoundRunner
from masked_team_league.league.selfplay import (
    SelfPlayOrchestrator,
    SelfPlayOrchestratorConfig,
    build_attack_teacher_jsonl_from_round,
    build_defense_teacher_jsonl_from_round,
)


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
        return [
            {
                "request_id": request["request_id"],
                "status": "completed",
                "battle_result": 0 if index % 3 != 1 else 1,
            }
            for index, request in enumerate(self.submitted_requests)
        ]


def test_league_round_runner_uses_attack_proposal_checkpoint_after_training(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    evaluator = OracleBatchEvaluator(_FakeOracleClient(), resources, cache=SimulationCache())
    orchestrator = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=1,
            root_dir=tmp_path / "selfplay",
            training_dir=tmp_path / "training",
            round_config=LeagueRoundConfig(
                teams=3,
                defenses=1,
                attacks_per_defense=4,
                oracle_top_k=1,
                attack_roles=("main",),
                defense_roles=("main",),
                defense_roster_candidates=1,
                defense_masks_per_roster=1,
                defense_max_masks_per_roster=1,
            ),
            proposal_epochs=1,
            proposal_model_dim=32,
            proposal_heads=4,
            proposal_layers=1,
        ),
    )
    first = orchestrator.run()
    checkpoint = Path(first.rounds[0].attack_proposal_checkpoint or "")
    assert checkpoint.exists()

    evaluator2 = OracleBatchEvaluator(_FakeOracleClient(), resources, cache=SimulationCache())
    out_dir = tmp_path / "round_with_checkpoint"
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator2,
        config=LeagueRoundConfig(
            teams=3,
            defenses=1,
            attacks_per_defense=2,
            oracle_top_k=1,
            seed=99,
            attack_roles=("main",),
            defense_roles=("main",),
            defense_roster_candidates=1,
            defense_masks_per_roster=1,
            defense_max_masks_per_roster=1,
            attack_proposal_checkpoint=checkpoint,
            attack_proposal_beam_size=1,
        ),
    )

    runner.run(out_dir)
    candidates = [json.loads(line) for line in (out_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]

    assert candidates
    assert int(candidates[0]["candidate_sources"]) == 1


def test_selfplay_orchestrator_runs_rounds_and_feeds_proposal_checkpoint_forward(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    evaluator = OracleBatchEvaluator(_FakeOracleClient(), resources, cache=SimulationCache())
    orchestrator = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=2,
            root_dir=tmp_path / "selfplay",
            training_dir=tmp_path / "training",
            round_config=LeagueRoundConfig(
                teams=3,
                defenses=1,
                attacks_per_defense=4,
                oracle_top_k=1,
                seed=31,
                attack_roles=("main",),
                defense_roles=("main",),
                defense_roster_candidates=1,
                defense_masks_per_roster=1,
                defense_max_masks_per_roster=1,
                attack_pool_max_active=4,
                defense_pool_max_active=4,
            ),
            proposal_epochs=1,
            proposal_model_dim=32,
            proposal_heads=4,
            proposal_layers=1,
            attack_proposal_beam_size=1,
        ),
    )

    summary = orchestrator.run()
    state = json.loads((tmp_path / "selfplay" / "orchestrator_state.json").read_text(encoding="utf-8"))
    second_candidates = [
        json.loads(line)
        for line in (tmp_path / "selfplay" / "round_0002" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(summary.rounds) == 2
    assert summary.rounds[0].attack_proposal_checkpoint
    assert summary.rounds[1].attack_proposal_checkpoint
    assert Path(summary.latest_attack_proposal_checkpoint or "").exists()
    assert state["latest_attack_proposal_checkpoint"] == summary.latest_attack_proposal_checkpoint
    assert int(second_candidates[0]["candidate_sources"]) == 1


def test_selfplay_orchestrator_validates_each_round_and_stops_when_gate_clears(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    evaluator = OracleBatchEvaluator(_FakeOracleClient(), resources, cache=SimulationCache())
    calls: list[dict[str, object]] = []

    def fake_validation_builder(**kwargs):
        calls.append(kwargs)
        state_path = Path(kwargs["selfplay_root"]) / "orchestrator_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["rounds"][0]["round_id"] == "round_0001"
        assert state["latest_attack_proposal_checkpoint"]
        assert state["latest_defense_proposal_checkpoint"]
        return {
            "schema_version": "learned_exploiter_validation_report.v1",
            "selfplay_root": str(kwargs["selfplay_root"]),
            "training_root": str(kwargs["training_root"]),
            "rounds": 1,
            "oracle_requests": 3,
            "latest_attack_proposal_checkpoint": "attack.pt",
            "latest_defense_proposal_checkpoint": "defense.pt",
            "exploiter_report": {},
            "defense_anti_meta_report": {},
            "red_line_violations": [],
            "production_ready": True,
        }

    orchestrator = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=3,
            root_dir=tmp_path / "selfplay",
            training_dir=tmp_path / "training",
            round_config=LeagueRoundConfig(
                teams=3,
                defenses=1,
                attacks_per_defense=4,
                oracle_top_k=1,
                seed=41,
                attack_roles=("main",),
                defense_roles=("main",),
                defense_roster_candidates=1,
                defense_masks_per_roster=1,
                defense_max_masks_per_roster=1,
            ),
            proposal_epochs=1,
            proposal_model_dim=32,
            proposal_heads=4,
            proposal_layers=1,
            attack_proposal_beam_size=1,
            defense_proposal_beam_size=1,
            validate_after_each_round=True,
            stop_when_validation_ready=True,
            validation_min_rounds=1,
            validation_min_oracle_requests=1,
        ),
        learned_validation_builder=fake_validation_builder,
    )

    summary = orchestrator.run()
    state = json.loads((tmp_path / "selfplay" / "orchestrator_state.json").read_text(encoding="utf-8"))
    report_path = Path(summary.rounds[0].validation_report or "")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert calls[0]["selfplay_root"] == tmp_path / "selfplay"
    assert calls[0]["training_root"] == tmp_path / "training"
    assert calls[0]["min_rounds"] == 1
    assert len(summary.rounds) == 1
    assert summary.production_ready is True
    assert summary.stop_reason == "validation_ready"
    assert summary.validation_reports[0].production_ready is True
    assert summary.rounds[0].validation_production_ready is True
    assert summary.rounds[0].validation_red_line_violations == ()
    assert report["production_ready"] is True
    assert state["production_ready"] is True
    assert state["stop_reason"] == "validation_ready"
    assert state["rounds"][0]["validation_report"] == str(report_path)


def test_selfplay_orchestrator_dispatches_real_queries_into_teacher_feedback(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    client = _FakeOracleClient()
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache())
    orchestrator = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=1,
            root_dir=tmp_path / "selfplay",
            training_dir=tmp_path / "training",
            round_config=LeagueRoundConfig(
                teams=3,
                defenses=1,
                attacks_per_defense=4,
                oracle_top_k=1,
                seed=51,
                attack_roles=("main",),
                defense_roles=("main",),
                defense_roster_candidates=1,
                defense_masks_per_roster=1,
                defense_max_masks_per_roster=1,
                active_sim_keep=0,
                active_real_keep=1,
            ),
            proposal_epochs=1,
            proposal_model_dim=32,
            proposal_heads=4,
            proposal_layers=1,
            attack_proposal_beam_size=1,
            defense_proposal_beam_size=1,
        ),
    )

    summary = orchestrator.run()
    record = summary.rounds[0]
    attack_teacher_rows = [
        json.loads(line)
        for line in Path(record.teacher_jsonl or "").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    defense_teacher_rows = [
        json.loads(line)
        for line in Path(record.defense_teacher_jsonl or "").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    real_dir = Path(record.real_query_feedback_dir or "")

    assert record.real_query_pairs == 1
    assert real_dir.exists()
    assert (real_dir / "real_query_pairs.jsonl").exists()
    assert any(row["source"] == "active_real_query" for row in attack_teacher_rows)
    assert any(row["source"] == "active_real_query" for row in defense_teacher_rows)


def test_run_selfplay_orchestrator_script_has_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_selfplay_orchestrator", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--rounds" in result.stdout
    assert "--backend" in result.stdout
    assert "--proposal-epochs" in result.stdout
    assert "--no-dispatch-real-queries" in result.stdout
    assert "--validate-after-each-round" in result.stdout
    assert "--stop-when-validation-ready" in result.stdout
    assert "--validation-min-rounds" in result.stdout


def test_selfplay_orchestrator_trains_and_feeds_defense_checkpoint_forward(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    evaluator = OracleBatchEvaluator(_FakeOracleClient(), resources, cache=SimulationCache())
    orchestrator = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=2,
            root_dir=tmp_path / "selfplay",
            training_dir=tmp_path / "training",
            round_config=LeagueRoundConfig(
                teams=3,
                defenses=1,
                attacks_per_defense=4,
                oracle_top_k=1,
                seed=131,
                attack_roles=("main",),
                defense_roles=("main",),
                defense_roster_candidates=1,
                defense_masks_per_roster=1,
                defense_max_masks_per_roster=1,
            ),
            proposal_epochs=1,
            proposal_model_dim=32,
            proposal_heads=4,
            proposal_layers=1,
            attack_proposal_beam_size=1,
            defense_proposal_beam_size=1,
        ),
    )

    summary = orchestrator.run()
    state = json.loads((tmp_path / "selfplay" / "orchestrator_state.json").read_text(encoding="utf-8"))
    second_defenses = [
        json.loads(line)
        for line in (tmp_path / "selfplay" / "round_0002" / "scored_defenses.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(summary.rounds) == 2
    assert summary.rounds[0].defense_proposal_checkpoint
    assert summary.rounds[1].defense_proposal_checkpoint
    assert Path(summary.latest_defense_proposal_checkpoint or "").exists()
    assert state["latest_defense_proposal_checkpoint"] == summary.latest_defense_proposal_checkpoint
    assert second_defenses
    assert second_defenses[0]["defense_oracle_explanation"]["roster_sources"] == "1"


def test_build_defense_teacher_jsonl_records_anti_meta_residual_targets(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    plan = LegalPlanGenerator(resources.loadouts, seed=401).generate_defense_plan(
        MatchFormat(n_teams=3),
        source="defense_oracle:exploiter",
    )
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    row = {
        "round_id": "round_0001",
        "defense_id": "def-00001",
        "defense_role": "exploiter",
        "defense_plan": asdict(plan),
        "strength": 0.8,
        "break_rate": 0.2,
        "ambiguity_score": 1.2,
        "defense_risk_report": {
            "estimated_break_rate": 0.35,
            "estimated_survival_rate": 0.65,
            "meta_attack_success": 0.30,
        },
    }
    (round_dir / "scored_defenses.jsonl").write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    out_path = tmp_path / "defense_teacher.jsonl"

    count = build_defense_teacher_jsonl_from_round(round_dir, out_path)
    teacher = json.loads(out_path.read_text(encoding="utf-8"))

    assert count == 1
    assert teacher["value_target"] == 0.65
    assert teacher["survival_rate"] == 0.65
    assert teacher["meta_attack_success"] == 0.30
    assert teacher["anti_meta_residual_target"] == 0.35
    assert teacher["gap_target"] == 1.2


def test_build_attack_teacher_jsonl_records_exploiter_target_feedback(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    generator = LegalPlanGenerator(resources.loadouts, seed=501)
    attack = generator.generate_attack_plan(MatchFormat(n_teams=3), source="attack_oracle:exploiter")
    defense = generator.generate_defense_plan(MatchFormat(n_teams=3), source="defense_oracle:main")
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    candidate = {
        "round_id": "round_0001",
        "defense_id": "def-00001",
        "attack_id": "atk-00001",
        "attack_role": "exploiter",
        "rank": 1,
        "belief_top1_top2_gap": 0.10,
        "attack_plan": asdict(attack),
        "defense_hash": defense.hash(),
    }
    main_attack = generator.generate_attack_plan(MatchFormat(n_teams=3), source="attack_oracle:main")
    main_candidate = {
        **candidate,
        "attack_id": "atk-00002",
        "attack_role": "main",
        "attack_plan": asdict(main_attack),
    }
    scored_attack = {"attack_id": "atk-00001", "defense_id": "def-00001", "strength": 0.65}
    main_scored_attack = {"attack_id": "atk-00002", "defense_id": "def-00001", "strength": 0.70}
    scored_defense = {
        "round_id": "round_0001",
        "defense_id": "def-00001",
        "defense_hash": defense.hash(),
        "strength": 0.75,
        "break_rate": 0.25,
    }
    (round_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in (candidate, main_candidate)),
        encoding="utf-8",
    )
    (round_dir / "scored_attacks.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in (scored_attack, main_scored_attack)),
        encoding="utf-8",
    )
    (round_dir / "scored_defenses.jsonl").write_text(json.dumps(scored_defense, separators=(",", ":")) + "\n", encoding="utf-8")
    out_path = tmp_path / "attack_teacher.jsonl"

    count = build_attack_teacher_jsonl_from_round(round_dir, out_path)
    teacher_lines = out_path.read_text(encoding="utf-8").splitlines()
    teacher = json.loads(teacher_lines[0])

    assert count == 2
    assert teacher["attack_role"] == "exploiter"
    assert teacher["target_defense_id"] == "def-00001"
    assert teacher["target_defense_hash"] == defense.hash()
    assert teacher["target_defense_strength"] == 0.75
    assert teacher["target_baseline_break_rate"] == 0.70
    assert teacher["exploiter_residual_target"] == -0.05
    assert teacher["role_weight"] > 1.0
    main_teacher = json.loads(teacher_lines[1])
    assert main_teacher["attack_role"] == "main"
    assert main_teacher["role_weight"] == 1.0


def test_build_attack_teacher_jsonl_uses_main_attack_as_exploiter_baseline(tmp_path: Path) -> None:
    heroes_path = tmp_path / "heroes.json"
    _write_heroes(heroes_path)
    resources = load_hero_resource_bundle(heroes_path)
    generator = LegalPlanGenerator(resources.loadouts, seed=601)
    main_attack = generator.generate_attack_plan(MatchFormat(n_teams=3), source="attack_oracle:main")
    exploiter_attack = generator.generate_attack_plan(MatchFormat(n_teams=3), source="attack_oracle:exploiter")
    underdog_attack = generator.generate_attack_plan(MatchFormat(n_teams=3), source="attack_oracle:underdog")
    defense = generator.generate_defense_plan(MatchFormat(n_teams=3), source="defense_oracle:main")
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    rows = [
        {
            "round_id": "round_0001",
            "defense_id": "def-00001",
            "attack_id": "atk-main",
            "attack_role": "main",
            "rank": 1,
            "belief_top1_top2_gap": 0.10,
            "attack_plan": asdict(main_attack),
            "defense_hash": defense.hash(),
        },
        {
            "round_id": "round_0001",
            "defense_id": "def-00001",
            "attack_id": "atk-exp",
            "attack_role": "exploiter",
            "rank": 1,
            "belief_top1_top2_gap": 0.10,
            "attack_plan": asdict(exploiter_attack),
            "defense_hash": defense.hash(),
        },
        {
            "round_id": "round_0001",
            "defense_id": "def-00001",
            "attack_id": "atk-under",
            "attack_role": "underdog",
            "rank": 1,
            "belief_top1_top2_gap": 0.10,
            "attack_plan": asdict(underdog_attack),
            "defense_hash": defense.hash(),
        },
    ]
    scored_attacks = [
        {"attack_id": "atk-main", "defense_id": "def-00001", "strength": 0.40},
        {"attack_id": "atk-exp", "defense_id": "def-00001", "strength": 0.65},
        {"attack_id": "atk-under", "defense_id": "def-00001", "strength": 0.95},
    ]
    scored_defense = {
        "round_id": "round_0001",
        "defense_id": "def-00001",
        "defense_hash": defense.hash(),
        "strength": 0.05,
        "break_rate": 0.95,
    }
    (round_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (round_dir / "scored_attacks.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in scored_attacks),
        encoding="utf-8",
    )
    (round_dir / "scored_defenses.jsonl").write_text(json.dumps(scored_defense, separators=(",", ":")) + "\n", encoding="utf-8")
    out_path = tmp_path / "attack_teacher.jsonl"

    count = build_attack_teacher_jsonl_from_round(round_dir, out_path)
    teachers = {
        row["attack_id"]: row
        for row in (json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines())
    }

    assert count == 3
    assert teachers["atk-exp"]["target_baseline_break_rate"] == 0.40
    assert teachers["atk-exp"]["exploiter_residual_target"] == 0.25
    assert teachers["atk-under"]["target_baseline_break_rate"] == 0.40
    assert teachers["atk-under"]["exploiter_residual_target"] == 0.55
