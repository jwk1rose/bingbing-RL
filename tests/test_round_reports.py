from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from masked_team_league.reports import (
    build_active_query_feedback_report,
    build_active_real_query_dispatch_validation_report,
    build_attack_oracle_failure_validation_report,
    build_belief_real_distribution_validation_report,
    build_data_engineering_validation_report,
    build_defense_anti_meta_effectiveness_report,
    build_exploiter_effectiveness_report,
    build_learned_exploiter_validation_report,
    build_league_round_report,
    build_league_selfplay_health_report,
    build_mask_explanation_validation_report,
    build_production_readiness_report,
    build_underdog_residual_validation_report,
    build_v4_conformance_validation_report,
    red_line_violations,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_build_league_round_report_reads_runtime_artifacts(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(
        json.dumps(
            {
                "oracle_requests": 9,
                "best_attack_success": 0.75,
                "worst_defense_break_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {"attack_role": "main", "belief_entropy": 1.0},
            {"attack_role": "underdog", "belief_entropy": 2.0},
        ],
    )
    _write_jsonl(round_dir / "scored_defenses.jsonl", [{"break_rate": 0.5, "ambiguity_score": 2.0}])
    _write_jsonl(
        round_dir / "oracle_results.jsonl",
        [{"status": "completed"}, {"status": "completed"}, {"status": "error", "error": "boom"}],
    )
    _write_jsonl(round_dir / "active_queries.jsonl", [{"query_id": "q1", "queue": "sim", "score": 2.0}])
    (round_dir / "league_state.json").write_text(
        json.dumps(
            {
                "attack_pool": [{"diversity_cluster": "a"}, {"diversity_cluster": "b"}],
                "defense_pool": [{"diversity_cluster": "a"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_league_round_report(round_dir, date="2026-07-05")
    payload = report.to_json_dict()

    assert payload["sim_games"] == 9
    assert payload["attack_oracle"]["top1"] == 0.75
    assert payload["defense_oracle"]["attack_success"] == 0.5
    assert payload["league"]["attack_pool"] == 2
    assert payload["league"]["clusters"] == 2
    assert payload["underdog"]["samples"] == 1
    assert payload["active_queries"][0]["query_id"] == "q1"
    assert payload["failure_cases"][0]["error"] == "boom"
    assert red_line_violations(payload) == ["oracle_result_errors"]


def test_build_league_round_report_extracts_domain_risk_metrics(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(
        json.dumps(
            {
                "oracle_requests": 6,
                "best_attack_success": 0.8,
                "worst_defense_break_rate": 0.4,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_role": "main",
                "belief_entropy": 1.5,
                "attack_risk_report": {
                    "expected_match_win": 0.8,
                    "worst_case_match_win": 0.55,
                    "expected_lane_win_rates": [0.75, 0.5, 1.0],
                    "backup_attack_count": 2,
                    "belief_case_count": 4,
                },
            },
            {
                "attack_role": "underdog",
                "belief_entropy": 0.5,
                "attack_risk_report": {
                    "expected_match_win": 0.6,
                    "worst_case_match_win": 0.25,
                    "expected_lane_win_rates": [0.5, 0.5, 0.5],
                    "underdog_gap": 0.2,
                    "underdog_residual_bonus": 0.05,
                    "backup_attack_count": 0,
                    "belief_case_count": 2,
                },
            },
        ],
    )
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "break_rate": 0.35,
                "ambiguity_score": 2.0,
                "hidden_count": 6,
                "defense_risk_report": {
                    "estimated_break_rate": 0.45,
                    "estimated_survival_rate": 0.55,
                    "hidden_count": 6,
                    "backup_defense_count": 1,
                    "underdog_defense_gap": 0.3,
                    "underdog_residual_bonus": 0.075,
                },
            }
        ],
    )
    _write_jsonl(round_dir / "oracle_results.jsonl", [{"status": "completed"} for _ in range(6)])
    _write_jsonl(round_dir / "active_queries.jsonl", [{"query_id": "q1", "queue": "real", "score": 3.0}])
    (round_dir / "league_state.json").write_text(
        json.dumps({"attack_pool": [{"diversity_cluster": "a"}], "defense_pool": [{"diversity_cluster": "b"}]}),
        encoding="utf-8",
    )

    payload = build_league_round_report(round_dir, date="2026-07-05").to_json_dict()

    assert payload["attack_oracle"]["belief_expected_mean"] == 0.7
    assert payload["attack_oracle"]["belief_worst_case_mean"] == 0.4
    assert payload["attack_oracle"]["backup_attack_mean"] == 1.0
    assert payload["attack_oracle"]["belief_case_mean"] == 3.0
    assert payload["attack_oracle"]["underdog_gap_mean"] == 0.2
    assert payload["attack_oracle"]["underdog_residual_bonus_mean"] == 0.05
    assert payload["defense_oracle"]["estimated_break_rate"] == 0.45
    assert payload["defense_oracle"]["estimated_survival_rate"] == 0.55
    assert payload["defense_oracle"]["hidden_count_mean"] == 6.0
    assert payload["defense_oracle"]["backup_defense_mean"] == 1.0
    assert payload["defense_oracle"]["underdog_gap_mean"] == 0.3
    assert payload["defense_oracle"]["underdog_residual_bonus_mean"] == 0.075
    assert payload["league"]["active_query_count"] == 1


def test_build_data_engineering_validation_report_checks_metadata_hashes_and_core_tables(tmp_path: Path):
    round_dir = tmp_path / "round_0001"
    table_dir = round_dir / "tables"
    table_dir.mkdir(parents=True)
    summary_text = json.dumps({"oracle_requests": 3}) + "\n"
    summary_path = round_dir / "summary.json"
    summary_path.write_text(summary_text, encoding="utf-8")
    expected_tables = {
        "loadouts.jsonl": "LoadoutTable",
        "single_matchups.jsonl": "SingleMatchupTable",
        "plan_matches.jsonl": "PlanMatchTable",
        "observations.jsonl": "ObservationTable",
        "league_strategies.jsonl": "LeagueStrategyTable",
    }
    for filename, table in expected_tables.items():
        _write_jsonl(table_dir / filename, [{"schema_version": "core_tables.v1", "table": table, "id": filename}])
    (round_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "run_metadata.v1",
                "run_id": "round_0001",
                "output_artifacts": [
                    {
                        "path": str(summary_path),
                        "kind": "json",
                        "role": "output",
                        "sha256": _sha256_text(summary_text),
                        "size_bytes": len(summary_text.encode("utf-8")),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_data_engineering_validation_report([round_dir])

    assert report["schema_version"] == "data_engineering_validation_report.v1"
    assert report["rounds"] == 1
    assert report["metadata_coverage"] == 1.0
    assert report["artifact_hash_coverage"] == 1.0
    assert report["core_table_coverage"] == 1.0
    assert report["round_reports"][0]["table_rows"]["LoadoutTable"] == 1
    assert report["round_reports"][0]["artifact_hash_mismatch_count"] == 0
    assert report["red_line_violations"] == []


def test_build_data_engineering_validation_report_flags_missing_metadata_tables_and_bad_hash(tmp_path: Path):
    round_a = tmp_path / "round_a"
    round_b = tmp_path / "round_b"
    round_a.mkdir()
    (round_a / "summary.json").write_text("{}\n", encoding="utf-8")
    (round_a / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "run_metadata.v1",
                "run_id": "round_a",
                "output_artifacts": [
                    {
                        "path": str(round_a / "summary.json"),
                        "kind": "json",
                        "role": "output",
                        "sha256": "bad-hash",
                        "size_bytes": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    round_b.mkdir()

    report = build_data_engineering_validation_report(
        [round_a, round_b],
        min_metadata_coverage=1.0,
        min_core_table_coverage=1.0,
        min_artifact_hash_coverage=1.0,
    )

    assert report["rounds"] == 2
    assert report["metadata_files"] == 1
    assert report["artifact_hash_mismatch_count"] == 1
    assert report["core_table_coverage"] == 0.0
    assert "run_metadata_missing" in report["red_line_violations"]
    assert "artifact_hash_mismatch" in report["red_line_violations"]
    assert "core_table_missing" in report["red_line_violations"]
    assert "core_table_coverage_low" in report["red_line_violations"]


def test_build_underdog_residual_validation_report_summarizes_attack_and_defense_fields(tmp_path: Path):
    round_dir = tmp_path / "round_0001"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_id": "atk-main",
                "defense_id": "def-1",
                "attack_role": "main",
                "attack_risk_report": {
                    "objective_score": 0.7,
                    "underdog_gap": 0.0,
                    "underdog_residual_bonus": 0.0,
                    "attack_cost": 100.0,
                    "reference_defense_cost": 100.0,
                },
            },
            {
                "attack_id": "atk-under",
                "defense_id": "def-1",
                "attack_role": "underdog",
                "attack_risk_report": {
                    "objective_score": 0.82,
                    "expected_match_win": 0.77,
                    "underdog_gap": 0.25,
                    "underdog_residual_bonus": 0.05,
                    "attack_cost": 75.0,
                    "reference_defense_cost": 100.0,
                },
            },
        ],
    )
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "defense_id": "def-under",
                "defense_role": "underdog",
                "defense_risk_report": {
                    "estimated_survival_rate": 0.62,
                    "underdog_defense_gap": 0.3,
                    "underdog_residual_bonus": 0.075,
                    "defense_cost": 70.0,
                    "reference_attack_cost": 100.0,
                },
            }
        ],
    )

    report = build_underdog_residual_validation_report(
        [round_dir],
        min_attack_residual_coverage=1.0,
        min_defense_residual_coverage=1.0,
        min_mean_attack_residual_bonus=0.01,
        min_mean_defense_residual_bonus=0.01,
    )

    assert report["schema_version"] == "underdog_residual_validation_report.v1"
    assert report["rounds"] == 1
    assert report["attack_rows"] == 2
    assert report["attack_underdog_rows"] == 1
    assert report["attack_residual_rows"] == 1
    assert report["attack_residual_coverage"] == 1.0
    assert report["mean_attack_underdog_gap"] == 0.25
    assert report["mean_attack_residual_bonus"] == 0.05
    assert report["mean_attack_objective_score"] == 0.82
    assert report["defense_rows"] == 1
    assert report["defense_underdog_rows"] == 1
    assert report["defense_residual_rows"] == 1
    assert report["defense_residual_coverage"] == 1.0
    assert report["mean_defense_underdog_gap"] == 0.3
    assert report["mean_defense_residual_bonus"] == 0.075
    assert report["mean_defense_objective_score"] == 0.62
    assert report["round_reports"][0]["round_dir"] == str(round_dir)
    assert report["red_line_violations"] == []


def test_build_underdog_residual_validation_report_flags_missing_and_weak_coverage(tmp_path: Path):
    round_dir = tmp_path / "round_0001"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_id": "atk-under",
                "defense_id": "def-1",
                "attack_role": "underdog",
                "attack_risk_report": {"objective_score": 0.5},
            }
        ],
    )
    _write_jsonl(round_dir / "scored_defenses.jsonl", [{"defense_id": "def-main", "defense_role": "main"}])

    report = build_underdog_residual_validation_report(
        [round_dir],
        min_attack_residual_coverage=1.0,
        min_defense_residual_coverage=1.0,
        min_mean_attack_residual_bonus=0.01,
        min_mean_defense_residual_bonus=0.01,
    )

    assert report["attack_underdog_rows"] == 1
    assert report["attack_residual_coverage"] == 0.0
    assert "attack_residual_coverage_low" in report["red_line_violations"]
    assert "attack_residual_bonus_non_positive" in report["red_line_violations"]
    assert "defense_underdog_rows_missing" in report["red_line_violations"]
    assert "defense_residual_coverage_low" in report["red_line_violations"]
    assert "defense_residual_bonus_non_positive" in report["red_line_violations"]


def test_report_data_engineering_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_data_engineering_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout


def test_report_underdog_residual_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_underdog_residual_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--min-attack-residual-coverage" in result.stdout


def test_build_league_selfplay_health_report_summarizes_pool_roles_payoffs_and_growth(tmp_path: Path):
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    round_a.mkdir()
    round_b.mkdir()
    (round_a / "summary.json").write_text(
        json.dumps({"round_id": "round_0001", "oracle_requests": 3, "best_attack_success": 0.55, "worst_defense_break_rate": 0.45}),
        encoding="utf-8",
    )
    (round_b / "summary.json").write_text(
        json.dumps({"round_id": "round_0002", "oracle_requests": 9, "best_attack_success": 0.75, "worst_defense_break_rate": 0.35}),
        encoding="utf-8",
    )
    (round_a / "league_state.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "attack_pool": [
                    {"strategy_id": "atk-1", "role": "main", "strength": 0.55, "diversity_cluster": "a", "created_iteration": 1, "active": True}
                ],
                "defense_pool": [
                    {"strategy_id": "def-1", "role": "main", "strength": 0.55, "diversity_cluster": "d", "created_iteration": 1, "active": True}
                ],
                "payoffs": [{"attack_id": "atk-1", "defense_id": "def-1", "attack_success": 0.45, "games": 3}],
            }
        ),
        encoding="utf-8",
    )
    (round_b / "league_state.json").write_text(
        json.dumps(
            {
                "iteration": 2,
                "attack_pool": [
                    {"strategy_id": "atk-1", "role": "main", "strength": 0.55, "diversity_cluster": "a", "created_iteration": 1, "active": True},
                    {"strategy_id": "atk-2", "role": "exploiter", "strength": 0.80, "diversity_cluster": "b", "created_iteration": 2, "active": True},
                    {"strategy_id": "atk-3", "role": "underdog", "strength": 0.65, "diversity_cluster": "c", "created_iteration": 2, "active": True},
                ],
                "defense_pool": [
                    {"strategy_id": "def-1", "role": "main", "strength": 0.55, "diversity_cluster": "d", "created_iteration": 1, "active": True},
                    {"strategy_id": "def-2", "role": "exploiter", "strength": 0.70, "diversity_cluster": "e", "created_iteration": 2, "active": True},
                    {"strategy_id": "def-3", "role": "underdog", "strength": 0.60, "diversity_cluster": "f", "created_iteration": 2, "active": True},
                ],
                "payoffs": [
                    {"attack_id": "atk-1", "defense_id": "def-1", "attack_success": 0.45, "games": 3},
                    {"attack_id": "atk-2", "defense_id": "def-1", "attack_success": 0.75, "games": 3},
                    {"attack_id": "atk-3", "defense_id": "def-1", "attack_success": 0.60, "games": 3},
                    {"attack_id": "atk-1", "defense_id": "def-2", "attack_success": 0.35, "games": 3},
                    {"attack_id": "atk-2", "defense_id": "def-2", "attack_success": 0.50, "games": 3},
                    {"attack_id": "atk-3", "defense_id": "def-3", "attack_success": 0.40, "games": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        round_b / "candidates.jsonl",
        [
            {"attack_id": "atk-1", "attack_role": "main", "defense_role": "main"},
            {"attack_id": "atk-2", "attack_role": "exploiter", "defense_role": "exploiter"},
            {"attack_id": "atk-3", "attack_role": "underdog", "defense_role": "underdog"},
        ],
    )
    _write_jsonl(
        round_b / "scored_defenses.jsonl",
        [
            {"defense_id": "def-1", "role": "main", "strength": 0.55, "break_rate": 0.45},
            {"defense_id": "def-2", "role": "exploiter", "strength": 0.70, "break_rate": 0.30},
            {"defense_id": "def-3", "role": "underdog", "strength": 0.60, "break_rate": 0.40},
        ],
    )

    report = build_league_selfplay_health_report(
        [round_a, round_b],
        min_attack_pool=3,
        min_defense_pool=3,
        min_total_clusters=6,
        min_payoff_density=0.60,
        required_attack_roles=("main", "exploiter", "underdog"),
        required_defense_roles=("main", "exploiter", "underdog"),
        min_active_pool_fraction=1.0,
        min_new_attack_strength_delta=0.0,
        min_new_defense_strength_delta=0.0,
    )

    assert report["schema_version"] == "league_selfplay_health_report.v1"
    assert report["rounds"] == 2
    assert report["latest_iteration"] == 2
    assert report["attack_pool"] == 3
    assert report["defense_pool"] == 3
    assert report["total_clusters"] == 6
    assert report["payoff_entries"] == 6
    assert report["payoff_density"] == 6 / 9
    assert report["attack_role_coverage"] == 1.0
    assert report["defense_role_coverage"] == 1.0
    assert report["new_attack_strength_delta"] > 0.0
    assert report["new_defense_strength_delta"] > 0.0
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []


def test_build_league_selfplay_health_report_flags_pool_collapse_and_missing_payoffs(tmp_path: Path):
    round_dir = tmp_path / "round_0001"
    round_dir.mkdir()
    (round_dir / "summary.json").write_text(json.dumps({"round_id": "round_0001", "oracle_requests": 0}), encoding="utf-8")
    (round_dir / "league_state.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "attack_pool": [
                    {"strategy_id": "atk-1", "role": "main", "strength": 0.5, "diversity_cluster": "same", "created_iteration": 1, "active": True},
                    {"strategy_id": "atk-2", "role": "main", "strength": 0.4, "diversity_cluster": "same", "created_iteration": 1, "active": False, "retired_reason": "retention"},
                ],
                "defense_pool": [],
                "payoffs": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_league_selfplay_health_report(
        [round_dir],
        min_attack_pool=3,
        min_defense_pool=1,
        min_total_clusters=2,
        min_payoff_density=0.1,
        required_attack_roles=("main", "exploiter"),
        required_defense_roles=("main",),
        min_active_pool_fraction=0.75,
    )

    assert "attack_pool_too_small" in report["red_line_violations"]
    assert "defense_pool_too_small" in report["red_line_violations"]
    assert "league_cluster_collapse" in report["red_line_violations"]
    assert "attack_role_coverage_low" in report["red_line_violations"]
    assert "defense_role_coverage_low" in report["red_line_violations"]
    assert "payoff_density_low" in report["red_line_violations"]
    assert "active_pool_fraction_low" in report["red_line_violations"]
    assert report["production_ready"] is False


def test_report_league_selfplay_health_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_league_selfplay_health.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--required-attack-role" in result.stdout
    assert "--min-payoff-density" in result.stdout


def test_build_active_real_query_dispatch_validation_report_summarizes_feedback(tmp_path: Path):
    validation_json = tmp_path / "validation_report.json"
    validation_json.write_text(
        json.dumps(
            {
                "schema_version": "active_real_query_dispatch_validation.v1",
                "module": "ActiveRealQueryDispatch",
                "round_dir": "round_0001",
                "out_dir": "real_queries",
                "queued_queries": 3,
                "dispatchable_queries": 2,
                "skipped_queries": 1,
                "skipped_query_reasons": {"missing_candidate_or_defense_artifact": 1},
                "dispatched_pairs": 2,
                "oracle_requests": 6,
                "oracle_result_errors": 0,
                "completion_rate": 1.0,
                "attack_teacher_rows": 2,
                "defense_teacher_rows": 2,
                "teacher_feedback_complete": True,
                "real_query_queue_validated": True,
                "submitted_request_count": 6,
            }
        ),
        encoding="utf-8",
    )

    report = build_active_real_query_dispatch_validation_report(
        [validation_json],
        min_reports=1,
        min_dispatched_pairs=2,
        min_completion_rate=1.0,
    )

    assert report["schema_version"] == "active_real_query_dispatch_validation_report.v1"
    assert report["reports"] == 1
    assert report["queued_queries"] == 3
    assert report["dispatchable_queries"] == 2
    assert report["skipped_queries"] == 1
    assert report["dispatched_pairs"] == 2
    assert report["oracle_requests"] == 6
    assert report["oracle_result_errors"] == 0
    assert report["completion_rate"] == 1.0
    assert report["attack_teacher_rows"] == 2
    assert report["defense_teacher_rows"] == 2
    assert report["teacher_feedback_complete_reports"] == 1
    assert report["real_query_queue_validated_reports"] == 1
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []


def test_build_active_real_query_dispatch_validation_report_flags_failed_feedback(tmp_path: Path):
    validation_json = tmp_path / "bad_validation_report.json"
    validation_json.write_text(
        json.dumps(
            {
                "schema_version": "active_real_query_dispatch_validation.v1",
                "module": "ActiveRealQueryDispatch",
                "round_dir": "round_0001",
                "out_dir": "real_queries",
                "queued_queries": 2,
                "dispatchable_queries": 2,
                "skipped_queries": 0,
                "dispatched_pairs": 0,
                "oracle_requests": 3,
                "oracle_result_errors": 1,
                "completion_rate": 2 / 3,
                "attack_teacher_rows": 0,
                "defense_teacher_rows": 0,
                "teacher_feedback_complete": False,
                "real_query_queue_validated": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_active_real_query_dispatch_validation_report(
        [validation_json, tmp_path / "missing.json"],
        min_reports=2,
        min_dispatched_pairs=1,
        min_completion_rate=0.99,
    )

    assert report["read_error_reports"] == 1
    assert "dispatch_validation_report_read_error" in report["red_line_violations"]
    assert "active_real_dispatched_pairs_low" in report["red_line_violations"]
    assert "active_real_oracle_result_errors" in report["red_line_violations"]
    assert "active_real_completion_rate_low" in report["red_line_violations"]
    assert "active_real_teacher_feedback_incomplete" in report["red_line_violations"]
    assert "active_real_queue_not_validated" in report["red_line_violations"]
    assert report["production_ready"] is False


def test_report_active_real_query_dispatch_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_active_real_query_dispatch_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--validation-json" in result.stdout
    assert "--min-dispatched-pairs" in result.stdout
    assert "--out-report" in result.stdout


def test_build_active_query_feedback_report_joins_queries_to_oracle_pairs(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "active_queries.jsonl",
        [
            {
                "queue": "sim",
                "query_id": "q1",
                "query_type": "mask_observation",
                "attack_id": "atk-1",
                "defense_id": "def-1",
                "score": 3.0,
                "info_gain": 1.0,
                "decision_impact": 0.5,
                "novelty": 0.2,
                "underdog_potential": 0.0,
                "cost": 3,
            },
            {
                "queue": "real",
                "query_id": "q2",
                "query_type": "underdog",
                "attack_id": "atk-2",
                "defense_id": "def-1",
                "score": 2.0,
                "info_gain": 0.4,
                "decision_impact": 0.9,
                "novelty": 0.5,
                "underdog_potential": 1.0,
                "cost": 3,
            },
        ],
    )
    _write_jsonl(
        round_dir / "oracle_pairs.jsonl",
        [
            {"attack_id": "atk-1", "defense_id": "def-1", "attack_success": 0.8, "round_win_rates": [1.0, 0.0, 1.0]},
            {"attack_id": "atk-2", "defense_id": "def-1", "attack_success": 0.25, "round_win_rates": [0.0, 1.0, 0.0]},
        ],
    )
    _write_jsonl(
        round_dir / "oracle_results.jsonl",
        [{"status": "completed"}, {"status": "error", "error": "boom"}],
    )

    report = build_active_query_feedback_report(round_dir)

    assert report["queries"] == 2
    assert report["oracle_result_rows"] == 2
    assert report["oracle_result_errors"] == 1
    assert report["unmatched_queries"] == 0
    assert report["matched_query_coverage"] == 1.0
    assert report["oracle_result_error_rate"] == 0.5
    assert report["real_queries"] == 1
    assert report["matched_real_queries"] == 1
    assert report["real_query_feedback_coverage"] == 1.0
    assert report["queues"]["sim"]["mean_attack_success"] == 0.8
    assert report["queues"]["real"]["mean_attack_success"] == 0.25
    assert report["queues"]["real"]["underdog_queries"] == 1
    assert report["query_feedback"][0]["query_id"] == "q1"
    assert report["query_feedback"][0]["attack_success"] == 0.8
    assert "oracle_result_errors" in report["red_line_violations"]


def test_build_active_query_feedback_report_flags_missing_feedback_and_real_query_shortfall(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "active_queries.jsonl",
        [
            {
                "queue": "real",
                "query_id": "q-missing",
                "query_type": "underdog",
                "attack_id": "atk-1",
                "defense_id": "def-1",
                "score": 3.0,
            }
        ],
    )
    _write_jsonl(round_dir / "oracle_pairs.jsonl", [])
    _write_jsonl(round_dir / "oracle_results.jsonl", [{"status": "completed"}])

    report = build_active_query_feedback_report(
        round_dir,
        min_matched_query_coverage=1.0,
        min_real_query_count=2,
    )

    assert report["matched_query_coverage"] == 0.0
    assert report["real_queries"] == 1
    assert report["matched_real_queries"] == 0
    assert "active_query_feedback_coverage_low" in report["red_line_violations"]
    assert "real_query_feedback_missing" in report["red_line_violations"]
    assert "real_query_count_low" in report["red_line_violations"]


def test_report_active_query_feedback_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_active_query_feedback.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout
    assert "--min-matched-query-coverage" in result.stdout


def test_report_active_query_feedback_script_merges_oracle_result_error_rate(tmp_path: Path):
    round_a = tmp_path / "round_a"
    round_b = tmp_path / "round_b"
    out_report = tmp_path / "active_query_feedback_report.json"
    round_a.mkdir()
    round_b.mkdir()
    _write_jsonl(
        round_a / "active_queries.jsonl",
        [{"queue": "real", "query_id": "qa", "attack_id": "atk-a", "defense_id": "def-a"}],
    )
    _write_jsonl(round_a / "oracle_pairs.jsonl", [{"attack_id": "atk-a", "defense_id": "def-a", "attack_success": 1.0}])
    _write_jsonl(round_a / "oracle_results.jsonl", [{"status": "completed"}, {"status": "error"}])
    _write_jsonl(
        round_b / "active_queries.jsonl",
        [{"queue": "real", "query_id": "qb", "attack_id": "atk-b", "defense_id": "def-b"}],
    )
    _write_jsonl(round_b / "oracle_pairs.jsonl", [{"attack_id": "atk-b", "defense_id": "def-b", "attack_success": 0.0}])
    _write_jsonl(round_b / "oracle_results.jsonl", [{"status": "completed"}, {"status": "completed"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_active_query_feedback.py",
            "--round-dir",
            str(round_a),
            "--round-dir",
            str(round_b),
            "--out-report",
            str(out_report),
            "--max-oracle-result-error-rate",
            "1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["oracle_result_rows"] == 4
    assert report["oracle_result_errors"] == 1
    assert report["oracle_result_error_rate"] == 0.25
    assert report["real_queries"] == 2
    assert report["matched_real_queries"] == 2


def test_build_mask_explanation_validation_report_summarizes_hidden_slot_coverage(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "defense_id": "def-1",
                "hidden_count": 2,
                "defense_risk_report": {
                    "hidden_count": 2,
                    "learned_mask_score": 3.5,
                    "counter_attack_risk_report": {"expected_match_win": 0.4},
                    "mask_explanation": {
                        "hidden_count": 2,
                        "learned_mask_score": 3.5,
                        "learned_score_weight": 0.05,
                        "hidden_slot_explanations": [
                            {"team_index": 0, "slot_index": 1, "hero_id": 11, "learned_slot_score": 2.0},
                            {"team_index": 1, "slot_index": 3, "hero_id": 22, "learned_slot_score": 1.5},
                        ],
                        "top_learned_slots": [
                            {"team_index": 0, "slot_index": 1, "hero_id": 11, "learned_slot_score": 2.0}
                        ],
                    },
                },
            }
        ],
    )

    report = build_mask_explanation_validation_report(round_dir)

    assert report["schema_version"] == "mask_explanation_validation_report.v1"
    assert report["defenses"] == 1
    assert report["mask_explanation_rows"] == 1
    assert report["total_hidden_slots"] == 2
    assert report["explained_hidden_slots"] == 2
    assert report["hidden_explanation_coverage"] == 1.0
    assert report["counter_attack_risk_rows"] == 1
    assert report["mean_learned_mask_score"] == 3.5
    assert report["red_line_violations"] == []


def test_build_mask_explanation_validation_report_flags_missing_or_degenerate_masks(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "defense_id": "def-visible",
                "hidden_count": 0,
                "defense_risk_report": {
                    "hidden_count": 0,
                    "mask_explanation": {"hidden_count": 0, "hidden_slot_explanations": []},
                },
            },
            {
                "defense_id": "def-missing",
                "hidden_count": 2,
                "defense_risk_report": {
                    "hidden_count": 2,
                    "mask_explanation": {"hidden_count": 2, "hidden_slot_explanations": [{"team_index": 0}]},
                },
            },
        ],
    )

    report = build_mask_explanation_validation_report(round_dir, min_hidden_explanation_coverage=0.95)

    assert report["defenses_with_no_hidden_slots"] == 1
    assert report["hidden_explanation_coverage"] == 0.5
    assert "no_hidden_slots" in report["red_line_violations"]
    assert "hidden_slot_explanation_coverage_low" in report["red_line_violations"]
    assert "counter_attack_risk_missing" in report["red_line_violations"]


def test_report_mask_explanation_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_mask_explanation_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout


def test_build_belief_real_distribution_validation_report_summarizes_real_meta_usage(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_id": "atk-1",
                "defense_id": "def-1",
                "belief_domain_stats": [
                    ["real_record_count", 3.0],
                    ["real_exact_record_count", 3.0],
                    ["real_similar_record_count", 0.0],
                    ["real_similarity_mean", 1.0],
                    ["real_match_result_mean", 0.75],
                    ["candidate_count", 1.0],
                    ["weight_entropy_normalized", 0.0],
                    ["ranker_applied", 1.0],
                ],
            },
            {
                "attack_id": "atk-2",
                "defense_id": "def-1",
                "belief_domain_stats": [
                    ["real_record_count", 2.0],
                    ["real_exact_record_count", 0.0],
                    ["real_similar_record_count", 2.0],
                    ["real_similarity_mean", 0.5],
                    ["real_match_result_mean", 0.5],
                    ["candidate_count", 12.0],
                    ["weight_entropy_normalized", 0.8],
                    ["ranker_applied", 1.0],
                ],
            },
        ],
    )
    _write_jsonl(
        round_dir / "oracle_pairs.jsonl",
        [
            {"attack_id": "atk-1", "defense_id": "def-1", "attack_success": 0.8},
            {"attack_id": "atk-2", "defense_id": "def-1", "attack_success": 0.4},
        ],
    )

    report = build_belief_real_distribution_validation_report(
        round_dir,
        min_real_coverage=0.9,
        min_mean_real_records=1.0,
        min_mean_real_similarity=0.25,
        max_oracle_alignment_mae=0.2,
    )

    assert report["schema_version"] == "belief_real_distribution_validation_report.v1"
    assert report["candidates"] == 2
    assert report["belief_domain_stats_rows"] == 2
    assert report["real_distribution_rows"] == 2
    assert report["real_distribution_coverage"] == 1.0
    assert report["exact_real_rows"] == 1
    assert report["similar_real_rows"] == 1
    assert report["mean_real_record_count"] == 2.5
    assert report["mean_real_similarity"] == 0.75
    assert report["oracle_alignment_rows"] == 2
    assert report["oracle_alignment_mae"] == 0.075
    assert report["candidate_rows"][0]["alignment_abs_error"] == 0.05
    assert report["red_line_violations"] == []


def test_build_belief_real_distribution_validation_report_flags_missing_real_meta(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {"attack_id": "atk-1", "defense_id": "def-1"},
            {
                "attack_id": "atk-2",
                "defense_id": "def-1",
                "belief_domain_stats": [
                    ["real_record_count", 0.0],
                    ["real_similarity_mean", 0.0],
                    ["real_match_result_mean", 0.5],
                ],
            },
        ],
    )

    report = build_belief_real_distribution_validation_report(
        round_dir,
        min_real_coverage=0.5,
        min_mean_real_records=1.0,
        min_mean_real_similarity=0.1,
    )

    assert report["belief_domain_stats_rows"] == 1
    assert report["real_distribution_coverage"] == 0.0
    assert "belief_domain_stats_missing" in report["red_line_violations"]
    assert "real_distribution_coverage_low" in report["red_line_violations"]
    assert "real_record_count_low" in report["red_line_violations"]
    assert "real_similarity_low" in report["red_line_violations"]


def test_report_belief_real_distribution_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_belief_real_distribution_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout


def test_build_exploiter_effectiveness_report_summarizes_target_residual_feedback(tmp_path: Path):
    teacher_path = tmp_path / "attack_teacher.jsonl"
    _write_jsonl(
        teacher_path,
        [
            {
                "round_id": "round_0001",
                "attack_id": "atk-main",
                "attack_role": "main",
                "attack_success": 0.50,
                "target_defense_id": "def-1",
                "target_defense_hash": "hash-1",
                "target_baseline_break_rate": 0.45,
                "exploiter_residual_target": 0.05,
                "role_weight": 1.0,
            },
            {
                "round_id": "round_0001",
                "attack_id": "atk-exp",
                "attack_role": "exploiter",
                "attack_success": 0.80,
                "target_defense_id": "def-1",
                "target_defense_hash": "hash-1",
                "target_baseline_break_rate": 0.45,
                "exploiter_residual_target": 0.35,
                "role_weight": 1.60,
            },
            {
                "round_id": "round_0002",
                "attack_id": "atk-under",
                "attack_role": "underdog",
                "attack_success": 0.62,
                "target_defense_id": "def-2",
                "target_defense_hash": "hash-2",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.12,
                "role_weight": 1.47,
                "source": "active_real_query",
            },
        ],
    )

    report = build_exploiter_effectiveness_report([teacher_path])

    assert report["schema_version"] == "exploiter_effectiveness_report.v1"
    assert report["teacher_rows"] == 3
    assert report["target_feedback_rows"] == 3
    assert report["role_stats"]["exploiter"]["samples"] == 1
    assert report["role_stats"]["exploiter"]["mean_residual"] == 0.35
    assert report["role_stats"]["exploiter"]["positive_residual_rate"] == 1.0
    assert report["role_stats"]["underdog"]["source_counts"]["active_real_query"] == 1
    assert report["anti_meta"]["samples"] == 2
    assert report["anti_meta"]["mean_residual"] == 0.235
    assert report["anti_meta"]["residual_lift_vs_main"] == 0.185
    assert report["anti_meta"]["positive_residual_rate"] == 1.0
    assert report["red_line_violations"] == []


def test_build_exploiter_effectiveness_report_flags_missing_or_weak_exploiters(tmp_path: Path):
    teacher_path = tmp_path / "attack_teacher.jsonl"
    _write_jsonl(
        teacher_path,
        [
            {
                "attack_id": "atk-exp",
                "attack_role": "exploiter",
                "attack_success": 0.30,
                "target_baseline_break_rate": 0.40,
                "exploiter_residual_target": -0.10,
                "role_weight": 1.25,
            }
        ],
    )

    report = build_exploiter_effectiveness_report([teacher_path], min_positive_residual_rate=0.75)

    assert "target_feedback_coverage_low" in report["red_line_violations"]
    assert "anti_meta_residual_non_positive" in report["red_line_violations"]
    assert "anti_meta_positive_rate_low" in report["red_line_violations"]


def test_build_exploiter_effectiveness_report_tracks_round_trend_from_training_root(tmp_path: Path):
    root = tmp_path / "training"
    round1 = root / "round_0001"
    round2 = root / "round_0002"
    round1.mkdir(parents=True)
    round2.mkdir(parents=True)
    _write_jsonl(
        round1 / "attack_teacher.jsonl",
        [
            {
                "teacher_group_id": "round_0001:def-1",
                "attack_id": "atk-main-r1",
                "attack_role": "main",
                "attack_success": 0.50,
                "target_defense_id": "def-1",
                "target_defense_hash": "hash-1",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.0,
                "role_weight": 1.0,
            },
            {
                "teacher_group_id": "round_0001:def-1",
                "attack_id": "atk-exp-r1",
                "attack_role": "exploiter",
                "attack_success": 0.60,
                "target_defense_id": "def-1",
                "target_defense_hash": "hash-1",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.10,
                "role_weight": 1.35,
            },
        ],
    )
    _write_jsonl(
        round2 / "attack_teacher.jsonl",
        [
            {
                "teacher_group_id": "round_0002:def-2",
                "attack_id": "atk-exp-r2",
                "attack_role": "exploiter",
                "attack_success": 0.80,
                "target_defense_id": "def-2",
                "target_defense_hash": "hash-2",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.30,
                "role_weight": 1.55,
            }
        ],
    )

    report = build_exploiter_effectiveness_report(training_root=root, min_trend_delta=0.0)

    assert tuple(report["round_stats"]) == ("round_0001", "round_0002")
    assert report["round_stats"]["round_0001"]["anti_meta"]["mean_residual"] == 0.10
    assert report["round_stats"]["round_0002"]["anti_meta"]["mean_residual"] == 0.30
    assert report["trend"]["rounds"] == ["round_0001", "round_0002"]
    assert report["trend"]["anti_meta_mean_residuals"] == [0.10, 0.30]
    assert report["trend"]["first_anti_meta_mean_residual"] == 0.10
    assert report["trend"]["last_anti_meta_mean_residual"] == 0.30
    assert report["trend"]["delta_anti_meta_mean_residual"] == 0.20
    assert report["trend"]["slope_per_round"] == 0.20
    assert report["trend"]["improving"] is True
    assert "anti_meta_residual_trend_non_positive" not in report["red_line_violations"]


def test_build_exploiter_effectiveness_report_flags_non_improving_round_trend(tmp_path: Path):
    root = tmp_path / "training"
    for round_id, residual in (("round_0001", 0.20), ("round_0002", 0.10)):
        round_dir = root / round_id
        round_dir.mkdir(parents=True)
        _write_jsonl(
            round_dir / "attack_teacher.jsonl",
            [
                {
                    "teacher_group_id": f"{round_id}:def-1",
                    "attack_id": f"atk-exp-{round_id}",
                    "attack_role": "exploiter",
                    "attack_success": 0.50 + residual,
                    "target_defense_id": "def-1",
                    "target_defense_hash": "hash-1",
                    "target_baseline_break_rate": 0.50,
                    "exploiter_residual_target": residual,
                    "role_weight": 1.25,
                }
            ],
        )

    report = build_exploiter_effectiveness_report(training_root=root, min_trend_delta=0.0)

    assert report["trend"]["delta_anti_meta_mean_residual"] == -0.10
    assert report["trend"]["improving"] is False
    assert "anti_meta_residual_trend_non_positive" in report["red_line_violations"]


def test_report_exploiter_effectiveness_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_exploiter_effectiveness.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--teacher-jsonl" in result.stdout
    assert "--training-root" in result.stdout
    assert "--out-report" in result.stdout


def test_build_defense_anti_meta_effectiveness_report_summarizes_residual_feedback(tmp_path: Path):
    teacher_path = tmp_path / "defense_teacher.jsonl"
    _write_jsonl(
        teacher_path,
        [
            {
                "round_id": "round_0001",
                "defense_id": "def-main",
                "defense_role": "main",
                "survival_rate": 0.55,
                "meta_attack_success": 0.50,
                "anti_meta_residual_target": 0.05,
                "role_weight": 1.0,
            },
            {
                "round_id": "round_0001",
                "defense_id": "def-anti",
                "defense_role": "anti_meta",
                "survival_rate": 0.78,
                "meta_attack_success": 0.50,
                "anti_meta_residual_target": 0.28,
                "role_weight": 1.45,
                "source": "selfplay_orchestrator",
            },
            {
                "round_id": "round_0002",
                "defense_id": "def-real",
                "defense_role": "underdog",
                "survival_rate": 0.70,
                "meta_attack_success": 0.55,
                "anti_meta_residual_target": 0.15,
                "role_weight": 1.30,
                "source": "active_real_query",
            },
        ],
    )

    report = build_defense_anti_meta_effectiveness_report([teacher_path])

    assert report["schema_version"] == "defense_anti_meta_effectiveness_report.v1"
    assert report["teacher_rows"] == 3
    assert report["anti_meta_feedback_rows"] == 3
    assert report["anti_meta_feedback_coverage"] == 1.0
    assert report["role_stats"]["anti_meta"]["samples"] == 1
    assert report["role_stats"]["anti_meta"]["mean_residual"] == 0.28
    assert report["role_stats"]["underdog"]["source_counts"]["active_real_query"] == 1
    assert report["anti_meta"]["samples"] == 3
    assert report["anti_meta"]["mean_residual"] == 0.16
    assert report["anti_meta"]["positive_residual_rate"] == 1.0
    assert report["anti_meta"]["mean_survival_lift"] == 0.16
    assert report["red_line_violations"] == []


def test_build_defense_anti_meta_effectiveness_report_flags_missing_or_weak_feedback(tmp_path: Path):
    teacher_path = tmp_path / "defense_teacher.jsonl"
    _write_jsonl(
        teacher_path,
        [
            {
                "defense_id": "def-weak",
                "defense_role": "anti_meta",
                "survival_rate": 0.35,
                "meta_attack_success": 0.45,
                "anti_meta_residual_target": -0.10,
            },
            {
                "defense_id": "def-missing",
                "defense_role": "main",
            },
        ],
    )

    report = build_defense_anti_meta_effectiveness_report([teacher_path], min_positive_residual_rate=0.75)

    assert "anti_meta_feedback_coverage_low" in report["red_line_violations"]
    assert "defense_anti_meta_residual_non_positive" in report["red_line_violations"]
    assert "defense_anti_meta_positive_rate_low" in report["red_line_violations"]


def test_build_defense_anti_meta_effectiveness_report_tracks_round_trend_from_training_root(tmp_path: Path):
    root = tmp_path / "training"
    for round_id, residual in (("round_0001", 0.10), ("round_0002", 0.25)):
        round_dir = root / round_id
        round_dir.mkdir(parents=True)
        _write_jsonl(
            round_dir / "defense_teacher.jsonl",
            [
                {
                    "teacher_group_id": f"{round_id}:anti_meta",
                    "defense_id": f"def-{round_id}",
                    "defense_role": "anti_meta",
                    "survival_rate": 0.50 + residual,
                    "meta_attack_success": 0.50,
                    "anti_meta_residual_target": residual,
                }
            ],
        )

    report = build_defense_anti_meta_effectiveness_report(training_root=root, min_trend_delta=0.0)

    assert report["trend"]["rounds"] == ["round_0001", "round_0002"]
    assert report["trend"]["anti_meta_mean_residuals"] == [0.10, 0.25]
    assert report["trend"]["delta_anti_meta_mean_residual"] == 0.15
    assert report["trend"]["improving"] is True
    assert "defense_anti_meta_residual_trend_non_positive" not in report["red_line_violations"]


def test_report_defense_anti_meta_effectiveness_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_defense_anti_meta_effectiveness.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--teacher-jsonl" in result.stdout
    assert "--training-root" in result.stdout
    assert "--out-report" in result.stdout


def test_build_learned_exploiter_validation_report_combines_attack_and_defense_feedback(tmp_path: Path):
    selfplay_root = tmp_path / "selfplay"
    training_root = tmp_path / "training"
    (training_root / "round_0001").mkdir(parents=True)
    (training_root / "round_0002").mkdir(parents=True)
    selfplay_root.mkdir()
    (selfplay_root / "orchestrator_state.json").write_text(
        json.dumps(
            {
                "root_dir": str(selfplay_root),
                "training_dir": str(training_root),
                "rounds": [
                    {
                        "round_id": "round_0001",
                        "oracle_requests": 9,
                        "attack_proposal_checkpoint": str(training_root / "round_0001" / "attack_proposal.pt"),
                        "defense_proposal_checkpoint": str(training_root / "round_0001" / "defense_proposal.pt"),
                    },
                    {
                        "round_id": "round_0002",
                        "oracle_requests": 12,
                        "attack_proposal_checkpoint": str(training_root / "round_0002" / "attack_proposal.pt"),
                        "defense_proposal_checkpoint": str(training_root / "round_0002" / "defense_proposal.pt"),
                    },
                ],
                "latest_attack_proposal_checkpoint": str(training_root / "round_0002" / "attack_proposal.pt"),
                "latest_defense_proposal_checkpoint": str(training_root / "round_0002" / "defense_proposal.pt"),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        training_root / "round_0001" / "attack_teacher.jsonl",
        [
            {
                "round_id": "round_0001",
                "attack_id": "atk-exp-r1",
                "attack_role": "exploiter",
                "attack_success": 0.60,
                "target_defense_id": "def-1",
                "target_defense_hash": "hash-1",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.10,
            }
        ],
    )
    _write_jsonl(
        training_root / "round_0002" / "attack_teacher.jsonl",
        [
            {
                "round_id": "round_0002",
                "attack_id": "atk-exp-r2",
                "attack_role": "exploiter",
                "attack_success": 0.75,
                "target_defense_id": "def-2",
                "target_defense_hash": "hash-2",
                "target_baseline_break_rate": 0.50,
                "exploiter_residual_target": 0.25,
            }
        ],
    )
    _write_jsonl(
        training_root / "round_0001" / "defense_teacher.jsonl",
        [
            {
                "round_id": "round_0001",
                "defense_id": "def-anti-r1",
                "defense_role": "anti_meta",
                "survival_rate": 0.62,
                "meta_attack_success": 0.50,
                "anti_meta_residual_target": 0.12,
            }
        ],
    )
    _write_jsonl(
        training_root / "round_0002" / "defense_teacher.jsonl",
        [
            {
                "round_id": "round_0002",
                "defense_id": "def-anti-r2",
                "defense_role": "anti_meta",
                "survival_rate": 0.72,
                "meta_attack_success": 0.50,
                "anti_meta_residual_target": 0.22,
            }
        ],
    )

    report = build_learned_exploiter_validation_report(
        selfplay_root=selfplay_root,
        training_root=training_root,
        min_rounds=2,
        min_oracle_requests=20,
        min_attack_trend_delta=0.0,
        min_defense_trend_delta=0.0,
    )

    assert report["schema_version"] == "learned_exploiter_validation_report.v1"
    assert report["rounds"] == 2
    assert report["oracle_requests"] == 21
    assert report["latest_attack_proposal_checkpoint"].endswith("attack_proposal.pt")
    assert report["latest_defense_proposal_checkpoint"].endswith("defense_proposal.pt")
    assert report["exploiter_report"]["trend"]["delta_anti_meta_mean_residual"] == 0.15
    assert report["defense_anti_meta_report"]["trend"]["delta_anti_meta_mean_residual"] == 0.10
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []


def test_build_learned_exploiter_validation_report_flags_missing_scale_and_checkpoints(tmp_path: Path):
    selfplay_root = tmp_path / "selfplay"
    training_root = tmp_path / "training"
    selfplay_root.mkdir()
    training_root.mkdir()
    (selfplay_root / "orchestrator_state.json").write_text(
        json.dumps(
            {
                "root_dir": str(selfplay_root),
                "training_dir": str(training_root),
                "rounds": [{"round_id": "round_0001", "oracle_requests": 3}],
                "latest_attack_proposal_checkpoint": None,
                "latest_defense_proposal_checkpoint": None,
            }
        ),
        encoding="utf-8",
    )

    report = build_learned_exploiter_validation_report(
        selfplay_root=selfplay_root,
        training_root=training_root,
        min_rounds=2,
        min_oracle_requests=10,
    )

    assert report["production_ready"] is False
    assert "validation_rounds_low" in report["red_line_violations"]
    assert "oracle_requests_low" in report["red_line_violations"]
    assert "missing_latest_attack_checkpoint" in report["red_line_violations"]
    assert "missing_latest_defense_checkpoint" in report["red_line_violations"]
    assert "attack_no_attack_teacher_rows" in report["red_line_violations"]
    assert "defense_no_defense_teacher_rows" in report["red_line_violations"]


def test_report_learned_exploiter_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_learned_exploiter_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--selfplay-root" in result.stdout
    assert "--training-root" in result.stdout
    assert "--out-report" in result.stdout


def test_build_production_readiness_report_combines_clean_reports(tmp_path: Path):
    learned_report = tmp_path / "learned_exploiter_validation_report.json"
    learned_report.write_text(
        json.dumps(
            {
                "schema_version": "learned_exploiter_validation_report.v1",
                "module": "LearnedExploiterValidationReport",
                "production_ready": True,
                "red_line_violations": [],
            }
        ),
        encoding="utf-8",
    )
    active_report = tmp_path / "active_query_feedback_report.json"
    active_report.write_text(
        json.dumps(
            {
                "schema_version": "active_query_feedback_report.v1",
                "module": "ActiveQueryFeedbackReport",
                "red_line_violations": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_production_readiness_report(
        [learned_report, active_report],
        required_schema_versions=(
            "learned_exploiter_validation_report.v1",
            "active_query_feedback_report.v1",
        ),
    )

    assert report["schema_version"] == "production_readiness_report.v1"
    assert report["module"] == "ProductionReadinessReport"
    assert report["reports"] == 2
    assert report["readable_reports"] == 2
    assert report["clean_reports"] == 2
    assert report["red_line_reports"] == 0
    assert report["production_ready_false_reports"] == 0
    assert report["clean_report_rate"] == 1.0
    assert report["missing_required_schema_versions"] == []
    assert report["schema_counts"] == {
        "active_query_feedback_report.v1": 1,
        "learned_exploiter_validation_report.v1": 1,
    }
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []


def test_build_production_readiness_report_counts_nested_real_calibration_ingestions(tmp_path: Path):
    real_calibration_report = tmp_path / "real_calibration_report.json"
    real_calibration_report.write_text(
        json.dumps(
            {
                "db_jsonl": "real_meta.jsonl",
                "ingestions": [
                    {
                        "schema_version": "real_calibration_ingestion_summary.v1",
                        "module": "RealCalibrationIngestionSummary",
                        "round_dir": "round_0001",
                        "db_path": "real_meta.jsonl",
                        "round_id": "round_0001",
                        "records_added": 1,
                        "skipped_pairs": 0,
                        "mean_match_result": 0.75,
                        "season": "S29",
                        "server": "oracle_backend",
                        "source_kind": "league_round_artifact",
                    }
                ],
                "total_records": 1,
                "drift": None,
            }
        ),
        encoding="utf-8",
    )

    report = build_production_readiness_report(
        [real_calibration_report],
        required_schema_versions=("real_calibration_ingestion_summary.v1",),
    )

    assert report["schema_counts"] == {"real_calibration_ingestion_summary.v1": 1}
    assert report["missing_required_schema_versions"] == []
    assert report["report_rows"][0]["schema_versions"] == ["real_calibration_ingestion_summary.v1"]
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []


def test_build_production_readiness_report_flags_missing_bad_and_red_reports(tmp_path: Path):
    clean_report = tmp_path / "clean.json"
    clean_report.write_text(
        json.dumps(
            {
                "schema_version": "learned_exploiter_validation_report.v1",
                "production_ready": True,
                "red_line_violations": [],
            }
        ),
        encoding="utf-8",
    )
    red_report = tmp_path / "red.json"
    red_report.write_text(
        json.dumps(
            {
                "schema_version": "underdog_residual_validation_report.v1",
                "production_ready": False,
                "red_line_violations": ["defense_underdog_rows_missing"],
            }
        ),
        encoding="utf-8",
    )
    broken_report = tmp_path / "broken.json"
    broken_report.write_text("[1, 2, 3]\n", encoding="utf-8")

    report = build_production_readiness_report(
        [clean_report, red_report, broken_report, tmp_path / "missing.json"],
        required_schema_versions=(
            "learned_exploiter_validation_report.v1",
            "active_query_feedback_report.v1",
        ),
        min_clean_report_rate=0.75,
    )

    assert report["reports"] == 4
    assert report["readable_reports"] == 2
    assert report["read_error_reports"] == 2
    assert report["clean_reports"] == 1
    assert report["red_line_reports"] == 1
    assert report["production_ready_false_reports"] == 1
    assert report["missing_required_schema_versions"] == ["active_query_feedback_report.v1"]
    assert report["production_ready"] is False
    assert "report_read_error" in report["red_line_violations"]
    assert "required_schema_missing" in report["red_line_violations"]
    assert "red_line_reports_present" in report["red_line_violations"]
    assert "production_ready_false" in report["red_line_violations"]
    assert "clean_report_rate_low" in report["red_line_violations"]


def test_report_production_readiness_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_production_readiness.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--report-json" in result.stdout
    assert "--required-schema-version" in result.stdout
    assert "--out-report" in result.stdout


def test_build_v4_conformance_validation_report_maps_required_evidence(tmp_path: Path):
    reports = {
        "learned.json": {"schema_version": "learned_exploiter_validation_report.v1", "production_ready": True},
        "league.json": {"schema_version": "league_selfplay_health_report.v1", "production_ready": True},
        "ablation_plan.json": {"schema_version": "v4_ablation_experiment_plan.v1"},
        "ablation_report.json": {"schema_version": "ablation_suite_report.v1", "red_line_violations": []},
        "real_ingest.json": {
            "ingestions": [{"schema_version": "real_calibration_ingestion_summary.v1"}],
            "drift": {"schema_version": "version_drift_report.v1", "drift_detected": False},
        },
        "real_samples.json": {"schema_version": "real_calibration_sample_build_summary.v1"},
        "real_holdout.json": {"schema_version": "real_calibration_validation_report.v1", "production_ready": True},
        "active_query.json": {"schema_version": "active_query_feedback_report.v1"},
        "active_real.json": {
            "schema_version": "active_real_query_dispatch_validation_report.v1",
            "production_ready": True,
        },
        "mask.json": {"schema_version": "mask_explanation_validation_report.v1"},
        "belief.json": {"schema_version": "belief_real_distribution_validation_report.v1"},
    }
    paths = []
    for filename, payload in reports.items():
        path = tmp_path / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    report = build_v4_conformance_validation_report(paths)

    assert report["schema_version"] == "v4_conformance_validation_report.v1"
    assert report["module"] == "V4ConformanceValidationReport"
    assert report["production_ready"] is True
    assert report["red_line_violations"] == []
    requirement_rows = {row["requirement_id"]: row for row in report["requirements"]}
    assert requirement_rows["learned_exploiter_anti_meta"]["status"] == "pass"
    assert requirement_rows["learned_exploiter_anti_meta"]["evidence_count"] == 2
    assert requirement_rows["full_ablation_feedback"]["status"] == "pass"
    assert requirement_rows["real_calibration_holdout"]["status"] == "pass"
    assert requirement_rows["active_real_query_dispatch"]["status"] == "pass"
    assert requirement_rows["mask_belief_validation"]["status"] == "pass"
    assert report["passed_requirements"] == report["requirements_total"]


def test_build_v4_conformance_validation_report_flags_missing_and_red_evidence(tmp_path: Path):
    learned_report = tmp_path / "learned.json"
    learned_report.write_text(
        json.dumps(
            {
                "schema_version": "learned_exploiter_validation_report.v1",
                "production_ready": False,
                "red_line_violations": ["oracle_requests_low"],
            }
        ),
        encoding="utf-8",
    )
    active_report = tmp_path / "active_real.json"
    active_report.write_text(
        json.dumps(
            {
                "schema_version": "active_real_query_dispatch_validation_report.v1",
                "production_ready": True,
                "red_line_violations": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_v4_conformance_validation_report(
        [learned_report, active_report, tmp_path / "missing_report.json"],
    )

    assert report["production_ready"] is False
    assert "report_read_error" in report["red_line_violations"]
    assert "learned_exploiter_anti_meta_red_lines_present" in report["red_line_violations"]
    assert "learned_exploiter_anti_meta_production_not_ready" in report["red_line_violations"]
    assert "full_ablation_feedback_evidence_missing" in report["red_line_violations"]
    assert "real_calibration_holdout_evidence_missing" in report["red_line_violations"]
    assert "mask_belief_validation_evidence_missing" in report["red_line_violations"]
    requirement_rows = {row["requirement_id"]: row for row in report["requirements"]}
    assert requirement_rows["learned_exploiter_anti_meta"]["child_red_line_violations"] == ["oracle_requests_low"]
    assert "league_selfplay_health_report.v1" in requirement_rows["learned_exploiter_anti_meta"]["missing_schema_versions"]
    assert "active_query_feedback_report.v1" in requirement_rows["active_real_query_dispatch"]["missing_schema_versions"]


def test_report_v4_conformance_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_v4_conformance_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--report-json" in result.stdout
    assert "--reports-root" in result.stdout
    assert "--out-report" in result.stdout


def test_report_v4_conformance_validation_script_reports_root_includes_plan_json(tmp_path: Path):
    (tmp_path / "v4_ablation_experiment_plan.json").write_text(
        json.dumps({"schema_version": "v4_ablation_experiment_plan.v1"}),
        encoding="utf-8",
    )
    (tmp_path / "ablation_suite_report.json").write_text(
        json.dumps({"schema_version": "ablation_suite_report.v1"}),
        encoding="utf-8",
    )
    out_report = tmp_path / "v4_conformance_validation_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/report_v4_conformance_validation.py",
            "--reports-root",
            str(tmp_path),
            "--out-report",
            str(out_report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["schema_counts"]["v4_ablation_experiment_plan.v1"] == 1
    requirement_rows = {row["requirement_id"]: row for row in report["requirements"]}
    assert requirement_rows["full_ablation_feedback"]["missing_schema_versions"] == []


def test_build_attack_oracle_failure_validation_report_summarizes_failure_annotations(tmp_path: Path):
    oracle_output = tmp_path / "attack_oracle_failure.json"
    oracle_output.write_text(
        json.dumps(
            {
                "schema_version": "attack_oracle_output.v1",
                "risk_report": {
                    "failure": "no legal attack candidates",
                    "failure_code": "NO_LEGAL_ATTACK_CANDIDATES",
                    "failure_stage": "candidate_generation",
                    "failure_context": {"generated_candidate_count": 0},
                },
                "diagnostics": [
                    {
                        "code": "NO_LEGAL_ATTACK_CANDIDATES",
                        "stage": "candidate_generation",
                        "severity": "error",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_id": "atk-ok",
                "defense_id": "def-1",
                "attack_risk_report": {"expected_match_win": 0.75, "belief_case_count": 3},
            }
        ],
    )

    report = build_attack_oracle_failure_validation_report(
        oracle_output_paths=(oracle_output,),
        round_dirs=(round_dir,),
    )

    assert report["schema_version"] == "attack_oracle_failure_validation_report.v1"
    assert report["oracle_outputs"] == 1
    assert report["candidate_rows"] == 1
    assert report["checked_rows"] == 2
    assert report["failure_rows"] == 1
    assert report["annotated_failure_rows"] == 1
    assert report["diagnostic_failure_rows"] == 1
    assert report["failure_annotation_coverage"] == 1.0
    assert report["failure_diagnostic_coverage"] == 1.0
    assert report["failure_stage_counts"] == {"candidate_generation": 1}
    assert report["failure_code_counts"] == {"NO_LEGAL_ATTACK_CANDIDATES": 1}
    assert report["normal_risk_report_rows"] == 1
    assert report["red_line_violations"] == []


def test_build_attack_oracle_failure_validation_report_flags_missing_failure_annotations(tmp_path: Path):
    oracle_output = tmp_path / "attack_oracle_bad_failure.json"
    oracle_output.write_text(
        json.dumps(
            {
                "schema_version": "attack_oracle_output.v1",
                "risk_report": {"failure": "no candidates"},
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_attack_oracle_failure_validation_report(
        oracle_output_paths=(oracle_output,),
        min_failure_annotation_coverage=1.0,
        min_failure_diagnostic_coverage=1.0,
    )

    assert report["failure_rows"] == 1
    assert report["annotated_failure_rows"] == 0
    assert "failure_code_missing" in report["red_line_violations"]
    assert "failure_stage_missing" in report["red_line_violations"]
    assert "failure_diagnostic_coverage_low" in report["red_line_violations"]
    assert "failure_annotation_coverage_low" in report["red_line_violations"]


def test_report_attack_oracle_failure_validation_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_attack_oracle_failure_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--oracle-output-json" in result.stdout
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout


def test_report_league_round_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/report_league_round.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--round-dir" in result.stdout
    assert "--out-report" in result.stdout
