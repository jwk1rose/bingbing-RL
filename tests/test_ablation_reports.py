from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from masked_team_league.reporting.ablation import (
    V4_REQUIRED_ABLATION_VARIANTS,
    build_ablation_suite_report,
    build_v4_ablation_experiment_plan,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_round_artifacts(
    round_dir: Path,
    *,
    attack_top1: float,
    defense_break_rate: float,
    expected_match_win: float,
    worst_case_match_win: float,
    active_queries: int,
) -> None:
    round_dir.mkdir(parents=True)
    (round_dir / "summary.json").write_text(
        json.dumps(
            {
                "oracle_requests": 9,
                "best_attack_success": attack_top1,
                "worst_defense_break_rate": defense_break_rate,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        round_dir / "candidates.jsonl",
        [
            {
                "attack_role": "main",
                "belief_entropy": 1.0,
                "attack_risk_report": {
                    "expected_match_win": expected_match_win,
                    "worst_case_match_win": worst_case_match_win,
                    "backup_attack_count": 1,
                    "belief_case_count": 3,
                },
            }
        ],
    )
    _write_jsonl(
        round_dir / "scored_defenses.jsonl",
        [
            {
                "break_rate": defense_break_rate,
                "ambiguity_score": 1.5,
                "defense_risk_report": {
                    "estimated_break_rate": defense_break_rate,
                    "estimated_survival_rate": 1.0 - defense_break_rate,
                    "hidden_count": 6,
                    "backup_defense_count": 1,
                },
            }
        ],
    )
    _write_jsonl(round_dir / "oracle_results.jsonl", [{"status": "completed"} for _ in range(9)])
    _write_jsonl(
        round_dir / "active_queries.jsonl",
        [{"query_id": f"q{i}", "queue": "sim", "score": 1.0} for i in range(active_queries)],
    )
    (round_dir / "league_state.json").write_text(
        json.dumps(
            {
                "attack_pool": [{"diversity_cluster": "attack-a"}, {"diversity_cluster": "attack-b"}],
                "defense_pool": [{"diversity_cluster": "def-a"}],
            }
        ),
        encoding="utf-8",
    )


def test_ablation_suite_reports_required_variant_gaps_and_metric_deltas(tmp_path: Path):
    baseline = tmp_path / "baseline"
    no_equips = tmp_path / "no_equipment_stars"
    _write_round_artifacts(
        baseline,
        attack_top1=0.8,
        defense_break_rate=0.3,
        expected_match_win=0.7,
        worst_case_match_win=0.5,
        active_queries=2,
    )
    _write_round_artifacts(
        no_equips,
        attack_top1=0.55,
        defense_break_rate=0.45,
        expected_match_win=0.5,
        worst_case_match_win=0.25,
        active_queries=0,
    )

    report = build_ablation_suite_report(
        {"baseline": baseline, "no_equipment_stars": no_equips},
        baseline_variant="baseline",
        date="2026-07-05",
    ).to_json_dict()

    assert set(report["variants"]) == {"baseline", "no_equipment_stars"}
    assert "no_position_features" in report["missing_required_variants"]
    assert "no_equipment_stars" not in report["missing_required_variants"]
    assert set(V4_REQUIRED_ABLATION_VARIANTS) >= {"baseline", "no_active_perception"}
    assert report["variant_reports"]["baseline"]["key_metrics"]["attack_top1"] == 0.8
    assert report["variant_reports"]["baseline"]["key_metrics"]["attack_expected_match_win_mean"] == 0.7
    assert report["variant_reports"]["no_equipment_stars"]["key_metrics"]["active_query_count"] == 0
    assert report["deltas_vs_baseline"]["no_equipment_stars"]["attack_top1"] == -0.25
    assert report["deltas_vs_baseline"]["no_equipment_stars"]["defense_attack_success"] == 0.15
    assert report["deltas_vs_baseline"]["no_equipment_stars"]["attack_worst_case_match_win_mean"] == -0.25


def test_run_ablation_suite_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_ablation_suite", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--variant" in result.stdout
    assert "--baseline" in result.stdout
    assert "--require-v4-variants" in result.stdout


def test_v4_ablation_experiment_plan_includes_required_variants_and_controls(tmp_path: Path):
    plan = build_v4_ablation_experiment_plan(
        root_dir=tmp_path / "suite",
        backend="http://127.0.0.1:18281",
        heroes_json=tmp_path / "heroes.json",
        decoded_dir=tmp_path / "decoded",
        suite_id="suite_smoke",
        teams=3,
        defenses=1,
        attacks_per_defense=4,
        oracle_top_k=1,
        defense_roster_candidates=2,
        defense_masks_per_roster=3,
        defense_max_masks_per_roster=16,
        seed=123,
    ).to_json_dict()

    assert plan["schema_version"] == "v4_ablation_experiment_plan.v1"
    assert plan["suite_id"] == "suite_smoke"
    assert tuple(plan["required_variants"]) == V4_REQUIRED_ABLATION_VARIANTS
    assert tuple(variant["variant_id"] for variant in plan["variants"]) == V4_REQUIRED_ABLATION_VARIANTS
    assert not plan["missing_required_variants"]

    variants = {variant["variant_id"]: variant for variant in plan["variants"]}
    baseline_command = variants["baseline"]["command"]
    assert baseline_command[:3] == [sys.executable, "-m", "masked_team_league.cli.commands.run_league_round"]
    assert "--backend" in baseline_command
    assert "--heroes-json" in baseline_command
    assert "--decoded-dir" in baseline_command

    no_underdog = variants["no_underdog_objective"]
    assert "role_loop_without_underdog" in no_underdog["implemented_controls"]
    assert _command_value(no_underdog["command"], "--attack-role") == "main"
    assert _command_value(no_underdog["command"], "--defense-role") == "main"

    no_active = variants["no_active_perception"]
    assert "active_perception_disabled" in no_active["implemented_controls"]
    assert _command_value(no_active["command"], "--active-sim-keep") == "0"
    assert _command_value(no_active["command"], "--active-real-keep") == "0"

    no_mask = variants["no_mask_ambiguity"]
    assert "single_mask_per_defense_roster" in no_mask["implemented_controls"]
    assert _command_value(no_mask["command"], "--defense-masks-per-roster") == "1"
    assert _command_value(no_mask["command"], "--defense-max-masks-per-roster") == "1"

    no_position = variants["no_position_features"]
    assert no_position["implemented_controls"] == ["position_features_disabled"]
    assert no_position["control_status"] == "implemented"
    assert "--disable-position-features" in no_position["command"]

    no_equips = variants["no_equipment_stars"]
    assert no_equips["implemented_controls"] == ["equipment_star_features_disabled"]
    assert "--disable-equipment-star-features" in no_equips["command"]

    no_future = variants["no_future_feasibility_mask"]
    assert no_future["implemented_controls"] == ["future_feasibility_action_mask_disabled"]
    assert "--disable-future-feasibility-mask" in no_future["command"]

    no_real = variants["no_real_calibration"]
    assert no_real["implemented_controls"] == ["real_calibration_disabled"]
    assert "--disable-real-calibration" in no_real["command"]


def test_run_v4_ablation_experiments_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_v4_ablation_experiments", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--out-plan" in result.stdout
    assert "--execute" in result.stdout
    assert "--variant" in result.stdout


def _command_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]
