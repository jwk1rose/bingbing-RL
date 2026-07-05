#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masked_team_league.training_schedule import build_v4_training_schedule, run_training_schedule, write_training_schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally execute a v4 model training schedule.")
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--out-schedule", type=Path, required=True)
    parser.add_argument("--out-status", type=Path, required=True)
    parser.add_argument("--single-team-train-jsonl", type=Path, default=None)
    parser.add_argument("--single-team-holdout-jsonl", type=Path, default=None)
    parser.add_argument("--belief-train-jsonl", type=Path, default=None)
    parser.add_argument("--belief-holdout-jsonl", type=Path, default=None)
    parser.add_argument("--belief-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--belief-round-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--attack-teacher-jsonl", type=Path, default=None)
    parser.add_argument("--defense-teacher-jsonl", type=Path, default=None)
    parser.add_argument("--mask-teacher-jsonl", type=Path, default=None)
    parser.add_argument("--mask-explanation-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--mask-explanation-validation-report", type=Path, default=None)
    parser.add_argument("--mask-min-hidden-explanation-coverage", type=float, default=0.95)
    parser.add_argument("--belief-real-validation-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--belief-real-distribution-report", type=Path, default=None)
    parser.add_argument("--belief-min-real-coverage", type=float, default=0.50)
    parser.add_argument("--belief-min-mean-real-records", type=float, default=1.0)
    parser.add_argument("--belief-min-mean-real-similarity", type=float, default=0.25)
    parser.add_argument("--belief-max-oracle-alignment-mae", type=float, default=0.35)
    parser.add_argument("--exploiter-training-root", type=Path, default=None)
    parser.add_argument("--exploiter-effectiveness-report", type=Path, default=None)
    parser.add_argument("--exploiter-min-target-coverage", type=float, default=0.95)
    parser.add_argument("--exploiter-min-positive-residual-rate", type=float, default=0.50)
    parser.add_argument("--exploiter-min-trend-delta", type=float, default=None)
    parser.add_argument("--defense-anti-meta-training-root", type=Path, default=None)
    parser.add_argument("--defense-anti-meta-effectiveness-report", type=Path, default=None)
    parser.add_argument("--defense-anti-meta-min-feedback-coverage", type=float, default=0.95)
    parser.add_argument("--defense-anti-meta-min-positive-residual-rate", type=float, default=0.50)
    parser.add_argument("--defense-anti-meta-min-mean-residual", type=float, default=0.0)
    parser.add_argument("--defense-anti-meta-min-trend-delta", type=float, default=None)
    parser.add_argument("--learned-exploiter-selfplay-root", type=Path, default=None)
    parser.add_argument("--learned-exploiter-training-root", type=Path, default=None)
    parser.add_argument("--learned-exploiter-validation-report", type=Path, default=None)
    parser.add_argument("--learned-exploiter-min-rounds", type=int, default=2)
    parser.add_argument("--learned-exploiter-min-oracle-requests", type=int, default=1)
    parser.add_argument("--learned-exploiter-no-require-latest-checkpoints", action="store_true")
    parser.add_argument("--learned-exploiter-min-attack-trend-delta", type=float, default=None)
    parser.add_argument("--learned-exploiter-min-defense-trend-delta", type=float, default=None)
    parser.add_argument("--attack-oracle-failure-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--attack-oracle-failure-output-json", type=Path, action="append", default=[])
    parser.add_argument("--attack-oracle-failure-validation-report", type=Path, default=None)
    parser.add_argument("--attack-oracle-min-failure-annotation-coverage", type=float, default=1.0)
    parser.add_argument("--attack-oracle-min-failure-diagnostic-coverage", type=float, default=1.0)
    parser.add_argument("--active-query-feedback-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--active-query-feedback-report", type=Path, default=None)
    parser.add_argument("--active-query-min-matched-coverage", type=float, default=1.0)
    parser.add_argument("--active-query-max-oracle-error-rate", type=float, default=0.0)
    parser.add_argument("--active-query-min-real-query-count", type=int, default=0)
    parser.add_argument("--active-real-dispatch-validation-json", type=Path, action="append", default=[])
    parser.add_argument("--active-real-dispatch-validation-report", type=Path, default=None)
    parser.add_argument("--active-real-dispatch-min-reports", type=int, default=1)
    parser.add_argument("--active-real-dispatch-min-dispatched-pairs", type=int, default=1)
    parser.add_argument("--active-real-dispatch-min-completion-rate", type=float, default=1.0)
    parser.add_argument("--data-engineering-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--data-engineering-validation-report", type=Path, default=None)
    parser.add_argument("--data-engineering-min-metadata-coverage", type=float, default=1.0)
    parser.add_argument("--data-engineering-min-core-table-coverage", type=float, default=1.0)
    parser.add_argument("--data-engineering-min-artifact-hash-coverage", type=float, default=1.0)
    parser.add_argument("--underdog-residual-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--underdog-residual-validation-report", type=Path, default=None)
    parser.add_argument("--underdog-min-attack-residual-coverage", type=float, default=0.95)
    parser.add_argument("--underdog-min-defense-residual-coverage", type=float, default=0.95)
    parser.add_argument("--underdog-min-mean-attack-residual-bonus", type=float, default=0.0)
    parser.add_argument("--underdog-min-mean-defense-residual-bonus", type=float, default=0.0)
    parser.add_argument("--league-health-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--league-selfplay-health-report", type=Path, default=None)
    parser.add_argument("--league-health-min-attack-pool", type=int, default=1)
    parser.add_argument("--league-health-min-defense-pool", type=int, default=1)
    parser.add_argument("--league-health-min-total-clusters", type=int, default=2)
    parser.add_argument("--league-health-min-payoff-density", type=float, default=0.0)
    parser.add_argument("--league-health-required-attack-role", action="append", default=None)
    parser.add_argument("--league-health-required-defense-role", action="append", default=None)
    parser.add_argument("--league-health-min-active-pool-fraction", type=float, default=0.0)
    parser.add_argument("--league-health-min-new-attack-strength-delta", type=float, default=None)
    parser.add_argument("--league-health-min-new-defense-strength-delta", type=float, default=None)
    parser.add_argument("--production-readiness-report-json", type=Path, action="append", default=[])
    parser.add_argument("--production-readiness-include-scheduled-reports", action="store_true")
    parser.add_argument("--production-readiness-report", type=Path, default=None)
    parser.add_argument("--production-readiness-required-schema-version", action="append", default=[])
    parser.add_argument("--production-readiness-min-clean-report-rate", type=float, default=1.0)
    parser.add_argument("--production-readiness-no-require-production-ready", action="store_true")
    parser.add_argument("--v4-conformance-report-json", type=Path, action="append", default=[])
    parser.add_argument("--v4-conformance-include-scheduled-reports", action="store_true")
    parser.add_argument("--v4-conformance-validation-report", type=Path, default=None)
    parser.add_argument("--real-round-dir", type=Path, action="append", default=[])
    parser.add_argument("--real-meta-db-jsonl", type=Path, default=None)
    parser.add_argument("--real-calibration-report", type=Path, default=None)
    parser.add_argument("--active-real-feedback-dir", type=Path, action="append", default=[])
    parser.add_argument("--build-real-calibration-samples-jsonl", type=Path, default=None)
    parser.add_argument("--build-real-calibration-samples-report", type=Path, default=None)
    parser.add_argument("--real-rank-segment", default="unknown")
    parser.add_argument("--real-server", default="oracle_backend")
    parser.add_argument("--real-season", default=None)
    parser.add_argument("--real-timestamp", type=float, default=None)
    parser.add_argument("--drift-baseline-season", default=None)
    parser.add_argument("--drift-current-season", default=None)
    parser.add_argument("--drift-delta-threshold", type=float, default=0.15)
    parser.add_argument("--drift-min-overlap", type=float, default=0.20)
    parser.add_argument("--real-calibration-validation-samples-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--real-calibration-validation-model-json", type=Path, default=None)
    parser.add_argument("--real-calibration-validation-report", type=Path, default=None)
    parser.add_argument("--real-calibration-validation-min-samples", type=int, default=100)
    parser.add_argument("--real-calibration-min-brier-improvement", type=float, default=0.0)
    parser.add_argument("--real-calibration-min-ece-improvement", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--proposal-model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--monitor-resources", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    schedule = build_v4_training_schedule(
        schedule_id=args.schedule_id,
        root_dir=args.root_dir,
        heroes_json=args.heroes_json,
        decoded_dir=args.decoded_dir,
        registry_path=args.registry,
        single_team_train_jsonl=args.single_team_train_jsonl,
        single_team_holdout_jsonl=args.single_team_holdout_jsonl,
        belief_train_jsonl=args.belief_train_jsonl,
        belief_holdout_jsonl=args.belief_holdout_jsonl,
        belief_round_dirs=args.belief_round_dir,
        belief_round_holdout_fraction=args.belief_round_holdout_fraction,
        attack_teacher_jsonl=args.attack_teacher_jsonl,
        defense_teacher_jsonl=args.defense_teacher_jsonl,
        mask_teacher_jsonl=args.mask_teacher_jsonl,
        mask_explanation_round_dirs=args.mask_explanation_round_dir,
        mask_explanation_validation_report=args.mask_explanation_validation_report,
        mask_min_hidden_explanation_coverage=args.mask_min_hidden_explanation_coverage,
        belief_real_validation_round_dirs=args.belief_real_validation_round_dir,
        belief_real_distribution_report=args.belief_real_distribution_report,
        belief_min_real_coverage=args.belief_min_real_coverage,
        belief_min_mean_real_records=args.belief_min_mean_real_records,
        belief_min_mean_real_similarity=args.belief_min_mean_real_similarity,
        belief_max_oracle_alignment_mae=args.belief_max_oracle_alignment_mae,
        exploiter_training_root=args.exploiter_training_root,
        exploiter_effectiveness_report=args.exploiter_effectiveness_report,
        exploiter_min_target_coverage=args.exploiter_min_target_coverage,
        exploiter_min_positive_residual_rate=args.exploiter_min_positive_residual_rate,
        exploiter_min_trend_delta=args.exploiter_min_trend_delta,
        defense_anti_meta_training_root=args.defense_anti_meta_training_root,
        defense_anti_meta_effectiveness_report=args.defense_anti_meta_effectiveness_report,
        defense_anti_meta_min_feedback_coverage=args.defense_anti_meta_min_feedback_coverage,
        defense_anti_meta_min_positive_residual_rate=args.defense_anti_meta_min_positive_residual_rate,
        defense_anti_meta_min_mean_residual=args.defense_anti_meta_min_mean_residual,
        defense_anti_meta_min_trend_delta=args.defense_anti_meta_min_trend_delta,
        learned_exploiter_selfplay_root=args.learned_exploiter_selfplay_root,
        learned_exploiter_training_root=args.learned_exploiter_training_root,
        learned_exploiter_validation_report=args.learned_exploiter_validation_report,
        learned_exploiter_min_rounds=args.learned_exploiter_min_rounds,
        learned_exploiter_min_oracle_requests=args.learned_exploiter_min_oracle_requests,
        learned_exploiter_require_latest_checkpoints=not args.learned_exploiter_no_require_latest_checkpoints,
        learned_exploiter_min_attack_trend_delta=args.learned_exploiter_min_attack_trend_delta,
        learned_exploiter_min_defense_trend_delta=args.learned_exploiter_min_defense_trend_delta,
        attack_oracle_failure_round_dirs=args.attack_oracle_failure_round_dir,
        attack_oracle_failure_output_jsons=args.attack_oracle_failure_output_json,
        attack_oracle_failure_validation_report=args.attack_oracle_failure_validation_report,
        attack_oracle_min_failure_annotation_coverage=args.attack_oracle_min_failure_annotation_coverage,
        attack_oracle_min_failure_diagnostic_coverage=args.attack_oracle_min_failure_diagnostic_coverage,
        active_query_feedback_round_dirs=args.active_query_feedback_round_dir,
        active_query_feedback_report=args.active_query_feedback_report,
        active_query_min_matched_coverage=args.active_query_min_matched_coverage,
        active_query_max_oracle_error_rate=args.active_query_max_oracle_error_rate,
        active_query_min_real_query_count=args.active_query_min_real_query_count,
        active_real_dispatch_validation_jsons=args.active_real_dispatch_validation_json,
        active_real_dispatch_validation_report=args.active_real_dispatch_validation_report,
        active_real_dispatch_min_reports=args.active_real_dispatch_min_reports,
        active_real_dispatch_min_dispatched_pairs=args.active_real_dispatch_min_dispatched_pairs,
        active_real_dispatch_min_completion_rate=args.active_real_dispatch_min_completion_rate,
        active_real_feedback_dirs=args.active_real_feedback_dir,
        build_real_calibration_samples_jsonl=args.build_real_calibration_samples_jsonl,
        build_real_calibration_samples_report=args.build_real_calibration_samples_report,
        data_engineering_round_dirs=args.data_engineering_round_dir,
        data_engineering_validation_report=args.data_engineering_validation_report,
        data_engineering_min_metadata_coverage=args.data_engineering_min_metadata_coverage,
        data_engineering_min_core_table_coverage=args.data_engineering_min_core_table_coverage,
        data_engineering_min_artifact_hash_coverage=args.data_engineering_min_artifact_hash_coverage,
        underdog_residual_round_dirs=args.underdog_residual_round_dir,
        underdog_residual_validation_report=args.underdog_residual_validation_report,
        underdog_min_attack_residual_coverage=args.underdog_min_attack_residual_coverage,
        underdog_min_defense_residual_coverage=args.underdog_min_defense_residual_coverage,
        underdog_min_mean_attack_residual_bonus=args.underdog_min_mean_attack_residual_bonus,
        underdog_min_mean_defense_residual_bonus=args.underdog_min_mean_defense_residual_bonus,
        league_health_round_dirs=args.league_health_round_dir,
        league_selfplay_health_report=args.league_selfplay_health_report,
        league_health_min_attack_pool=args.league_health_min_attack_pool,
        league_health_min_defense_pool=args.league_health_min_defense_pool,
        league_health_min_total_clusters=args.league_health_min_total_clusters,
        league_health_min_payoff_density=args.league_health_min_payoff_density,
        league_health_required_attack_roles=args.league_health_required_attack_role or ("main", "exploiter", "underdog"),
        league_health_required_defense_roles=args.league_health_required_defense_role or ("main", "exploiter", "underdog"),
        league_health_min_active_pool_fraction=args.league_health_min_active_pool_fraction,
        league_health_min_new_attack_strength_delta=args.league_health_min_new_attack_strength_delta,
        league_health_min_new_defense_strength_delta=args.league_health_min_new_defense_strength_delta,
        production_readiness_report_paths=args.production_readiness_report_json,
        production_readiness_include_scheduled_reports=args.production_readiness_include_scheduled_reports,
        production_readiness_report=args.production_readiness_report,
        production_readiness_required_schema_versions=args.production_readiness_required_schema_version,
        production_readiness_min_clean_report_rate=args.production_readiness_min_clean_report_rate,
        production_readiness_require_production_ready=not args.production_readiness_no_require_production_ready,
        v4_conformance_report_paths=args.v4_conformance_report_json,
        v4_conformance_include_scheduled_reports=args.v4_conformance_include_scheduled_reports,
        v4_conformance_validation_report=args.v4_conformance_validation_report,
        real_round_dirs=args.real_round_dir,
        real_meta_db_jsonl=args.real_meta_db_jsonl,
        real_calibration_report=args.real_calibration_report,
        real_rank_segment=args.real_rank_segment,
        real_server=args.real_server,
        real_season=args.real_season,
        real_timestamp=args.real_timestamp,
        drift_baseline_season=args.drift_baseline_season,
        drift_current_season=args.drift_current_season,
        drift_delta_threshold=args.drift_delta_threshold,
        drift_min_overlap=args.drift_min_overlap,
        real_calibration_validation_samples_jsonl=args.real_calibration_validation_samples_jsonl,
        real_calibration_validation_model_json=args.real_calibration_validation_model_json,
        real_calibration_validation_report=args.real_calibration_validation_report,
        real_calibration_validation_min_samples=args.real_calibration_validation_min_samples,
        real_calibration_min_brier_improvement=args.real_calibration_min_brier_improvement,
        real_calibration_min_ece_improvement=args.real_calibration_min_ece_improvement,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_dim=args.model_dim,
        proposal_model_dim=args.proposal_model_dim,
        heads=args.heads,
        layers=args.layers,
        seed=args.seed,
        camp_group=args.camp_group,
    )
    write_training_schedule(args.out_schedule, schedule)
    summary = run_training_schedule(
        schedule,
        execute=args.execute,
        cwd=ROOT,
        status_path=args.out_status,
        monitor_resources=args.monitor_resources,
    )
    print(
        json.dumps(
            {
                "schedule": str(args.out_schedule),
                "status": str(args.out_status),
                "jobs": len(summary.jobs),
                "executed": summary.executed,
                "failed": [job.job_id for job in summary.jobs if job.status == "failed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(job.status != "failed" for job in summary.jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
