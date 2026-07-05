from __future__ import annotations

import json
import subprocess
import sys

from masked_team_league.training.schedule import (
    RecurringTrainingSchedulerState,
    ScheduledTrainingJob,
    TrainingJobStatus,
    TrainingResourceSnapshot,
    TrainingRunSummary,
    TrainingSchedule,
    build_scheduler_red_line_check,
    build_v4_training_schedule,
    collect_training_resource_snapshot,
    load_training_schedule,
    run_recurring_training_scheduler,
    run_training_schedule,
    write_training_schedule,
)
from masked_team_league.data_engineering.run_metadata import load_run_metadata_manifest


def test_build_v4_training_schedule_wires_all_model_jobs(tmp_path):
    root = tmp_path / "training"
    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        decoded_dir=tmp_path / "decoded",
        single_team_train_jsonl=tmp_path / "single_train.jsonl",
        single_team_holdout_jsonl=tmp_path / "single_holdout.jsonl",
        belief_train_jsonl=tmp_path / "belief_train.jsonl",
        belief_holdout_jsonl=tmp_path / "belief_holdout.jsonl",
        attack_teacher_jsonl=tmp_path / "attack_teacher.jsonl",
        defense_teacher_jsonl=tmp_path / "defense_teacher.jsonl",
        mask_teacher_jsonl=tmp_path / "mask_teacher.jsonl",
        epochs=2,
        device="cuda:0",
        proposal_model_dim=64,
    )

    job_ids = [job.job_id for job in schedule.jobs]
    assert job_ids == [
        "single-team-split-manifest",
        "single-team-train",
        "single-team-select-best",
        "belief-ranker-split-manifest",
        "belief-ranker-train",
        "belief-ranker-select-best",
        "attack-proposal-train",
        "attack-proposal-select-best",
        "defense-proposal-train",
        "defense-proposal-select-best",
        "mask-selection-train",
        "mask-selection-select-best",
        "exploiter-effectiveness-report",
        "defense-anti-meta-effectiveness-report",
    ]
    assert schedule.registry_path == str(root / "checkpoint_registry.json")
    assert schedule.jobs[1].depends_on == ("single-team-split-manifest",)
    assert "masked_team_league.cli.commands.train_single_team_model" in schedule.jobs[1].command
    assert "--device" in schedule.jobs[1].command
    job_by_id = {job.job_id: job for job in schedule.jobs}
    assert "masked_team_league.cli.commands.train_defense_proposal" in job_by_id["defense-proposal-train"].command
    assert "masked_team_league.cli.commands.train_mask_selection" in job_by_id["mask-selection-train"].command
    assert job_by_id["mask-selection-select-best"].depends_on == ("mask-selection-train",)
    assert job_by_id["exploiter-effectiveness-report"].depends_on == ("attack-proposal-train",)
    assert job_by_id["defense-anti-meta-effectiveness-report"].depends_on == ("defense-proposal-train",)


def test_build_v4_training_schedule_wires_real_calibration_ingest_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    db_jsonl = tmp_path / "real_meta.jsonl"
    report_json = tmp_path / "real_calibration_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        single_team_train_jsonl=tmp_path / "single_train.jsonl",
        real_round_dirs=(round_a, round_b),
        real_meta_db_jsonl=db_jsonl,
        real_calibration_report=report_json,
        real_rank_segment="top",
        real_server="oracle_backend",
        real_season="S29",
        real_timestamp=123.0,
        drift_baseline_season="S28",
        drift_current_season="S29",
    )

    job = schedule.jobs[-1]

    assert job.job_id == "real-calibration-ingest"
    assert job.stage == "real_calibration"
    assert job.depends_on == ("single-team-select-best",)
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(db_jsonl), str(report_json))
    assert "masked_team_league.cli.commands.ingest_real_calibration" in job.command
    assert job.command.count("--round-dir") == 2
    assert "--db-jsonl" in job.command
    assert str(db_jsonl) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--season" in job.command
    assert "S29" in job.command
    assert "--drift-baseline-season" in job.command
    assert "--drift-current-season" in job.command


def test_build_v4_training_schedule_wires_active_real_feedback_into_real_calibration(tmp_path):
    root = tmp_path / "training"
    feedback_a = tmp_path / "real_query_feedback_a"
    feedback_b = tmp_path / "real_query_feedback_b"
    db_jsonl = tmp_path / "real_meta.jsonl"
    report_json = tmp_path / "real_calibration_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        active_real_feedback_dirs=(feedback_a, feedback_b),
        real_meta_db_jsonl=db_jsonl,
        real_calibration_report=report_json,
        real_rank_segment="top",
        real_server="oracle_backend",
        real_season="S29",
    )

    job = schedule.jobs[-1]

    assert job.job_id == "real-calibration-ingest"
    assert job.stage == "real_calibration"
    assert job.inputs == (str(feedback_a), str(feedback_b))
    assert job.outputs == (str(db_jsonl), str(report_json))
    assert "masked_team_league.cli.commands.ingest_real_calibration" in job.command
    assert job.command.count("--active-real-feedback-dir") == 2
    assert str(feedback_a) in job.command
    assert str(feedback_b) in job.command
    assert "--db-jsonl" in job.command
    assert str(db_jsonl) in job.command
    assert "--season" in job.command
    assert "S29" in job.command


def test_build_v4_training_schedule_builds_real_calibration_samples_for_validation(tmp_path):
    root = tmp_path / "training"
    round_dir = tmp_path / "round_0001"
    feedback_dir = tmp_path / "real_query_feedback"
    samples_jsonl = tmp_path / "real_calibration_samples.jsonl"
    sample_report = tmp_path / "real_calibration_samples_report.json"
    calibration_model = tmp_path / "real_feature_calibrator.json"
    validation_report = tmp_path / "real_calibration_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        real_round_dirs=(round_dir,),
        active_real_feedback_dirs=(feedback_dir,),
        real_meta_db_jsonl=tmp_path / "real_meta.jsonl",
        real_season="S29",
        build_real_calibration_samples_jsonl=samples_jsonl,
        build_real_calibration_samples_report=sample_report,
        real_calibration_validation_model_json=calibration_model,
        real_calibration_validation_report=validation_report,
        real_calibration_validation_min_samples=1,
    )
    job_by_id = {job.job_id: job for job in schedule.jobs}
    sample_job = job_by_id["real-calibration-sample-build"]
    validation_job = job_by_id["real-calibration-validation-report"]

    assert "masked_team_league.cli.commands.build_real_calibration_samples" in sample_job.command
    assert sample_job.command.count("--round-dir") == 1
    assert sample_job.command.count("--active-real-feedback-dir") == 1
    assert str(round_dir) in sample_job.command
    assert str(feedback_dir) in sample_job.command
    assert "--out-jsonl" in sample_job.command
    assert str(samples_jsonl) in sample_job.command
    assert "--out-report" in sample_job.command
    assert str(sample_report) in sample_job.command
    assert sample_job.outputs == (str(samples_jsonl), str(sample_report))
    assert validation_job.inputs == (str(samples_jsonl), str(calibration_model))
    assert validation_job.depends_on == ("real-calibration-sample-build",)


def test_build_v4_training_schedule_wires_real_calibration_validation_report_job(tmp_path):
    root = tmp_path / "training"
    samples_jsonl = tmp_path / "real_calibration_holdout.jsonl"
    calibration_json = tmp_path / "real_feature_calibrator.json"
    report_json = tmp_path / "real_calibration_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        real_calibration_validation_samples_jsonl=(samples_jsonl,),
        real_calibration_validation_model_json=calibration_json,
        real_calibration_validation_report=report_json,
        real_calibration_validation_min_samples=50,
        real_calibration_min_brier_improvement=0.02,
        real_calibration_min_ece_improvement=0.01,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "real-calibration-validation-report"
    assert job.stage == "real_calibration_validation_report"
    assert job.inputs == (str(samples_jsonl), str(calibration_json))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_real_calibration_validation" in job.command
    assert "--samples-jsonl" in job.command
    assert str(samples_jsonl) in job.command
    assert "--calibration-json" in job.command
    assert str(calibration_json) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-samples" in job.command
    assert "50" in job.command
    assert "--min-brier-improvement" in job.command
    assert "0.02" in job.command
    assert "--min-ece-improvement" in job.command
    assert "0.01" in job.command


def test_build_v4_training_schedule_builds_belief_ranker_dataset_from_round_artifacts(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        belief_round_dirs=(round_a, round_b),
        belief_round_holdout_fraction=0.25,
        epochs=2,
        seed=123,
    )
    job_by_id = {job.job_id: job for job in schedule.jobs}

    build_job = job_by_id["belief-ranker-build-dataset"]
    train_job = job_by_id["belief-ranker-train"]

    assert build_job.stage == "build_belief_ranker_dataset"
    assert build_job.inputs == (str(round_a), str(round_b))
    assert build_job.outputs == (
        str(root / "datasets" / "belief_ranker" / "belief_ranker_train.jsonl"),
        str(root / "datasets" / "belief_ranker" / "belief_ranker_holdout.jsonl"),
        str(root / "datasets" / "belief_ranker" / "split_manifest.json"),
    )
    assert "masked_team_league.cli.commands.build_belief_ranker_dataset" in build_job.command
    assert build_job.command.count("--round-dir") == 2
    assert str(round_a) in build_job.command
    assert str(round_b) in build_job.command
    assert "--holdout-fraction" in build_job.command
    assert "0.25" in build_job.command
    assert "--dataset-id" in build_job.command
    assert "unit:belief_ranker_rounds" in build_job.command
    assert train_job.depends_on == ("belief-ranker-build-dataset",)
    assert train_job.inputs == build_job.outputs
    assert str(root / "datasets" / "belief_ranker" / "belief_ranker_train.jsonl") in train_job.command
    assert str(root / "datasets" / "belief_ranker" / "belief_ranker_holdout.jsonl") in train_job.command


def test_belief_ranker_round_dataset_schedule_uses_train_metric_when_holdout_fraction_is_zero(tmp_path):
    root = tmp_path / "training"
    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        belief_round_dirs=(tmp_path / "round_0001",),
        belief_round_holdout_fraction=0.0,
    )
    job_by_id = {job.job_id: job for job in schedule.jobs}
    train_job = job_by_id["belief-ranker-train"]
    select_job = job_by_id["belief-ranker-select-best"]

    assert "--holdout-jsonl" not in train_job.command
    assert "--metric" in select_job.command
    assert "train_top1_accuracy" in select_job.command
    assert "holdout_top1_accuracy" not in select_job.command


def test_build_v4_training_schedule_wires_exploiter_effectiveness_report_job(tmp_path):
    root = tmp_path / "training"
    attack_teacher = tmp_path / "attack_teacher.jsonl"
    report_json = tmp_path / "exploiter_effectiveness_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        attack_teacher_jsonl=attack_teacher,
        exploiter_effectiveness_report=report_json,
        exploiter_min_target_coverage=0.9,
        exploiter_min_positive_residual_rate=0.6,
        exploiter_min_trend_delta=0.0,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "exploiter-effectiveness-report"
    assert job.stage == "exploiter_effectiveness_report"
    assert job.depends_on == ("attack-proposal-train",)
    assert job.inputs == (str(attack_teacher),)
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_exploiter_effectiveness" in job.command
    assert "--teacher-jsonl" in job.command
    assert str(attack_teacher) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-target-coverage" in job.command
    assert "0.9" in job.command
    assert "--min-positive-residual-rate" in job.command
    assert "0.6" in job.command
    assert "--min-trend-delta" in job.command
    assert "0.0" in job.command


def test_build_v4_training_schedule_wires_mask_explanation_validation_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "mask_explanation_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        mask_explanation_round_dirs=(round_a, round_b),
        mask_explanation_validation_report=report_json,
        mask_min_hidden_explanation_coverage=0.8,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "mask-explanation-validation-report"
    assert job.stage == "mask_explanation_validation_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_mask_explanation_validation" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-hidden-explanation-coverage" in job.command
    assert "0.8" in job.command


def test_build_v4_training_schedule_wires_belief_real_distribution_validation_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "belief_real_distribution_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        belief_real_validation_round_dirs=(round_a, round_b),
        belief_real_distribution_report=report_json,
        belief_min_real_coverage=0.7,
        belief_min_mean_real_records=2.0,
        belief_min_mean_real_similarity=0.4,
        belief_max_oracle_alignment_mae=0.25,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "belief-real-distribution-validation-report"
    assert job.stage == "belief_real_distribution_validation_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_belief_real_distribution_validation" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-real-coverage" in job.command
    assert "0.7" in job.command
    assert "--min-mean-real-records" in job.command
    assert "2.0" in job.command
    assert "--min-mean-real-similarity" in job.command
    assert "0.4" in job.command
    assert "--max-oracle-alignment-mae" in job.command
    assert "0.25" in job.command


def test_build_v4_training_schedule_wires_defense_anti_meta_effectiveness_report_job(tmp_path):
    root = tmp_path / "training"
    defense_teacher = tmp_path / "defense_teacher.jsonl"
    training_root = tmp_path / "selfplay_training"
    report_json = tmp_path / "defense_anti_meta_effectiveness_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        defense_teacher_jsonl=defense_teacher,
        defense_anti_meta_training_root=training_root,
        defense_anti_meta_effectiveness_report=report_json,
        defense_anti_meta_min_feedback_coverage=0.8,
        defense_anti_meta_min_positive_residual_rate=0.6,
        defense_anti_meta_min_mean_residual=0.05,
        defense_anti_meta_min_trend_delta=0.0,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "defense-anti-meta-effectiveness-report"
    assert job.stage == "defense_anti_meta_effectiveness_report"
    assert job.depends_on == ("defense-proposal-train",)
    assert job.inputs == (str(defense_teacher), str(training_root))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_defense_anti_meta_effectiveness" in job.command
    assert "--teacher-jsonl" in job.command
    assert str(defense_teacher) in job.command
    assert "--training-root" in job.command
    assert str(training_root) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-feedback-coverage" in job.command
    assert "0.8" in job.command
    assert "--min-positive-residual-rate" in job.command
    assert "0.6" in job.command
    assert "--min-mean-residual" in job.command
    assert "0.05" in job.command
    assert "--min-trend-delta" in job.command
    assert "0.0" in job.command


def test_build_v4_training_schedule_wires_learned_exploiter_validation_report_job(tmp_path):
    root = tmp_path / "training"
    selfplay_root = tmp_path / "selfplay"
    training_root = tmp_path / "selfplay_training"
    report_json = tmp_path / "learned_exploiter_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        learned_exploiter_selfplay_root=selfplay_root,
        learned_exploiter_training_root=training_root,
        learned_exploiter_validation_report=report_json,
        learned_exploiter_min_rounds=3,
        learned_exploiter_min_oracle_requests=100,
        learned_exploiter_min_attack_trend_delta=0.01,
        learned_exploiter_min_defense_trend_delta=0.02,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "learned-exploiter-validation-report"
    assert job.stage == "learned_exploiter_validation_report"
    assert job.inputs == (str(selfplay_root), str(training_root))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_learned_exploiter_validation" in job.command
    assert "--selfplay-root" in job.command
    assert str(selfplay_root) in job.command
    assert "--training-root" in job.command
    assert str(training_root) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-rounds" in job.command
    assert "3" in job.command
    assert "--min-oracle-requests" in job.command
    assert "100" in job.command
    assert "--min-attack-trend-delta" in job.command
    assert "0.01" in job.command
    assert "--min-defense-trend-delta" in job.command
    assert "0.02" in job.command


def test_build_v4_training_schedule_wires_attack_oracle_failure_validation_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    oracle_output = tmp_path / "attack_oracle_output.json"
    report_json = tmp_path / "attack_oracle_failure_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        attack_oracle_failure_round_dirs=(round_a,),
        attack_oracle_failure_output_jsons=(oracle_output,),
        attack_oracle_failure_validation_report=report_json,
        attack_oracle_min_failure_annotation_coverage=0.9,
        attack_oracle_min_failure_diagnostic_coverage=0.8,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "attack-oracle-failure-validation-report"
    assert job.stage == "attack_oracle_failure_validation_report"
    assert job.inputs == (str(oracle_output), str(round_a))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_attack_oracle_failure_validation" in job.command
    assert "--oracle-output-json" in job.command
    assert str(oracle_output) in job.command
    assert "--round-dir" in job.command
    assert str(round_a) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-failure-annotation-coverage" in job.command
    assert "0.9" in job.command
    assert "--min-failure-diagnostic-coverage" in job.command
    assert "0.8" in job.command


def test_build_v4_training_schedule_wires_active_query_feedback_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "active_query_feedback_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        active_query_feedback_round_dirs=(round_a, round_b),
        active_query_feedback_report=report_json,
        active_query_min_matched_coverage=0.8,
        active_query_max_oracle_error_rate=0.1,
        active_query_min_real_query_count=1,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "active-query-feedback-report"
    assert job.stage == "active_query_feedback_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_active_query_feedback" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-matched-query-coverage" in job.command
    assert "0.8" in job.command
    assert "--max-oracle-result-error-rate" in job.command
    assert "0.1" in job.command
    assert "--min-real-query-count" in job.command
    assert "1" in job.command


def test_build_v4_training_schedule_wires_active_real_dispatch_validation_report_job(tmp_path):
    root = tmp_path / "training"
    validation_a = tmp_path / "real_queries_a" / "validation_report.json"
    validation_b = tmp_path / "real_queries_b" / "validation_report.json"
    report_json = tmp_path / "active_real_query_dispatch_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        active_real_dispatch_validation_jsons=(validation_a, validation_b),
        active_real_dispatch_validation_report=report_json,
        active_real_dispatch_min_reports=2,
        active_real_dispatch_min_dispatched_pairs=3,
        active_real_dispatch_min_completion_rate=0.95,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "active-real-dispatch-validation-report"
    assert job.stage == "active_real_query_dispatch_validation_report"
    assert job.inputs == (str(validation_a), str(validation_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_active_real_query_dispatch_validation" in job.command
    assert job.command.count("--validation-json") == 2
    assert str(validation_a) in job.command
    assert str(validation_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-reports" in job.command
    assert "2" in job.command
    assert "--min-dispatched-pairs" in job.command
    assert "3" in job.command
    assert "--min-completion-rate" in job.command
    assert "0.95" in job.command


def test_build_v4_training_schedule_wires_data_engineering_validation_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "data_engineering_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        data_engineering_round_dirs=(round_a, round_b),
        data_engineering_validation_report=report_json,
        data_engineering_min_metadata_coverage=0.9,
        data_engineering_min_core_table_coverage=0.8,
        data_engineering_min_artifact_hash_coverage=0.7,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "data-engineering-validation-report"
    assert job.stage == "data_engineering_validation_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_data_engineering_validation" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-metadata-coverage" in job.command
    assert "0.9" in job.command
    assert "--min-core-table-coverage" in job.command
    assert "0.8" in job.command
    assert "--min-artifact-hash-coverage" in job.command
    assert "0.7" in job.command


def test_build_v4_training_schedule_wires_underdog_residual_validation_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "underdog_residual_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        underdog_residual_round_dirs=(round_a, round_b),
        underdog_residual_validation_report=report_json,
        underdog_min_attack_residual_coverage=0.9,
        underdog_min_defense_residual_coverage=0.8,
        underdog_min_mean_attack_residual_bonus=0.01,
        underdog_min_mean_defense_residual_bonus=0.02,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "underdog-residual-validation-report"
    assert job.stage == "underdog_residual_validation_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_underdog_residual_validation" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-attack-residual-coverage" in job.command
    assert "0.9" in job.command
    assert "--min-defense-residual-coverage" in job.command
    assert "0.8" in job.command
    assert "--min-mean-attack-residual-bonus" in job.command
    assert "0.01" in job.command
    assert "--min-mean-defense-residual-bonus" in job.command
    assert "0.02" in job.command


def test_build_v4_training_schedule_wires_production_readiness_report_job(tmp_path):
    root = tmp_path / "training"
    learned_report = tmp_path / "learned_exploiter_validation_report.json"
    active_report = tmp_path / "active_query_feedback_report.json"
    output_report = tmp_path / "production_readiness_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        production_readiness_report_paths=(learned_report, active_report),
        production_readiness_report=output_report,
        production_readiness_required_schema_versions=("learned_exploiter_validation_report.v1",),
        production_readiness_min_clean_report_rate=0.9,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "production-readiness-report"
    assert job.stage == "production_readiness_report"
    assert job.inputs == (str(learned_report), str(active_report))
    assert job.outputs == (str(output_report),)
    assert "masked_team_league.cli.commands.report_production_readiness" in job.command
    assert job.command.count("--report-json") == 2
    assert str(learned_report) in job.command
    assert str(active_report) in job.command
    assert "--out-report" in job.command
    assert str(output_report) in job.command
    assert "--required-schema-version" in job.command
    assert "learned_exploiter_validation_report.v1" in job.command
    assert "--min-clean-report-rate" in job.command
    assert "0.9" in job.command


def test_build_v4_training_schedule_wires_v4_conformance_validation_report_job(tmp_path):
    root = tmp_path / "training"
    learned_report = tmp_path / "learned_exploiter_validation_report.json"
    active_report = tmp_path / "active_real_query_dispatch_validation_report.json"
    output_report = tmp_path / "v4_conformance_validation_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        v4_conformance_report_paths=(learned_report, active_report),
        v4_conformance_validation_report=output_report,
    )

    job = schedule.jobs[-1]
    assert job.job_id == "v4-conformance-validation-report"
    assert job.stage == "v4_conformance_validation_report"
    assert "masked_team_league.cli.commands.report_v4_conformance_validation" in job.command
    assert str(learned_report) in job.command
    assert str(active_report) in job.command
    assert job.inputs == (str(learned_report), str(active_report))
    assert job.outputs == (str(output_report),)


def test_build_v4_training_schedule_auto_collects_scheduled_reports_for_v4_conformance(tmp_path):
    root = tmp_path / "training"
    round_dir = tmp_path / "round_0001"
    validation_json = tmp_path / "active_real_dispatch_validation.json"
    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        active_query_feedback_round_dirs=(round_dir,),
        active_real_dispatch_validation_jsons=(validation_json,),
        v4_conformance_include_scheduled_reports=True,
    )

    job_by_id = {job.job_id: job for job in schedule.jobs}
    conformance = job_by_id["v4-conformance-validation-report"]
    assert conformance.outputs == (str(root / "reports" / "v4_conformance_validation_report.json"),)
    assert str(root / "reports" / "active_query_feedback_report.json") in conformance.inputs
    assert str(root / "reports" / "active_real_query_dispatch_validation_report.json") in conformance.inputs
    assert "active-query-feedback-report" in conformance.depends_on
    assert "active-real-dispatch-validation-report" in conformance.depends_on


def test_build_v4_training_schedule_wires_league_selfplay_health_report_job(tmp_path):
    root = tmp_path / "training"
    round_a = tmp_path / "round_0001"
    round_b = tmp_path / "round_0002"
    report_json = tmp_path / "league_selfplay_health_report.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        league_health_round_dirs=(round_a, round_b),
        league_selfplay_health_report=report_json,
        league_health_min_attack_pool=8,
        league_health_min_defense_pool=6,
        league_health_min_total_clusters=5,
        league_health_min_payoff_density=0.25,
        league_health_required_attack_roles=("main", "exploiter"),
        league_health_required_defense_roles=("main", "underdog"),
        league_health_min_active_pool_fraction=0.75,
        league_health_min_new_attack_strength_delta=0.01,
        league_health_min_new_defense_strength_delta=0.02,
    )

    job = schedule.jobs[-1]

    assert job.job_id == "league-selfplay-health-report"
    assert job.stage == "league_selfplay_health_report"
    assert job.inputs == (str(round_a), str(round_b))
    assert job.outputs == (str(report_json),)
    assert "masked_team_league.cli.commands.report_league_selfplay_health" in job.command
    assert job.command.count("--round-dir") == 2
    assert str(round_a) in job.command
    assert str(round_b) in job.command
    assert "--out-report" in job.command
    assert str(report_json) in job.command
    assert "--min-attack-pool" in job.command
    assert "8" in job.command
    assert "--min-defense-pool" in job.command
    assert "6" in job.command
    assert "--min-total-clusters" in job.command
    assert "5" in job.command
    assert "--min-payoff-density" in job.command
    assert "0.25" in job.command
    assert job.command.count("--required-attack-role") == 2
    assert job.command.count("--required-defense-role") == 2
    assert "--min-active-pool-fraction" in job.command
    assert "0.75" in job.command
    assert "--min-new-attack-strength-delta" in job.command
    assert "0.01" in job.command
    assert "--min-new-defense-strength-delta" in job.command
    assert "0.02" in job.command


def test_build_v4_training_schedule_auto_collects_scheduled_reports_for_production_readiness(tmp_path):
    root = tmp_path / "training"
    active_round = tmp_path / "active_round"
    data_round = tmp_path / "data_round"
    calibration_samples = tmp_path / "real_calibration_holdout.jsonl"
    calibration_model = tmp_path / "real_feature_calibrator.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        active_query_feedback_round_dirs=(active_round,),
        data_engineering_round_dirs=(data_round,),
        real_calibration_validation_samples_jsonl=(calibration_samples,),
        real_calibration_validation_model_json=calibration_model,
        production_readiness_include_scheduled_reports=True,
        production_readiness_required_schema_versions=(
            "active_query_feedback_report.v1",
            "data_engineering_validation_report.v1",
            "real_calibration_validation_report.v1",
        ),
    )
    job_by_id = {job.job_id: job for job in schedule.jobs}
    readiness = schedule.jobs[-1]
    expected_inputs = (
        str(root / "reports" / "active_query_feedback_report.json"),
        str(root / "reports" / "data_engineering_validation_report.json"),
        str(root / "reports" / "real_calibration_validation_report.json"),
    )

    assert readiness.job_id == "production-readiness-report"
    assert readiness.inputs == expected_inputs
    assert readiness.outputs == (str(root / "reports" / "production_readiness_report.json"),)
    assert readiness.depends_on == (
        job_by_id["active-query-feedback-report"].job_id,
        job_by_id["data-engineering-validation-report"].job_id,
        job_by_id["real-calibration-validation-report"].job_id,
    )
    assert readiness.command.count("--report-json") == 3
    for path in expected_inputs:
        assert path in readiness.command
    assert readiness.command.count("--required-schema-version") == 3


def test_build_v4_training_schedule_auto_collects_real_calibration_sample_build_report(tmp_path):
    root = tmp_path / "training"
    round_dir = tmp_path / "round_0001"
    sample_jsonl = tmp_path / "real_calibration_samples.jsonl"
    sample_report = tmp_path / "real_calibration_samples_report.json"
    calibration_model = tmp_path / "real_feature_calibrator.json"

    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=root,
        heroes_json=tmp_path / "heroes.json",
        real_round_dirs=(round_dir,),
        real_meta_db_jsonl=tmp_path / "real_meta.jsonl",
        real_season="S29",
        build_real_calibration_samples_jsonl=sample_jsonl,
        build_real_calibration_samples_report=sample_report,
        real_calibration_validation_model_json=calibration_model,
        production_readiness_include_scheduled_reports=True,
        production_readiness_required_schema_versions=(
            "real_calibration_sample_build_summary.v1",
            "real_calibration_validation_report.v1",
        ),
    )
    job_by_id = {job.job_id: job for job in schedule.jobs}
    readiness = schedule.jobs[-1]

    assert readiness.job_id == "production-readiness-report"
    assert readiness.inputs == (
        str(root / "reports" / "real_calibration_report.json"),
        str(sample_report),
        str(root / "reports" / "real_calibration_validation_report.json"),
    )
    assert readiness.depends_on == (
        job_by_id["real-calibration-ingest"].job_id,
        job_by_id["real-calibration-sample-build"].job_id,
        job_by_id["real-calibration-validation-report"].job_id,
    )
    assert readiness.command.count("--report-json") == 3
    assert str(sample_report) in readiness.command
    assert "real_calibration_sample_build_summary.v1" in readiness.command


def test_training_schedule_round_trips_and_dry_run_writes_status(tmp_path):
    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=tmp_path / "training",
        heroes_json=tmp_path / "heroes.json",
        single_team_train_jsonl=tmp_path / "single_train.jsonl",
    )
    schedule_path = tmp_path / "schedule.json"
    status_path = tmp_path / "status.json"
    write_training_schedule(schedule_path, schedule)

    loaded = load_training_schedule(schedule_path)
    summary = run_training_schedule(loaded, execute=False, status_path=status_path)
    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert loaded == schedule
    assert summary.executed is False
    assert all(row.status == "dry_run" for row in summary.jobs)
    assert payload["executed"] is False
    assert payload["jobs"][0]["status"] == "dry_run"
    manifest = load_run_metadata_manifest(tmp_path / "training" / "run_metadata.json")
    manifest_payload = manifest.to_json_dict()
    artifact_paths = {row["path"] for row in manifest_payload["input_artifacts"] + manifest_payload["output_artifacts"]}
    assert manifest_payload["run_id"] == "unit"
    assert manifest_payload["data_version"] == "v4"
    assert manifest_payload["random_seed"] == 0
    assert manifest_payload["generation_config_hash"]
    assert str(status_path) in artifact_paths


def test_collect_training_resource_snapshot_parses_memory_load_and_gpu_csv(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32768000 kB\n"
        "MemAvailable:  16384000 kB\n",
        encoding="utf-8",
    )

    snapshot = collect_training_resource_snapshot(
        now=123.0,
        cpu_count=16,
        loadavg=(1.0, 2.0, 3.0),
        meminfo_path=meminfo,
        nvidia_smi_output="0, 4096, 24576\n1, 2048, 24576\n",
    )
    payload = snapshot.to_json_dict()

    assert payload["timestamp"] == 123.0
    assert payload["cpu_count"] == 16
    assert payload["loadavg_1m"] == 1.0
    assert payload["memory_total_mb"] == 32000.0
    assert payload["memory_available_mb"] == 16000.0
    assert payload["gpu_count"] == 2
    assert payload["gpu_memory_used_mb"] == 6144.0
    assert payload["gpu_memory_total_mb"] == 49152.0


def test_run_training_schedule_dry_run_can_write_resource_snapshots(tmp_path):
    schedule = build_v4_training_schedule(
        schedule_id="unit",
        root_dir=tmp_path / "training",
        heroes_json=tmp_path / "heroes.json",
        single_team_train_jsonl=tmp_path / "single_train.jsonl",
    )
    status_path = tmp_path / "status.json"
    calls = {"count": 0}

    def fake_snapshot():
        calls["count"] += 1
        return TrainingResourceSnapshot(
            timestamp=float(calls["count"]),
            cpu_count=8,
            loadavg_1m=0.1,
            loadavg_5m=0.2,
            loadavg_15m=0.3,
            memory_total_mb=1000.0,
            memory_available_mb=900.0,
            gpu_count=1,
            gpu_memory_used_mb=100.0,
            gpu_memory_total_mb=1000.0,
        )

    summary = run_training_schedule(
        schedule,
        execute=False,
        status_path=status_path,
        monitor_resources=True,
        resource_snapshot_fn=fake_snapshot,
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert summary.jobs[0].resource_before is not None
    assert summary.jobs[0].resource_after is not None
    assert payload["jobs"][0]["resource_before"]["cpu_count"] == 8
    assert payload["jobs"][0]["resource_after"]["timestamp"] == 2.0
    assert calls["count"] == len(summary.jobs) * 2


def test_recurring_training_scheduler_persists_iterations_and_sleeps_between_runs(tmp_path):
    sleeps: list[float] = []
    now = {"value": 100.0}

    def now_fn():
        now["value"] += 1.0
        return now["value"]

    def schedule_factory(iteration: int, run_dir):
        return TrainingSchedule(
            schedule_id=f"scheduler-r{iteration:04d}",
            root_dir=str(run_dir),
            registry_path=str(run_dir / "registry.json"),
            created_at=now_fn(),
            jobs=(
                ScheduledTrainingJob(
                    job_id="noop",
                    stage="noop",
                    command=(sys.executable, "-c", "print('noop')"),
                    inputs=(),
                    outputs=(),
                ),
            ),
        )

    state = run_recurring_training_scheduler(
        scheduler_id="unit-scheduler",
        root_dir=tmp_path / "scheduler",
        schedule_factory=schedule_factory,
        iterations=2,
        interval_seconds=5.0,
        execute=False,
        sleep_fn=sleeps.append,
        now_fn=now_fn,
    )
    state_path = tmp_path / "scheduler" / "scheduler_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert isinstance(state, RecurringTrainingSchedulerState)
    assert state.stopped is False
    assert len(state.iterations) == 2
    assert state.iterations[0].status == "completed"
    assert state.iterations[0].schedule_path.endswith("iteration_0001/schedule.json")
    assert payload["iterations"][1]["status_path"].endswith("iteration_0002/status.json")
    assert sleeps == [5.0]


def test_recurring_training_scheduler_stops_on_red_line_violations(tmp_path):
    def schedule_factory(iteration: int, run_dir):
        return TrainingSchedule(
            schedule_id=f"scheduler-r{iteration:04d}",
            root_dir=str(run_dir),
            registry_path=str(run_dir / "registry.json"),
            created_at=1.0,
            jobs=(),
        )

    state = run_recurring_training_scheduler(
        scheduler_id="unit-scheduler",
        root_dir=tmp_path / "scheduler",
        schedule_factory=schedule_factory,
        iterations=3,
        interval_seconds=0.0,
        execute=False,
        red_line_check_fn=lambda _run_dir, _summary: ("single_model_ece_high",),
    )

    assert state.stopped is True
    assert state.stop_reason == "red_line_violations"
    assert state.iterations[0].red_line_violations == ("single_model_ece_high",)
    assert len(state.iterations) == 1


def test_scheduler_red_line_check_reads_iteration_real_calibration_drift(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "real_calibration_report.json").write_text(
        json.dumps(
            {
                "drift": {
                    "drift_detected": True,
                    "baseline_season": "S28",
                    "current_season": "S29",
                }
            }
        ),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("real_calibration_drift_detected",)


def test_scheduler_red_line_check_reads_iteration_exploiter_effectiveness_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "exploiter_effectiveness_report.json").write_text(
        json.dumps({"red_line_violations": ["anti_meta_residual_non_positive"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("anti_meta_residual_non_positive",)


def test_scheduler_red_line_check_reads_iteration_mask_explanation_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "mask_explanation_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["hidden_slot_explanation_coverage_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("hidden_slot_explanation_coverage_low",)


def test_scheduler_red_line_check_reads_iteration_belief_real_distribution_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "belief_real_distribution_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["real_distribution_coverage_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("real_distribution_coverage_low",)


def test_scheduler_red_line_check_reads_iteration_defense_anti_meta_effectiveness_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "defense_anti_meta_effectiveness_report.json").write_text(
        json.dumps({"red_line_violations": ["defense_anti_meta_residual_non_positive"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("defense_anti_meta_residual_non_positive",)


def test_scheduler_red_line_check_reads_iteration_attack_oracle_failure_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "attack_oracle_failure_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["failure_annotation_coverage_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("failure_annotation_coverage_low",)


def test_scheduler_red_line_check_reads_iteration_active_query_feedback_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "active_query_feedback_report.json").write_text(
        json.dumps({"red_line_violations": ["real_query_feedback_missing"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("real_query_feedback_missing",)


def test_scheduler_red_line_check_reads_iteration_data_engineering_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "data_engineering_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["core_table_missing"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("core_table_missing",)


def test_scheduler_red_line_check_reads_iteration_underdog_residual_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "underdog_residual_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["attack_residual_coverage_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("attack_residual_coverage_low",)


def test_scheduler_red_line_check_reads_iteration_learned_exploiter_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "learned_exploiter_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["validation_rounds_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("validation_rounds_low",)


def test_scheduler_red_line_check_reads_iteration_real_calibration_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "real_calibration_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["real_calibration_brier_not_improved"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("real_calibration_brier_not_improved",)


def test_scheduler_red_line_check_reads_iteration_production_readiness_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "production_readiness_report.json").write_text(
        json.dumps({"red_line_violations": ["required_schema_missing"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("required_schema_missing",)


def test_scheduler_red_line_check_reads_iteration_league_selfplay_health_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "league_selfplay_health_report.json").write_text(
        json.dumps({"red_line_violations": ["payoff_density_low"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("payoff_density_low",)


def test_scheduler_red_line_check_reads_iteration_active_real_dispatch_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "active_real_query_dispatch_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["active_real_teacher_feedback_incomplete"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("active_real_teacher_feedback_incomplete",)


def test_scheduler_red_line_check_reads_iteration_v4_conformance_validation_report(tmp_path):
    run_dir = tmp_path / "scheduler" / "iteration_0001"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "v4_conformance_validation_report.json").write_text(
        json.dumps({"red_line_violations": ["full_ablation_feedback_evidence_missing"]}),
        encoding="utf-8",
    )
    check = build_scheduler_red_line_check()

    violations = check(
        run_dir,
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("full_ablation_feedback_evidence_missing",)


def test_scheduler_red_line_check_reads_external_report_lists(tmp_path):
    report = tmp_path / "daily_report.json"
    report.write_text(json.dumps({"red_line_violations": ["oracle_result_errors"]}), encoding="utf-8")
    check = build_scheduler_red_line_check(extra_report_paths=(report,))

    violations = check(
        tmp_path / "iteration_0001",
        TrainingRunSummary(schedule_id="unit", executed=False, jobs=()),
    )

    assert violations == ("oracle_result_errors",)


def test_training_schedule_clis_expose_exploiter_and_mask_validation_args():
    schedule_help = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_training_schedule", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    daemon_help = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_training_scheduler_daemon", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert schedule_help.returncode == 0
    assert daemon_help.returncode == 0
    assert "--exploiter-training-root" in schedule_help.stdout
    assert "--exploiter-effectiveness-report" in schedule_help.stdout
    assert "--exploiter-min-trend-delta" in schedule_help.stdout
    assert "--exploiter-training-root" in daemon_help.stdout
    assert "--exploiter-effectiveness-report" in daemon_help.stdout
    assert "--exploiter-min-trend-delta" in daemon_help.stdout
    assert "--mask-explanation-round-dir" in schedule_help.stdout
    assert "--mask-explanation-validation-report" in schedule_help.stdout
    assert "--mask-min-hidden-explanation-coverage" in schedule_help.stdout
    assert "--mask-explanation-round-dir" in daemon_help.stdout
    assert "--mask-explanation-validation-report" in daemon_help.stdout
    assert "--mask-min-hidden-explanation-coverage" in daemon_help.stdout
    assert "--belief-real-validation-round-dir" in schedule_help.stdout
    assert "--belief-real-distribution-report" in schedule_help.stdout
    assert "--belief-min-real-coverage" in schedule_help.stdout
    assert "--belief-real-validation-round-dir" in daemon_help.stdout
    assert "--belief-real-distribution-report" in daemon_help.stdout
    assert "--belief-min-real-coverage" in daemon_help.stdout
    assert "--defense-anti-meta-training-root" in schedule_help.stdout
    assert "--defense-anti-meta-effectiveness-report" in schedule_help.stdout
    assert "--defense-anti-meta-min-feedback-coverage" in schedule_help.stdout
    assert "--defense-anti-meta-training-root" in daemon_help.stdout
    assert "--defense-anti-meta-effectiveness-report" in daemon_help.stdout
    assert "--defense-anti-meta-min-feedback-coverage" in daemon_help.stdout
    assert "--attack-oracle-failure-round-dir" in schedule_help.stdout
    assert "--attack-oracle-failure-output-json" in schedule_help.stdout
    assert "--attack-oracle-failure-validation-report" in schedule_help.stdout
    assert "--attack-oracle-failure-round-dir" in daemon_help.stdout
    assert "--attack-oracle-failure-output-json" in daemon_help.stdout
    assert "--attack-oracle-failure-validation-report" in daemon_help.stdout
    assert "--active-query-feedback-round-dir" in schedule_help.stdout
    assert "--active-query-feedback-report" in schedule_help.stdout
    assert "--active-query-min-matched-coverage" in schedule_help.stdout
    assert "--active-query-feedback-round-dir" in daemon_help.stdout
    assert "--active-query-feedback-report" in daemon_help.stdout
    assert "--active-query-min-matched-coverage" in daemon_help.stdout
    assert "--active-real-dispatch-validation-json" in schedule_help.stdout
    assert "--active-real-dispatch-validation-report" in schedule_help.stdout
    assert "--active-real-dispatch-min-dispatched-pairs" in schedule_help.stdout
    assert "--active-real-dispatch-validation-json" in daemon_help.stdout
    assert "--active-real-dispatch-validation-report" in daemon_help.stdout
    assert "--active-real-dispatch-min-dispatched-pairs" in daemon_help.stdout
    assert "--learned-exploiter-selfplay-root" in schedule_help.stdout
    assert "--learned-exploiter-validation-report" in schedule_help.stdout
    assert "--learned-exploiter-min-rounds" in schedule_help.stdout
    assert "--learned-exploiter-selfplay-root" in daemon_help.stdout
    assert "--learned-exploiter-validation-report" in daemon_help.stdout
    assert "--learned-exploiter-min-rounds" in daemon_help.stdout
    assert "--data-engineering-round-dir" in schedule_help.stdout
    assert "--data-engineering-validation-report" in schedule_help.stdout
    assert "--data-engineering-min-metadata-coverage" in schedule_help.stdout
    assert "--data-engineering-round-dir" in daemon_help.stdout
    assert "--data-engineering-validation-report" in daemon_help.stdout
    assert "--data-engineering-min-metadata-coverage" in daemon_help.stdout
    assert "--underdog-residual-round-dir" in schedule_help.stdout
    assert "--underdog-residual-validation-report" in schedule_help.stdout
    assert "--underdog-min-attack-residual-coverage" in schedule_help.stdout
    assert "--underdog-residual-round-dir" in daemon_help.stdout
    assert "--underdog-residual-validation-report" in daemon_help.stdout
    assert "--underdog-min-attack-residual-coverage" in daemon_help.stdout
    assert "--real-calibration-validation-samples-jsonl" in schedule_help.stdout
    assert "--real-calibration-validation-model-json" in schedule_help.stdout
    assert "--real-calibration-validation-report" in schedule_help.stdout
    assert "--real-calibration-validation-samples-jsonl" in daemon_help.stdout
    assert "--real-calibration-validation-model-json" in daemon_help.stdout
    assert "--real-calibration-validation-report" in daemon_help.stdout
    assert "--production-readiness-report-json" in schedule_help.stdout
    assert "--production-readiness-include-scheduled-reports" in schedule_help.stdout
    assert "--production-readiness-report" in schedule_help.stdout
    assert "--production-readiness-required-schema-version" in schedule_help.stdout
    assert "--production-readiness-report-json" in daemon_help.stdout
    assert "--production-readiness-include-scheduled-reports" in daemon_help.stdout
    assert "--production-readiness-report" in daemon_help.stdout
    assert "--production-readiness-required-schema-version" in daemon_help.stdout
    assert "--league-health-round-dir" in schedule_help.stdout
    assert "--league-selfplay-health-report" in schedule_help.stdout
    assert "--league-health-min-payoff-density" in schedule_help.stdout
    assert "--league-health-round-dir" in daemon_help.stdout
    assert "--league-selfplay-health-report" in daemon_help.stdout
    assert "--league-health-min-payoff-density" in daemon_help.stdout
    assert "--v4-conformance-report-json" in schedule_help.stdout
    assert "--v4-conformance-include-scheduled-reports" in schedule_help.stdout
    assert "--v4-conformance-validation-report" in schedule_help.stdout
    assert "--v4-conformance-report-json" in daemon_help.stdout
    assert "--v4-conformance-include-scheduled-reports" in daemon_help.stdout
    assert "--v4-conformance-validation-report" in daemon_help.stdout


def test_recurring_training_scheduler_stops_on_failed_job(tmp_path):
    def schedule_factory(iteration: int, run_dir):
        return TrainingSchedule(
            schedule_id=f"scheduler-r{iteration:04d}",
            root_dir=str(run_dir),
            registry_path=str(run_dir / "registry.json"),
            created_at=1.0,
            jobs=(),
        )

    def failed_runner(schedule, **_kwargs):
        return TrainingRunSummary(
            schedule_id=schedule.schedule_id,
            executed=True,
            jobs=(
                TrainingJobStatus(
                    job_id="bad-job",
                    status="failed",
                    returncode=2,
                    started_at=1.0,
                    finished_at=2.0,
                ),
            ),
        )

    state = run_recurring_training_scheduler(
        scheduler_id="unit-scheduler",
        root_dir=tmp_path / "scheduler",
        schedule_factory=schedule_factory,
        iterations=3,
        interval_seconds=0.0,
        execute=True,
        run_schedule_fn=failed_runner,
    )

    assert state.stopped is True
    assert state.stop_reason == "failed_jobs"
    assert state.iterations[0].failed_jobs == ("bad-job",)
    assert len(state.iterations) == 1


def test_run_training_schedule_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_training_schedule", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--out-schedule" in result.stdout
    assert "--single-team-train-jsonl" in result.stdout
    assert "--belief-round-dir" in result.stdout
    assert "--belief-round-holdout-fraction" in result.stdout
    assert "--defense-teacher-jsonl" in result.stdout
    assert "--real-round-dir" in result.stdout
    assert "--active-real-feedback-dir" in result.stdout
    assert "--build-real-calibration-samples-jsonl" in result.stdout
    assert "--real-meta-db-jsonl" in result.stdout
    assert "--mask-teacher-jsonl" in result.stdout
    assert "--execute" in result.stdout
    assert "--monitor-resources" in result.stdout


def test_run_training_scheduler_daemon_script_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "masked_team_league.cli.commands.run_training_scheduler_daemon", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--interval-seconds" in result.stdout
    assert "--iterations" in result.stdout
    assert "--state-json" in result.stdout
    assert "--belief-round-dir" in result.stdout
    assert "--belief-round-holdout-fraction" in result.stdout
    assert "--real-round-dir" in result.stdout
    assert "--active-real-feedback-dir" in result.stdout
    assert "--build-real-calibration-samples-jsonl" in result.stdout
    assert "--real-meta-db-jsonl" in result.stdout
    assert "--mask-teacher-jsonl" in result.stdout
