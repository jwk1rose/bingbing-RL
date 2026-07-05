from __future__ import annotations

from pathlib import Path

from masked_team_league.reporting.ablation import AblationSuiteReport
from masked_team_league.reporting.metrics import DailyTrainingReport
from masked_team_league.reporting.contracts import (
    OUTPUT_CONTRACTS,
    REQUIRED_RUNTIME_TRAINING_SCHEMA_VERSIONS,
    output_contract_registry,
)
from masked_team_league.real_platform.calibration import RealCalibrationIngestionSummary, RealCalibrationSampleBuildSummary, VersionDriftReport
from masked_team_league.real_platform.calibration import build_real_calibration_validation_report
from masked_team_league.reporting.reports import (
    build_active_query_feedback_report,
    build_active_real_query_dispatch_validation_report,
    build_attack_oracle_failure_validation_report,
    build_belief_real_distribution_validation_report,
    build_data_engineering_validation_report,
    build_defense_anti_meta_effectiveness_report,
    build_learned_exploiter_validation_report,
    build_league_selfplay_health_report,
    build_mask_explanation_validation_report,
    build_production_readiness_report,
    build_underdog_residual_validation_report,
)
from masked_team_league.training.schedule import (
    RecurringTrainingSchedulerState,
    TrainingRunSummary,
    TrainingSchedule,
)


def test_output_contract_registry_covers_runtime_and_training_artifacts():
    registry = output_contract_registry()
    versions = {contract.schema_version for contract in registry}

    assert REQUIRED_RUNTIME_TRAINING_SCHEMA_VERSIONS <= versions
    assert len(versions) == len(registry)
    assert set(OUTPUT_CONTRACTS) == versions

    for contract in registry:
        assert contract.module
        assert contract.artifact
        assert contract.required_fields
        assert "schema_version" in contract.required_fields


def test_output_contract_registry_versions_are_documented():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path("docs/module_output_contracts.md"),
            Path("docs/core_tables_schema.md"),
            Path("docs/run_metadata_schema.md"),
            Path("docs/legality_diagnostics_schema.md"),
        )
    )

    for schema_version in REQUIRED_RUNTIME_TRAINING_SCHEMA_VERSIONS:
        assert schema_version in docs


def test_report_and_training_outputs_emit_registered_schema_versions(tmp_path: Path):
    daily = DailyTrainingReport(
        date="2026-07-05",
        sim_games=1,
        real_matches=0,
        single_model={},
        attack_oracle={},
        defense_oracle={},
        league={},
        underdog={},
        active_queries=(),
        failure_cases=(),
    ).to_json_dict()
    schedule = TrainingSchedule(
        schedule_id="unit",
        root_dir="exports/unit",
        registry_path="exports/unit/checkpoint_registry.json",
        created_at=1.0,
        jobs=(),
    ).to_json_dict()
    run_summary = TrainingRunSummary(schedule_id="unit", executed=False, jobs=()).to_json_dict()
    scheduler_state = RecurringTrainingSchedulerState(
        scheduler_id="sched",
        root_dir="exports/sched",
        stopped=True,
        stop_reason=None,
        iterations=(),
    ).to_json_dict()
    ingestion = RealCalibrationIngestionSummary(
        round_dir="round",
        db_path="real_meta.jsonl",
        round_id="round_0001",
        records_added=0,
        skipped_pairs=0,
        mean_match_result=0.0,
        season="S29",
        server="oracle_backend",
    ).to_json_dict()
    sample_build = RealCalibrationSampleBuildSummary(
        out_jsonl="real_calibration_samples.jsonl",
        source_dirs=("round",),
        samples_written=1,
        skipped_pairs=0,
        mean_label=1.0,
        mean_sim_probability=0.5,
        source_kinds=("league_round_artifact",),
    ).to_json_dict()
    drift = VersionDriftReport(
        baseline_season="S28",
        current_season="S29",
        baseline_records=0,
        current_records=0,
        baseline_mean_match_result=0.0,
        current_mean_match_result=0.0,
        match_result_delta=0.0,
        observation_overlap=0.0,
        drift_detected=False,
    ).to_json_dict()
    ablation = AblationSuiteReport(
        suite_id="suite",
        date="2026-07-05",
        baseline_variant="baseline",
        variants=("baseline",),
        missing_required_variants=(),
        variant_reports={},
        deltas_vs_baseline={},
    ).to_json_dict()
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    (round_dir / "scored_defenses.jsonl").write_text(
        '{"defense_id":"def-1","hidden_count":1,"defense_risk_report":{"hidden_count":1,"learned_mask_score":1.0,"counter_attack_risk_report":{},"mask_explanation":{"hidden_slot_explanations":[{"team_index":0}]}}}\n',
        encoding="utf-8",
    )
    (round_dir / "candidates.jsonl").write_text(
        '{"attack_id":"atk-1","defense_id":"def-1","belief_domain_stats":[["real_record_count",1],["real_exact_record_count",1],["real_similarity_mean",1],["real_match_result_mean",1]]}\n',
        encoding="utf-8",
    )
    (round_dir / "oracle_pairs.jsonl").write_text(
        '{"attack_id":"atk-1","defense_id":"def-1","attack_success":1}\n',
        encoding="utf-8",
    )
    (round_dir / "active_queries.jsonl").write_text(
        '{"queue":"real","query_id":"q1","query_type":"underdog","attack_id":"atk-1","defense_id":"def-1","score":1}\n',
        encoding="utf-8",
    )
    (round_dir / "oracle_results.jsonl").write_text('{"status":"completed"}\n', encoding="utf-8")
    (round_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (round_dir / "league_state.json").write_text(
        '{"iteration":1,"attack_pool":[{"strategy_id":"atk-1","role":"main","strength":0.7,"diversity_cluster":"a","created_iteration":1,"active":true}],"defense_pool":[{"strategy_id":"def-1","role":"main","strength":0.6,"diversity_cluster":"d","created_iteration":1,"active":true}],"payoffs":[{"attack_id":"atk-1","defense_id":"def-1","attack_success":0.4,"games":3}]}\n',
        encoding="utf-8",
    )
    (round_dir / "run_metadata.json").write_text(
        '{"schema_version":"run_metadata.v1","run_id":"round","output_artifacts":[]}\n',
        encoding="utf-8",
    )
    table_dir = round_dir / "tables"
    table_dir.mkdir()
    for filename, table in {
        "loadouts.jsonl": "LoadoutTable",
        "single_matchups.jsonl": "SingleMatchupTable",
        "plan_matches.jsonl": "PlanMatchTable",
        "observations.jsonl": "ObservationTable",
        "league_strategies.jsonl": "LeagueStrategyTable",
    }.items():
        (table_dir / filename).write_text(
            f'{{"schema_version":"core_tables.v1","table":"{table}"}}\n',
            encoding="utf-8",
        )
    oracle_output = tmp_path / "attack_oracle_output.json"
    oracle_output.write_text(
        '{"schema_version":"attack_oracle_output.v1","risk_report":{"failure":"no legal belief candidates","failure_code":"NO_LEGAL_BELIEF_CANDIDATES","failure_stage":"belief"},"diagnostics":[{"code":"NO_LEGAL_BELIEF_CANDIDATES","stage":"belief"}]}\n',
        encoding="utf-8",
    )
    defense_teacher = tmp_path / "defense_teacher.jsonl"
    defense_teacher.write_text(
        '{"round_id":"round_0001","defense_id":"def-1","defense_role":"anti_meta","survival_rate":0.7,"meta_attack_success":0.5,"anti_meta_residual_target":0.2}\n',
        encoding="utf-8",
    )
    active_real_dispatch_validation_json = tmp_path / "active_real_dispatch_validation.json"
    active_real_dispatch_validation_json.write_text(
        '{"schema_version":"active_real_query_dispatch_validation.v1","module":"ActiveRealQueryDispatch","round_dir":"round","out_dir":"real_queries","queued_queries":1,"dispatchable_queries":1,"skipped_queries":0,"dispatched_pairs":1,"oracle_requests":3,"oracle_result_errors":0,"completion_rate":1.0,"attack_teacher_rows":1,"defense_teacher_rows":1,"teacher_feedback_complete":true,"real_query_queue_validated":true,"submitted_request_count":3}\n',
        encoding="utf-8",
    )
    mask_validation = build_mask_explanation_validation_report(round_dir)
    active_query_feedback = build_active_query_feedback_report(round_dir)
    active_real_dispatch_validation = build_active_real_query_dispatch_validation_report(
        [active_real_dispatch_validation_json],
        min_dispatched_pairs=1,
    )
    belief_real_validation = build_belief_real_distribution_validation_report(round_dir)
    data_engineering_validation = build_data_engineering_validation_report([round_dir])
    underdog_residual_validation = build_underdog_residual_validation_report([round_dir])
    defense_anti_meta = build_defense_anti_meta_effectiveness_report([defense_teacher])
    league_health = build_league_selfplay_health_report(
        [round_dir],
        required_attack_roles=("main",),
        required_defense_roles=("main",),
        min_attack_pool=1,
        min_defense_pool=1,
        min_total_clusters=2,
        min_payoff_density=1.0,
    )
    training_root = tmp_path / "training"
    round_training = training_root / "round_0001"
    round_training.mkdir(parents=True)
    (round_training / "attack_teacher.jsonl").write_text(
        '{"round_id":"round_0001","attack_id":"atk-1","attack_role":"exploiter","attack_success":0.7,"target_defense_id":"def-1","target_defense_hash":"hash-1","target_baseline_break_rate":0.5,"exploiter_residual_target":0.2}\n',
        encoding="utf-8",
    )
    (round_training / "defense_teacher.jsonl").write_text(
        '{"round_id":"round_0001","defense_id":"def-1","defense_role":"anti_meta","survival_rate":0.7,"meta_attack_success":0.5,"anti_meta_residual_target":0.2}\n',
        encoding="utf-8",
    )
    selfplay_root = tmp_path / "selfplay"
    selfplay_root.mkdir()
    (selfplay_root / "orchestrator_state.json").write_text(
        '{"rounds":[{"round_id":"round_0001","oracle_requests":3}],"latest_attack_proposal_checkpoint":"attack.pt","latest_defense_proposal_checkpoint":"defense.pt"}\n',
        encoding="utf-8",
    )
    learned_exploiter_validation = build_learned_exploiter_validation_report(
        selfplay_root=selfplay_root,
        training_root=training_root,
        min_rounds=1,
        min_oracle_requests=1,
    )
    attack_failure_validation = build_attack_oracle_failure_validation_report(oracle_output_paths=(oracle_output,))
    production_ready_source = tmp_path / "learned_exploiter_validation_report.json"
    production_ready_source.write_text(
        '{"schema_version":"learned_exploiter_validation_report.v1","production_ready":true,"red_line_violations":[]}\n',
        encoding="utf-8",
    )
    production_readiness = build_production_readiness_report(
        [production_ready_source],
        required_schema_versions=("learned_exploiter_validation_report.v1",),
    )
    calibration_samples = tmp_path / "real_calibration_holdout.jsonl"
    calibration_samples.write_text(
        '{"sim_probability":0.8,"label":0,"features":{"hidden_fraction":0}}\n'
        '{"sim_probability":0.8,"label":1,"features":{"hidden_fraction":1}}\n',
        encoding="utf-8",
    )
    calibration_json = tmp_path / "real_calibration_model.json"
    calibration_json.write_text('{"model":{"logit_scale":1,"bias":0,"feature_weights":{}}}\n', encoding="utf-8")
    real_calibration_validation = build_real_calibration_validation_report(
        samples_jsonl=(calibration_samples,),
        calibration_json=calibration_json,
        min_samples=1,
        min_brier_improvement=-1.0,
        min_ece_improvement=-1.0,
    )

    payloads = (
        daily,
        schedule,
        run_summary,
        scheduler_state,
        ingestion,
        sample_build,
        drift,
        ablation,
        active_query_feedback,
        active_real_dispatch_validation,
        mask_validation,
        belief_real_validation,
        data_engineering_validation,
        underdog_residual_validation,
        defense_anti_meta,
        league_health,
        learned_exploiter_validation,
        attack_failure_validation,
        production_readiness,
        real_calibration_validation,
    )
    versions = {payload["schema_version"] for payload in payloads}

    assert versions <= set(OUTPUT_CONTRACTS)
    assert versions == {
        "daily_training_report.v1",
        "training_schedule.v1",
        "training_run_summary.v1",
        "recurring_training_scheduler_state.v1",
        "real_calibration_ingestion_summary.v1",
        "real_calibration_sample_build_summary.v1",
        "version_drift_report.v1",
        "ablation_suite_report.v1",
        "active_query_feedback_report.v1",
        "active_real_query_dispatch_validation_report.v1",
        "mask_explanation_validation_report.v1",
        "belief_real_distribution_validation_report.v1",
        "data_engineering_validation_report.v1",
        "underdog_residual_validation_report.v1",
        "defense_anti_meta_effectiveness_report.v1",
        "league_selfplay_health_report.v1",
        "learned_exploiter_validation_report.v1",
        "attack_oracle_failure_validation_report.v1",
        "production_readiness_report.v1",
        "real_calibration_validation_report.v1",
    }
