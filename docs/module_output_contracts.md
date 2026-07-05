# Module Output Contracts

This document defines the stable JSON contracts for runtime module outputs. The goal is to make module results inspectable by runners, dashboards, reports, and training ingestion without depending on Python dataclass internals.

## Common Rules

- Every contract has `schema_version`.
- Every contract has `module`.
- Every contract has `metadata`, derived from `ResultMetadata`.
- Every contract has `explanation`, `risk_report`, and `diagnostics`.
- Plans may be included as JSON objects, but hashes are always present for joins and compact indexing.
- `diagnostics` is an array of structured issues; successful outputs may use an empty array.

## AttackOracleOutput

Schema version: `attack_oracle_output.v1`

Required fields:

- `schema_version`: `attack_oracle_output.v1`.
- `module`: `AttackOracle`.
- `metadata`: model/data/simulator/seed/calibration lineage.
- `ranked_attack_hashes`: hashes of ranked attack plans.
- `ranked_attacks`: full attack plan payloads.
- `predicted_scores`: surrogate scores.
- `simulated_scores`: successive-halving simulation scores.
- `belief_summary`: belief entropy, feasible count, top-1/top-2 gap, and domain stats.
- `halving_traces`: per-stage games, kept attack hashes, and scores.
- `explanation`: compact human/debug explanation.
- `risk_report`: lane rates, worst-case belief case, backups, and other risk fields.
- `diagnostics`: structured module failures.

Stable `risk_report` fields for underdog objectives:

- `objective_score`: score used for final objective ranking.
- `attack_cost`: selected attack resource cost.
- `reference_defense_cost`: weighted belief defense resource cost.
- `underdog_gap`: positive when the selected attack is cheaper than the belief/reference defense.
- `underdog_residual_bonus`: objective bonus contributed by the underdog gap. This is not a simulator win rate.

## Attack Oracle Failure Validation Report

Schema version: `attack_oracle_failure_validation_report.v1`

This report audits whether AttackOracle failure outputs and round candidate risk reports preserve machine-readable failure annotations and diagnostics.

Required fields:

- `schema_version`: `attack_oracle_failure_validation_report.v1`.
- `module`: `AttackOracleFailureValidationReport`.
- `oracle_output_paths`: direct `attack_oracle_output.v1` JSON inputs scanned.
- `round_dirs`: league round directories whose `candidates.jsonl` rows were scanned.
- `oracle_outputs`: count of direct output JSON paths supplied.
- `candidate_rows`: candidate rows scanned from round artifacts.
- `candidate_risk_report_rows`: candidate rows with `attack_risk_report`.
- `checked_rows`: total risk-bearing rows checked.
- `failure_rows`: rows with `failure`, `failure_code`, or `failure_stage`.
- `annotated_failure_rows`: failure rows carrying both `failure_code` and `failure_stage`.
- `diagnostic_failure_rows`: failure rows with a matching structured diagnostic.
- `failure_annotation_coverage`: `annotated_failure_rows / failure_rows`, or 1.0 when there are no failures.
- `failure_diagnostic_coverage`: `diagnostic_failure_rows / failure_rows`, or 1.0 when there are no failures.
- `normal_risk_report_rows`: non-failure risk rows.
- `failure_stage_counts`: failure rows grouped by stage.
- `failure_code_counts`: failure rows grouped by code.
- `validation_rows`: compact per-row validation details.
- `red_line_violations`: machine-readable failures such as `candidate_risk_report_missing`, `failure_code_missing`, `failure_stage_missing`, `failure_annotation_coverage_low`, and `failure_diagnostic_coverage_low`.

## League Round Candidate Rows

Artifact: `candidates.jsonl`

Stable belief inspection fields:

- `belief_candidates`: feasible weighted belief candidate count used by AttackOracle.
- `belief_entropy`: entropy of the weighted belief distribution.
- `belief_top1_top2_gap`: probability gap between the top two belief candidates.
- `belief_domain_stats`: key/value diagnostics from BeliefEngine, including hidden-domain sizes, candidate weight stats, `real_record_count`, `real_exact_record_count`, `real_similar_record_count`, `real_similarity_mean`, `real_match_result_mean`, `defense_pool_record_count`, and `ranker_applied`.

## DefenseOracleOutput

Schema version: `defense_oracle_output.v1`

Required fields:

- `schema_version`: `defense_oracle_output.v1`.
- `module`: `DefenseOracle`.
- `metadata`: model/data/simulator/seed/calibration lineage.
- `best_defense_hash`: hash of selected defense, or null.
- `best_defense`: full selected defense plan, or null.
- `backup_defense_hashes`: hashes of backup defenses.
- `backup_defenses`: full backup defense plans.
- `estimated_attack_success`: estimated break rate from the counter-attack path.
- `ambiguity_score`: mask/belief ambiguity score.
- `worst_case_attack_hash`: hash of the best counter attack, or null.
- `worst_case_attack`: full worst-case attack plan, or null.
- `explanation`: compact human/debug explanation.
- `risk_report`: break/survival rates, mask explanation, backups, counter-attack risk.
- `diagnostics`: structured module failures.

Stable `risk_report` fields for underdog objectives:

- `defense_cost`: selected defense roster resource cost.
- `reference_attack_cost`: weighted attack-meta resource cost when attack meta exists.
- `underdog_defense_gap`: positive when the selected defense is cheaper than the attack-meta reference.
- `underdog_residual_bonus`: objective bonus contributed by the defense underdog gap. This is not a break-rate or survival-rate metric.

## ActiveRealQueryDispatch Validation

Schema version: `active_real_query_dispatch_validation.v1`

Required fields:

- `schema_version`: `active_real_query_dispatch_validation.v1`.
- `module`: `ActiveRealQueryDispatch`.
- `round_dir`: source round artifact directory.
- `out_dir`: feedback artifact directory.
- `queued_queries`: real-queue rows selected from `active_queries.jsonl`.
- `dispatchable_queries`: rows with both attack and defense plan artifacts available.
- `skipped_queries`: rows skipped before oracle submission.
- `skipped_query_reasons`: count by skip reason.
- `dispatched_pairs`: attack/defense pairs submitted to the evaluator.
- `oracle_requests`: concrete simulator request count.
- `oracle_result_errors`: simulator results whose status is not completed or cached.
- `completion_rate`: completed-or-cached request fraction.
- `attack_teacher_rows`: attack feedback rows written.
- `defense_teacher_rows`: defense feedback rows written.
- `teacher_feedback_complete`: true when every dispatched pair produced attack and defense teacher feedback.
- `real_query_queue_validated`: true when at least one real query was present, all dispatchable pairs returned without oracle errors, and teacher feedback is complete.
- `submitted_request_count`: same request count as `oracle_requests`, kept for dashboards that distinguish queued rows from simulator requests.

## Active Real-Query Dispatch Validation Report

Schema version: `active_real_query_dispatch_validation_report.v1`

This scheduled report aggregates one or more `active_real_query_dispatch_validation.v1` files so recurring training and production readiness can verify that active real-query results reached proposal-teacher feedback.

Required fields:

- `schema_version`: `active_real_query_dispatch_validation_report.v1`.
- `module`: `ActiveRealQueryDispatchValidationReport`.
- `validation_report_paths`: source dispatch validation JSON files.
- `reports`: total validation report paths checked.
- `readable_reports`: validation reports loaded as JSON objects.
- `read_error_reports`: missing or malformed validation reports.
- `queued_queries`: aggregate real-query rows selected from active-query queues.
- `dispatchable_queries`: aggregate rows with complete attack/defense plan artifacts.
- `skipped_queries`: aggregate pre-submission skipped rows.
- `skipped_query_reasons`: merged skip-reason counts.
- `dispatched_pairs`: aggregate pairs submitted to the evaluator.
- `oracle_requests`: aggregate concrete simulator request count.
- `oracle_result_errors`: aggregate simulator result errors.
- `completion_rate`: completed-or-cached requests divided by all submitted requests.
- `attack_teacher_rows`: aggregate attack feedback rows written.
- `defense_teacher_rows`: aggregate defense feedback rows written.
- `teacher_feedback_complete_reports`: readable child reports with complete attack and defense teacher feedback.
- `real_query_queue_validated_reports`: readable child reports whose queue validation flag is true.
- `report_rows`: normalized per-child validation status and metrics.
- `red_line_violations`: aggregate failures such as `active_real_dispatch_reports_low`, `dispatch_validation_report_read_error`, `no_active_real_queries`, `active_real_dispatched_pairs_low`, `active_real_oracle_result_errors`, `active_real_completion_rate_low`, `active_real_teacher_feedback_incomplete`, and `active_real_queue_not_validated`.
- `production_ready`: true only when all active real-query dispatch gates pass.

## Attack Teacher Feedback Rows

JSONL rows consumed by proposal training may come from self-play artifacts or active real-query feedback. Required stable fields:

- `teacher_group_id`: grouping key for candidate weighting, usually round plus defense target.
- `defense_id`: defense target identifier.
- `attack_id`: attack candidate identifier.
- `attack_role`: `main`, `exploiter`, or `underdog`.
- `attack_plan`: full attack plan payload.
- `attack_success`: oracle/surrogate attack success target.
- `gap_target`: belief or decision-gap auxiliary target.
- `target_defense_id`: defense target used by best-response feedback.
- `target_defense_hash`: canonical target defense hash when available.
- `target_defense_strength`: defense survival/strength estimate when available.
- `target_baseline_break_rate`: baseline attack break rate for the target defense.
- `exploiter_residual_target`: realized attack success minus target baseline break rate.
- `role_weight`: multiplicative training weight for role-conditioned exploiter/underdog feedback.
- `source`: `selfplay_orchestrator`, `active_real_query`, or another explicit provenance value.

## Exploiter Effectiveness Report

Schema version: `exploiter_effectiveness_report.v1`

This report audits whether learned attack exploiters and underdog exploiters are producing positive residual against their target defenses.

Required fields:

- `schema_version`: `exploiter_effectiveness_report.v1`.
- `teacher_jsonl_paths`: attack teacher JSONL inputs used by the report.
- `teacher_rows`: total attack teacher rows scanned.
- `target_feedback_rows`: rows with target defense id/hash, target baseline break rate, and exploiter residual.
- `target_feedback_coverage`: `target_feedback_rows / teacher_rows`.
- `role_stats`: per-role samples, target coverage, mean attack success, mean baseline break rate, mean residual, positive residual rate, mean role weight, and source counts.
- `anti_meta`: aggregate over `exploiter` and `underdog` roles, including residual lift against `main`.
- `round_stats`: the same role and anti-meta metrics grouped by inferred round id.
- `trend`: ordered round ids, per-round anti-meta residuals, first/last residual, residual delta, slope per round, and whether the trend is improving.
- `red_line_violations`: machine-readable failures such as `target_feedback_coverage_low`, `no_anti_meta_samples`, `anti_meta_residual_non_positive`, and `anti_meta_positive_rate_low`.

## Defense Anti-Meta Effectiveness Report

Schema version: `defense_anti_meta_effectiveness_report.v1`

This report audits whether defense teacher rows contain usable anti-meta residual feedback and whether defense residual quality improves across rounds.

Required fields:

- `schema_version`: `defense_anti_meta_effectiveness_report.v1`.
- `module`: `DefenseAntiMetaEffectivenessReport`.
- `teacher_jsonl_paths`: defense teacher JSONL inputs used by the report.
- `teacher_rows`: total defense teacher rows scanned.
- `anti_meta_feedback_rows`: rows with `survival_rate`, `meta_attack_success`, and `anti_meta_residual_target`.
- `anti_meta_feedback_coverage`: `anti_meta_feedback_rows / teacher_rows`.
- `role_stats`: per-defense-role samples, feedback coverage, mean survival rate, mean meta attack success, mean residual, positive residual rate, mean role weight, and source counts.
- `anti_meta`: aggregate defense anti-meta residual metrics over rows with feedback.
- `round_stats`: the same role and anti-meta metrics grouped by inferred round id.
- `trend`: ordered round ids, per-round defense anti-meta residuals, first/last residual, residual delta, slope per round, and whether the trend is improving.
- `red_line_violations`: machine-readable failures such as `anti_meta_feedback_coverage_low`, `no_defense_anti_meta_feedback`, `defense_anti_meta_residual_non_positive`, `defense_anti_meta_positive_rate_low`, and `defense_anti_meta_residual_trend_non_positive`.

## Learned Exploiter Validation Report

Schema version: `learned_exploiter_validation_report.v1`

This report combines multi-round self-play scale checks with the attack exploiter and defense anti-meta effectiveness reports. It is the production validation gate for learned attack/defense proposal feedback loops.

Required fields:

- `schema_version`: `learned_exploiter_validation_report.v1`.
- `module`: `LearnedExploiterValidationReport`.
- `selfplay_root`: self-play artifact root containing `orchestrator_state.json`.
- `training_root`: training artifact root containing per-round teacher JSONL files.
- `rounds`: number of completed self-play rounds in the orchestrator state.
- `round_ids`: ordered round ids from the orchestrator state.
- `oracle_requests`: total oracle requests recorded across the self-play rounds.
- `latest_attack_proposal_checkpoint`: latest attack proposal checkpoint path from the orchestrator state.
- `latest_defense_proposal_checkpoint`: latest defense proposal checkpoint path from the orchestrator state.
- `exploiter_report`: embedded `exploiter_effectiveness_report.v1` payload.
- `defense_anti_meta_report`: embedded `defense_anti_meta_effectiveness_report.v1` payload.
- `red_line_violations`: combined failures such as `validation_rounds_low`, `oracle_requests_low`, missing latest checkpoints, `attack_*` report failures, and `defense_*` report failures.
- `production_ready`: true only when `red_line_violations` is empty.

## League Self-Play Health Report

Schema version: `league_selfplay_health_report.v1`

This report validates the LeagueManager / PSRO self-play state from one or more round artifact directories. It is the scheduler-readable gate for attack/defense pool size, active retention, role coverage, diversity clusters, payoff matrix coverage, and new-strategy strength deltas.

Required fields:

- `schema_version`: `league_selfplay_health_report.v1`.
- `module`: `LeagueSelfPlayHealthReport`.
- `round_dirs`: source league round artifact directories.
- `rounds`: number of round directories summarized.
- `latest_round_dir`: round directory selected as the latest by `(iteration, input order)`.
- `latest_iteration`: latest league iteration.
- `attack_pool`: attack strategies in the latest league state.
- `defense_pool`: defense strategies in the latest league state.
- `active_attack_pool`: active attack strategies after retention.
- `active_defense_pool`: active defense strategies after retention.
- `active_pool_fraction`: active strategies divided by all attack and defense strategies.
- `attack_clusters`: attack-side diversity clusters.
- `defense_clusters`: defense-side diversity clusters.
- `total_clusters`: combined attack/defense diversity clusters.
- `attack_role_counts`: latest attack-pool role counts.
- `defense_role_counts`: latest defense-pool role counts.
- `attack_role_coverage`: required attack roles present in active attack strategies.
- `defense_role_coverage`: required defense roles present in active defense strategies.
- `payoff_entries`: payoff matrix rows in the latest league state.
- `payoff_density`: payoff entries divided by `attack_pool * defense_pool`.
- `new_attack_strength_delta`: current-iteration attack strength mean minus prior attack strength mean.
- `new_defense_strength_delta`: current-iteration defense strength mean minus prior defense strength mean.
- `round_reports`: per-round pool, role, payoff, oracle, and growth metrics.
- `red_line_violations`: failures such as `attack_pool_too_small`, `defense_pool_too_small`, `league_cluster_collapse`, `attack_role_coverage_low`, `defense_role_coverage_low`, `payoff_density_low`, `no_payoffs`, `active_pool_fraction_low`, `new_attack_strength_delta_low`, and `new_defense_strength_delta_low`.
- `production_ready`: true only when all league health gates pass.

## Production Readiness Report

Schema version: `production_readiness_report.v1`

This report aggregates module validation reports into one scheduler-readable production gate.

Required fields:

- `schema_version`: `production_readiness_report.v1`.
- `module`: `ProductionReadinessReport`.
- `report_paths`: validation report files included in the gate.
- `reports`: total report paths checked.
- `readable_reports`: report files that loaded as JSON objects.
- `read_error_reports`: missing or malformed report files.
- `clean_reports`: reports without red lines and without explicit `production_ready=false`.
- `red_line_reports`: readable child reports with non-empty `red_line_violations`.
- `clean_report_rate`: clean reports divided by total report paths.
- `production_ready_checked_reports`: child reports that expose `production_ready`.
- `production_ready_reports`: child reports explicitly marked ready.
- `production_ready_false_reports`: child reports explicitly marked not ready.
- `schema_counts`: count of readable reports by schema version.
- `required_schema_versions`: required child schema versions for this gate.
- `missing_required_schema_versions`: required schema versions not present.
- `report_rows`: per-report schema, schema_versions, red-line, readiness, and read-error details.
- `red_line_violations`: aggregate gate failures such as `report_read_error`, `required_schema_missing`, `red_line_reports_present`, `production_ready_false`, and `clean_report_rate_low`.
- `production_ready`: true only when every production readiness gate passes.

`schema_counts` includes top-level report schemas and known nested report containers. This is required for `scripts/ingest_real_calibration.py`, which writes a wrapper `real_calibration_report.json` containing one or more nested `real_calibration_ingestion_summary.v1` rows under `ingestions`.

## v4 Conformance Validation Report

Schema version: `v4_conformance_validation_report.v1`

This report maps validation evidence to the v4 production priorities in `docs/spec_conformance_matrix.md`.

Required fields:

- `schema_version`: `v4_conformance_validation_report.v1`.
- `module`: `V4ConformanceValidationReport`.
- `report_paths`: validation report files included in the conformance gate.
- `reports`: total report paths checked.
- `readable_reports`: report files that loaded as JSON objects.
- `read_error_reports`: missing or malformed report files.
- `schema_counts`: count of readable reports by top-level and known nested schema versions.
- `requirements_total`: number of v4 production-priority requirement rows checked.
- `passed_requirements`: requirement rows with all required evidence present and clean.
- `failed_requirements`: requirement rows that are missing evidence, have child red lines, or have explicit `production_ready=false`.
- `requirements`: per-requirement evidence rows, including `requirement_id`, `required_schema_versions`, `evidence_schema_versions`, `missing_schema_versions`, `child_red_line_violations`, `production_not_ready_paths`, and `status`.
- `report_rows`: per-report schema, schema_versions, red-line, readiness, and read-error details.
- `red_line_violations`: aggregate failures such as `report_read_error`, `<requirement>_evidence_missing`, `<requirement>_red_lines_present`, and `<requirement>_production_not_ready`.
- `production_ready`: true only when every v4 production-priority requirement row passes.

## Mask Explanation Validation Report

Schema version: `mask_explanation_validation_report.v1`

This report audits whether defense rows contain usable mask/risk explanations for learned mask validation.

Required fields:

- `schema_version`: `mask_explanation_validation_report.v1`.
- `module`: `MaskExplanationValidationReport`.
- `round_dir`: source league round artifact directory.
- `defenses`: scored defense row count.
- `risk_report_rows`: defense rows with `defense_risk_report`.
- `mask_explanation_rows`: defense rows with `defense_risk_report.mask_explanation`.
- `learned_mask_score_rows`: defense rows with numeric `learned_mask_score`.
- `counter_attack_risk_rows`: defense rows with nested counter-attack risk reports.
- `defenses_with_no_hidden_slots`: selected defenses whose mask hid zero slots.
- `total_hidden_slots`: total hidden slots across scored defenses.
- `explained_hidden_slots`: hidden slots covered by `hidden_slot_explanations`.
- `hidden_explanation_coverage`: `explained_hidden_slots / total_hidden_slots`.
- `mean_hidden_count`: average hidden slots per defense.
- `mean_learned_mask_score`: average learned mask score where present.
- `defense_rows`: per-defense validation details.
- `red_line_violations`: machine-readable failures such as `no_hidden_slots`, `mask_explanation_missing`, `hidden_slot_explanation_coverage_low`, `learned_mask_score_missing`, and `counter_attack_risk_missing`.

## Belief Real Distribution Validation Report

Schema version: `belief_real_distribution_validation_report.v1`

This report audits whether belief-model candidate generation is using real-distribution similarity evidence and whether the real-distribution estimate aligns with oracle feedback.

Required fields:

- `schema_version`: `belief_real_distribution_validation_report.v1`.
- `module`: `BeliefRealDistributionValidationReport`.
- `round_dir`: source league round artifact directory, or comma-joined directories for merged reports.
- `candidates`: candidate row count.
- `belief_domain_stats_rows`: candidate rows with parseable `belief_domain_stats`.
- `real_distribution_rows`: candidate rows with `real_record_count > 0`.
- `real_distribution_coverage`: `real_distribution_rows / candidates`.
- `exact_real_rows`: rows with exact real-meta matches.
- `similar_real_rows`: rows with similarity-backed real-meta matches.
- `mean_real_record_count`: average real record count for rows with real evidence.
- `mean_real_similarity`: average real similarity for rows with real evidence.
- `mean_real_match_result`: average real-meta match result for rows with real evidence.
- `mean_weight_entropy_normalized`: average belief weight entropy where present.
- `oracle_alignment_rows`: rows where real-meta estimate could be compared against `oracle_pairs.attack_success`.
- `oracle_alignment_mae`: mean absolute error between `real_match_result_mean` and oracle attack success.
- `candidate_rows`: compact per-candidate validation details.
- `red_line_violations`: machine-readable failures such as `belief_domain_stats_missing`, `real_distribution_coverage_low`, `real_record_count_low`, `real_similarity_low`, and `real_oracle_alignment_error_high`.

## Daily Training Report

Schema version: `daily_training_report.v1`

Required fields:

- `schema_version`: `daily_training_report.v1`.
- `module`: `DailyTrainingReport`.
- `date`: reporting date or run date label.
- `sim_games`: simulator request/game count covered by the report.
- `real_matches`: real-platform match count covered by the report.
- `single_model`: value-model metrics such as Brier, ECE, and AUC.
- `attack_oracle`: attack-search metrics and risk aggregates.
- `defense_oracle`: defense-search metrics and risk aggregates.
- `league`: pool, diversity, and active-query metrics.
- `underdog`: underdog sample and success metrics.
- `active_queries`: active-perception query rows included in the round.
- `failure_cases`: failed oracle/backend cases.

## Active Query Feedback Report

Schema version: `active_query_feedback_report.v1`

Required fields:

- `schema_version`: `active_query_feedback_report.v1`.
- `module`: `ActiveQueryFeedbackReport`.
- `round_dir`: source league round directory.
- `queries`: active-query rows read from `active_queries.jsonl`.
- `matched_queries`: rows matched to submitted oracle pairs.
- `unmatched_queries`: rows without matching oracle pair artifacts.
- `matched_query_coverage`: `matched_queries / queries`, or 1.0 when no queries exist.
- `real_queries`: active queries in the real queue.
- `matched_real_queries`: real-queue rows with oracle pair feedback.
- `real_query_feedback_coverage`: `matched_real_queries / real_queries`, or 1.0 when no real queries exist.
- `sim_queries`: active queries in the sim queue.
- `matched_sim_queries`: sim-queue rows with oracle pair feedback.
- `sim_query_feedback_coverage`: `matched_sim_queries / sim_queries`, or 1.0 when no sim queries exist.
- `oracle_pairs`: oracle pair rows available for joining.
- `oracle_result_rows`: raw oracle result rows inspected.
- `oracle_result_errors`: non-completed/non-cached oracle results.
- `oracle_result_error_rate`: `oracle_result_errors / oracle_results`.
- `queues`: per-queue score, cost, underdog, and realized attack-success stats.
- `query_feedback`: joined per-query feedback rows.
- `red_line_violations`: machine-readable failures such as `no_active_queries`, `active_query_feedback_coverage_low`, `real_query_count_low`, `real_query_feedback_missing`, and `oracle_result_errors`.

## Data Engineering Validation Report

Schema version: `data_engineering_validation_report.v1`

This report audits run reproducibility metadata, artifact hash references, and core table coverage across one or more league round artifact directories.

Required fields:

- `schema_version`: `data_engineering_validation_report.v1`.
- `module`: `DataEngineeringValidationReport`.
- `round_dirs`: source league round directories.
- `rounds`: number of round directories inspected.
- `metadata_files`: number of rounds with readable `run_metadata.json`.
- `metadata_coverage`: `metadata_files / rounds`.
- `artifact_refs`: input/output artifact references listed in run metadata.
- `artifact_verified`: artifact references whose resolved file exists and matches the recorded SHA-256 hash.
- `artifact_missing_count`: artifact references whose file cannot be resolved.
- `artifact_hash_mismatch_count`: artifact references with mismatched SHA-256 hashes.
- `artifact_hash_coverage`: `artifact_verified / artifact_refs`, or 1.0 when there are no artifact refs.
- `core_table_files_expected`: expected core table file count across all rounds.
- `core_table_files_found`: core table files found across all rounds.
- `core_table_coverage`: `core_table_files_found / core_table_files_expected`.
- `core_table_empty_count`: core table files that exist but have no rows.
- `core_table_schema_mismatch_count`: core table files with rows whose `schema_version` or `table` field does not match the expected table.
- `round_reports`: per-round metadata, artifact, and table diagnostics.
- `red_line_violations`: machine-readable failures such as `run_metadata_missing`, `artifact_missing`, `artifact_hash_mismatch`, `core_table_missing`, `core_table_empty`, and coverage-low red lines.

## Underdog Residual Validation Report

Schema version: `underdog_residual_validation_report.v1`

This report audits whether attack-side and defense-side underdog residual objectives are actually present in production round artifacts and whether the selected underdog rows receive positive residual bonus signal.

Required fields:

- `schema_version`: `underdog_residual_validation_report.v1`.
- `module`: `UnderdogResidualValidationReport`.
- `round_dirs`: source league round directories.
- `rounds`: number of round directories inspected.
- `attack_rows`: candidate rows inspected from `candidates.jsonl`.
- `attack_underdog_rows`: attack rows with `attack_role=underdog` or positive underdog residual markers.
- `attack_residual_rows`: underdog attack rows with both `underdog_gap` and `underdog_residual_bonus` in `attack_risk_report`.
- `attack_residual_coverage`: `attack_residual_rows / attack_underdog_rows`.
- `mean_attack_underdog_gap`: mean attack-side underdog gap over residual rows.
- `mean_attack_residual_bonus`: mean attack-side objective bonus from residual rows.
- `mean_attack_objective_score`: mean attack objective score, falling back to expected match win when needed.
- `defense_rows`: defense rows inspected from `scored_defenses.jsonl`.
- `defense_underdog_rows`: defense rows with `defense_role=underdog` or positive defense underdog residual markers.
- `defense_residual_rows`: underdog defense rows with both `underdog_defense_gap` and `underdog_residual_bonus` in `defense_risk_report`.
- `defense_residual_coverage`: `defense_residual_rows / defense_underdog_rows`.
- `mean_defense_underdog_gap`: mean defense-side underdog gap over residual rows.
- `mean_defense_residual_bonus`: mean defense-side objective bonus from residual rows.
- `mean_defense_objective_score`: mean defense objective score, falling back to survival rate or `1 - break_rate`.
- `round_reports`: per-round residual coverage and mean diagnostics.
- `red_line_violations`: machine-readable failures such as `attack_underdog_rows_missing`, `defense_underdog_rows_missing`, `attack_residual_coverage_low`, `defense_residual_coverage_low`, `attack_residual_bonus_non_positive`, and `defense_residual_bonus_non_positive`.

## Ablation Suite Report

Schema version: `ablation_suite_report.v1`

Required fields:

- `schema_version`: `ablation_suite_report.v1`.
- `module`: `AblationSuiteReport`.
- `suite_id`: ablation suite identifier.
- `date`: report date or run label.
- `baseline_variant`: variant used as baseline for deltas.
- `variants`: variants included in the report.
- `missing_required_variants`: required v4 variants not present.
- `variant_reports`: per-variant round reports and key metrics.
- `deltas_vs_baseline`: key metric deltas against the baseline variant.

## V4 Ablation Experiment Plan

Schema version: `v4_ablation_experiment_plan.v1`

Required fields:

- `schema_version`: `v4_ablation_experiment_plan.v1`.
- `suite_id`: ablation suite identifier.
- `root_dir`: root directory where variant rounds will be written.
- `baseline_variant`: baseline variant id.
- `required_variants`: complete required v4 variant list.
- `missing_required_variants`: required variants omitted by the selected plan.
- `variants`: executable per-variant commands and control metadata.

## Training Schedule

Schema version: `training_schedule.v1`

Required fields:

- `schema_version`: `training_schedule.v1`.
- `module`: `TrainingSchedule`.
- `schedule_id`: stable schedule/run id.
- `root_dir`: root output directory for training jobs.
- `registry_path`: checkpoint registry path used by the schedule.
- `created_at`: creation timestamp.
- `jobs`: ordered training DAG job definitions.
- `metadata`: extra schedule-level metadata.

## Training Run Summary

Schema version: `training_run_summary.v1`

Required fields:

- `schema_version`: `training_run_summary.v1`.
- `module`: `TrainingRunSummary`.
- `schedule_id`: schedule id being executed.
- `executed`: false for dry-run status, true for actual execution.
- `jobs`: per-job status rows with return code, timestamps, logs, and optional resource snapshots.

## Recurring Training Scheduler State

Schema version: `recurring_training_scheduler_state.v1`

Required fields:

- `schema_version`: `recurring_training_scheduler_state.v1`.
- `module`: `RecurringTrainingSchedulerState`.
- `scheduler_id`: recurring scheduler id.
- `root_dir`: scheduler output root.
- `stopped`: whether the scheduler has stopped.
- `stop_reason`: stop reason when stopped.
- `next_run_at`: next scheduled timestamp when applicable.
- `iterations`: per-iteration schedule/status paths and red-line results.

## Real Calibration Ingestion Summary

Schema version: `real_calibration_ingestion_summary.v1`

`scripts/ingest_real_calibration.py` writes these summaries inside a wrapper `real_calibration_report.json` under `ingestions`, alongside aggregate fields such as `db_jsonl`, `total_records`, and optional `drift`. Production readiness counts the nested summary schema when checking required schema versions.

Required fields:

- `schema_version`: `real_calibration_ingestion_summary.v1`.
- `module`: `RealCalibrationIngestionSummary`.
- `round_dir`: source league round directory, or source active-real feedback directory when `source_kind=active_real_query_feedback`.
- `db_path`: RealMetaDB JSONL path written or appended.
- `round_id`: source round id, inferred from the active feedback summary's `round_dir` when ingesting active-real feedback.
- `records_added`: number of real-meta records added.
- `skipped_pairs`: number of pairs skipped due missing artifacts or lane results.
- `mean_match_result`: mean match result for records added.
- `season`: season label.
- `server`: source server/backend label.
- `source_kind`: `league_round_artifact` or `active_real_query_feedback`.

## Real Calibration Sample Build Summary

Schema version: `real_calibration_sample_build_summary.v1`

Required fields:

- `schema_version`: `real_calibration_sample_build_summary.v1`.
- `module`: `RealCalibrationSampleBuildSummary`.
- `out_jsonl`: calibration sample JSONL written for feature fitting or holdout validation.
- `source_dirs`: source league round directories and/or active-real feedback directories.
- `samples_written`: sample rows written.
- `skipped_pairs`: pairs skipped because prediction, label, or full plan artifacts were missing.
- `mean_label`: mean real oracle label in written rows.
- `mean_sim_probability`: mean pre-oracle prediction probability in written rows.
- `source_kinds`: source classes represented in the output, such as `league_round_artifact` and `active_real_query_feedback`.

## Real Calibration Validation Report

Schema version: `real_calibration_validation_report.v1`

This report validates a fitted real calibration model on holdout samples and checks whether it improves simulator probabilities.

Required fields:

- `schema_version`: `real_calibration_validation_report.v1`.
- `module`: `RealCalibrationValidationReport`.
- `samples_jsonl`: holdout sample JSONL files.
- `calibration_json`: fitted real calibration model JSON.
- `samples`: holdout sample count.
- `labels_mean`: mean real label.
- `raw_prediction_mean`: mean raw simulator probability.
- `base_prediction_mean`: mean base calibrator probability.
- `calibrated_prediction_mean`: mean final calibrated probability.
- `raw_brier`: raw simulator Brier score.
- `base_brier`: base calibrator Brier score.
- `calibrated_brier`: final calibrator Brier score.
- `brier_improvement`: raw minus calibrated Brier.
- `base_brier_improvement`: base minus calibrated Brier.
- `raw_ece`: raw simulator expected calibration error.
- `base_ece`: base calibrator expected calibration error.
- `calibrated_ece`: final calibrator expected calibration error.
- `ece_improvement`: raw minus calibrated ECE.
- `base_ece_improvement`: base minus calibrated ECE.
- `feature_names`: feature keys observed in the holdout rows.
- `red_line_violations`: failures such as `real_calibration_holdout_samples_low`, `real_calibration_brier_not_improved`, and `real_calibration_ece_not_improved`.
- `production_ready`: true only when holdout sample count and calibration improvements pass thresholds.

## Version Drift Report

Schema version: `version_drift_report.v1`

Required fields:

- `schema_version`: `version_drift_report.v1`.
- `module`: `VersionDriftReport`.
- `baseline_season`: baseline season label.
- `current_season`: current season label.
- `baseline_records`: records in the baseline slice.
- `current_records`: records in the current slice.
- `baseline_mean_match_result`: baseline mean result.
- `current_mean_match_result`: current mean result.
- `match_result_delta`: current minus baseline mean result.
- `observation_overlap`: current observation overlap with baseline observations.
- `drift_detected`: true when configured drift thresholds are crossed.

## Diagnostics

For now, oracle output diagnostics use:

- `NO_LEGAL_BELIEF_CANDIDATES`: AttackOracle could not construct any legal belief completion for a mask observation. The `stage` is `belief`.
- `NO_LEGAL_ATTACK_CANDIDATES`: AttackOracle generated no legal attacks after structural filtering and optional underdog fallback. The `stage` is `candidate_generation`.
- `NO_ATTACK_SURVIVED_HALVING`: AttackOracle had candidates but none survived the successive-halving phase. The `stage` is `successive_halving`.
- `MODULE_FAILURE`: fallback for failures without a domain-specific code.

Additional codes may be added without removing existing fields.
