from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import DailyTrainingReport


ANTI_META_ROLES = {"exploiter", "underdog"}
CORE_TABLE_FILES = {
    "loadouts.jsonl": "LoadoutTable",
    "single_matchups.jsonl": "SingleMatchupTable",
    "plan_matches.jsonl": "PlanMatchTable",
    "observations.jsonl": "ObservationTable",
    "league_strategies.jsonl": "LeagueStrategyTable",
}
V4_CONFORMANCE_REQUIREMENTS = (
    {
        "requirement_id": "learned_exploiter_anti_meta",
        "title": "Production learned exploiter and defense anti-meta loop",
        "required_schema_versions": (
            "learned_exploiter_validation_report.v1",
            "league_selfplay_health_report.v1",
        ),
    },
    {
        "requirement_id": "full_ablation_feedback",
        "title": "Full v4 ablation and production feedback validation",
        "required_schema_versions": (
            "v4_ablation_experiment_plan.v1",
            "ablation_suite_report.v1",
        ),
    },
    {
        "requirement_id": "real_calibration_holdout",
        "title": "Real feature calibration holdout on real-meta feedback",
        "required_schema_versions": (
            "real_calibration_ingestion_summary.v1",
            "real_calibration_sample_build_summary.v1",
            "real_calibration_validation_report.v1",
        ),
    },
    {
        "requirement_id": "active_real_query_dispatch",
        "title": "Scaled active real-query dispatch and feedback",
        "required_schema_versions": (
            "active_query_feedback_report.v1",
            "active_real_query_dispatch_validation_report.v1",
        ),
    },
    {
        "requirement_id": "mask_belief_validation",
        "title": "Learned mask/risk explanation and belief real-distribution validation",
        "required_schema_versions": (
            "mask_explanation_validation_report.v1",
            "belief_real_distribution_validation_report.v1",
        ),
    },
)


def build_active_query_feedback_report(
    round_dir: Path | str,
    *,
    min_matched_query_coverage: float = 1.0,
    max_oracle_result_error_rate: float = 0.0,
    min_real_query_count: int = 0,
) -> dict[str, Any]:
    root = Path(round_dir)
    active_queries = _read_jsonl(root / "active_queries.jsonl")
    oracle_pairs = _read_jsonl(root / "oracle_pairs.jsonl")
    results = _read_jsonl(root / "oracle_results.jsonl")
    pair_by_key = {
        (str(row.get("attack_id")), str(row.get("defense_id"))): row
        for row in oracle_pairs
        if row.get("attack_id") is not None and row.get("defense_id") is not None
    }
    query_feedback: list[dict[str, Any]] = []
    unmatched = 0
    for query in active_queries:
        key = (str(query.get("attack_id")), str(query.get("defense_id")))
        pair = pair_by_key.get(key)
        attack_success = None
        round_win_rates = None
        if pair is None:
            unmatched += 1
        else:
            attack_success = float(pair.get("attack_success", 0.0) or 0.0)
            round_win_rates = pair.get("round_win_rates")
        query_feedback.append(
            {
                "queue": str(query.get("queue", "unknown")),
                "query_id": str(query.get("query_id", "")),
                "query_type": str(query.get("query_type", "unknown")),
                "attack_id": query.get("attack_id"),
                "defense_id": query.get("defense_id"),
                "attack_role": query.get("attack_role"),
                "defense_role": query.get("defense_role"),
                "score": float(query.get("score", 0.0) or 0.0),
                "info_gain": float(query.get("info_gain", 0.0) or 0.0),
                "decision_impact": float(query.get("decision_impact", 0.0) or 0.0),
                "novelty": float(query.get("novelty", 0.0) or 0.0),
                "underdog_potential": float(query.get("underdog_potential", 0.0) or 0.0),
                "cost": float(query.get("cost", 0.0) or 0.0),
                "attack_success": attack_success,
                "round_win_rates": round_win_rates,
            }
        )
    matched_queries = len(active_queries) - unmatched
    real_rows = [row for row in query_feedback if row.get("queue") == "real"]
    sim_rows = [row for row in query_feedback if row.get("queue") == "sim"]
    matched_real = sum(1 for row in real_rows if _is_number(row.get("attack_success")))
    matched_sim = sum(1 for row in sim_rows if _is_number(row.get("attack_success")))
    oracle_errors = sum(1 for row in results if row.get("status") != "completed" and row.get("status") != "cached")
    report = {
        "schema_version": "active_query_feedback_report.v1",
        "module": "ActiveQueryFeedbackReport",
        "round_dir": str(root),
        "queries": len(active_queries),
        "matched_queries": matched_queries,
        "unmatched_queries": unmatched,
        "matched_query_coverage": 1.0 if not active_queries else matched_queries / len(active_queries),
        "real_queries": len(real_rows),
        "matched_real_queries": matched_real,
        "real_query_feedback_coverage": 1.0 if not real_rows else matched_real / len(real_rows),
        "sim_queries": len(sim_rows),
        "matched_sim_queries": matched_sim,
        "sim_query_feedback_coverage": 1.0 if not sim_rows else matched_sim / len(sim_rows),
        "oracle_pairs": len(oracle_pairs),
        "oracle_result_rows": len(results),
        "oracle_result_errors": oracle_errors,
        "oracle_result_error_rate": 0.0 if not results else oracle_errors / len(results),
        "queues": _active_query_queue_stats(query_feedback),
        "query_feedback": query_feedback,
    }
    report["red_line_violations"] = _active_query_feedback_red_lines(
        report,
        min_matched_query_coverage=min_matched_query_coverage,
        max_oracle_result_error_rate=max_oracle_result_error_rate,
        min_real_query_count=min_real_query_count,
    )
    return report


def build_active_real_query_dispatch_validation_report(
    validation_report_paths: Sequence[Path | str],
    *,
    min_reports: int = 1,
    min_dispatched_pairs: int = 1,
    min_completion_rate: float = 1.0,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in validation_report_paths)
    rows = [_active_real_dispatch_report_row(path) for path in paths]
    readable_rows = [row for row in rows if row["readable"]]
    queued_queries = sum(int(row.get("queued_queries", 0) or 0) for row in readable_rows)
    dispatchable_queries = sum(int(row.get("dispatchable_queries", 0) or 0) for row in readable_rows)
    skipped_queries = sum(int(row.get("skipped_queries", 0) or 0) for row in readable_rows)
    dispatched_pairs = sum(int(row.get("dispatched_pairs", 0) or 0) for row in readable_rows)
    oracle_requests = sum(int(row.get("oracle_requests", 0) or 0) for row in readable_rows)
    oracle_result_errors = sum(int(row.get("oracle_result_errors", 0) or 0) for row in readable_rows)
    attack_teacher_rows = sum(int(row.get("attack_teacher_rows", 0) or 0) for row in readable_rows)
    defense_teacher_rows = sum(int(row.get("defense_teacher_rows", 0) or 0) for row in readable_rows)
    completed_requests = max(0, oracle_requests - oracle_result_errors)
    report = {
        "schema_version": "active_real_query_dispatch_validation_report.v1",
        "module": "ActiveRealQueryDispatchValidationReport",
        "validation_report_paths": [str(path) for path in paths],
        "reports": len(rows),
        "readable_reports": len(readable_rows),
        "read_error_reports": len(rows) - len(readable_rows),
        "queued_queries": queued_queries,
        "dispatchable_queries": dispatchable_queries,
        "skipped_queries": skipped_queries,
        "skipped_query_reasons": _merge_count_mappings(row.get("skipped_query_reasons") for row in readable_rows),
        "dispatched_pairs": dispatched_pairs,
        "oracle_requests": oracle_requests,
        "oracle_result_errors": oracle_result_errors,
        "completion_rate": 1.0 if oracle_requests <= 0 else completed_requests / oracle_requests,
        "attack_teacher_rows": attack_teacher_rows,
        "defense_teacher_rows": defense_teacher_rows,
        "teacher_feedback_complete_reports": sum(1 for row in readable_rows if row.get("teacher_feedback_complete") is True),
        "real_query_queue_validated_reports": sum(1 for row in readable_rows if row.get("real_query_queue_validated") is True),
        "report_rows": rows,
    }
    red_lines = _active_real_dispatch_validation_red_lines(
        report,
        min_reports=min_reports,
        min_dispatched_pairs=min_dispatched_pairs,
        min_completion_rate=min_completion_rate,
    )
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


def build_mask_explanation_validation_report(
    round_dir: Path | str,
    *,
    min_hidden_explanation_coverage: float = 0.95,
) -> dict[str, Any]:
    root = Path(round_dir)
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    rows: list[dict[str, Any]] = []
    risk_report_rows = 0
    mask_explanation_rows = 0
    learned_mask_score_rows = 0
    counter_attack_risk_rows = 0
    defenses_with_no_hidden_slots = 0
    total_hidden_slots = 0
    explained_hidden_slots = 0
    learned_scores: list[float] = []
    hidden_counts: list[float] = []
    for row in defenses:
        defense_id = str(row.get("defense_id", ""))
        risk_report = row.get("defense_risk_report")
        if not isinstance(risk_report, Mapping):
            rows.append(
                {
                    "defense_id": defense_id,
                    "has_risk_report": False,
                    "has_mask_explanation": False,
                    "hidden_count": 0,
                    "explained_hidden_slots": 0,
                    "has_counter_attack_risk_report": False,
                }
            )
            continue
        risk_report_rows += 1
        hidden_count = _int_from_first_number(risk_report.get("hidden_count"), row.get("hidden_count"))
        total_hidden_slots += hidden_count
        hidden_counts.append(float(hidden_count))
        if hidden_count <= 0:
            defenses_with_no_hidden_slots += 1
        mask_explanation = risk_report.get("mask_explanation")
        has_mask_explanation = isinstance(mask_explanation, Mapping)
        if has_mask_explanation:
            mask_explanation_rows += 1
        hidden_explanations = (
            mask_explanation.get("hidden_slot_explanations", ())
            if isinstance(mask_explanation, Mapping)
            else ()
        )
        hidden_explanation_count = len(hidden_explanations) if isinstance(hidden_explanations, Sequence) and not isinstance(hidden_explanations, (str, bytes)) else 0
        explained_hidden_slots += min(hidden_count, hidden_explanation_count)
        learned_mask_score = risk_report.get("learned_mask_score")
        if _is_number(learned_mask_score):
            learned_mask_score_rows += 1
            learned_scores.append(float(learned_mask_score))
        has_counter_risk = isinstance(risk_report.get("counter_attack_risk_report"), Mapping)
        if has_counter_risk:
            counter_attack_risk_rows += 1
        rows.append(
            {
                "defense_id": defense_id,
                "has_risk_report": True,
                "has_mask_explanation": has_mask_explanation,
                "hidden_count": hidden_count,
                "explained_hidden_slots": min(hidden_count, hidden_explanation_count),
                "raw_hidden_slot_explanations": hidden_explanation_count,
                "has_learned_mask_score": _is_number(learned_mask_score),
                "learned_mask_score": float(learned_mask_score) if _is_number(learned_mask_score) else None,
                "has_counter_attack_risk_report": has_counter_risk,
            }
        )
    coverage = 0.0 if total_hidden_slots <= 0 else explained_hidden_slots / total_hidden_slots
    report = {
        "schema_version": "mask_explanation_validation_report.v1",
        "module": "MaskExplanationValidationReport",
        "round_dir": str(root),
        "defenses": len(defenses),
        "risk_report_rows": risk_report_rows,
        "mask_explanation_rows": mask_explanation_rows,
        "learned_mask_score_rows": learned_mask_score_rows,
        "counter_attack_risk_rows": counter_attack_risk_rows,
        "defenses_with_no_hidden_slots": defenses_with_no_hidden_slots,
        "total_hidden_slots": total_hidden_slots,
        "explained_hidden_slots": explained_hidden_slots,
        "hidden_explanation_coverage": coverage,
        "mean_hidden_count": _mean(hidden_counts),
        "mean_learned_mask_score": _mean(learned_scores),
        "defense_rows": rows,
    }
    report["red_line_violations"] = _mask_explanation_red_lines(
        report,
        min_hidden_explanation_coverage=min_hidden_explanation_coverage,
    )
    return report


def build_belief_real_distribution_validation_report(
    round_dir: Path | str,
    *,
    min_real_coverage: float = 0.50,
    min_mean_real_records: float = 1.0,
    min_mean_real_similarity: float = 0.25,
    max_oracle_alignment_mae: float = 0.35,
) -> dict[str, Any]:
    root = Path(round_dir)
    candidates = _read_jsonl(root / "candidates.jsonl")
    oracle_pairs = _read_jsonl(root / "oracle_pairs.jsonl")
    pair_by_key = {
        (str(row.get("attack_id")), str(row.get("defense_id"))): row
        for row in oracle_pairs
        if row.get("attack_id") is not None and row.get("defense_id") is not None
    }
    rows: list[dict[str, Any]] = []
    real_records: list[float] = []
    real_similarities: list[float] = []
    real_match_results: list[float] = []
    weight_entropies: list[float] = []
    alignment_errors: list[float] = []
    stats_rows = 0
    real_rows = 0
    exact_rows = 0
    similar_rows = 0
    for row in candidates:
        stats = _belief_domain_stats_mapping(row.get("belief_domain_stats"))
        if stats:
            stats_rows += 1
        attack_id = str(row.get("attack_id", ""))
        defense_id = str(row.get("defense_id", ""))
        real_record_count = _float_or_zero(stats.get("real_record_count"))
        exact_count = _float_or_zero(stats.get("real_exact_record_count"))
        similar_count = _float_or_zero(stats.get("real_similar_record_count"))
        real_similarity = _float_or_zero(stats.get("real_similarity_mean"))
        real_match_result = _float_or_zero(stats.get("real_match_result_mean"))
        weight_entropy = _float_or_zero(stats.get("weight_entropy_normalized"))
        candidate_count = _float_or_zero(stats.get("candidate_count"))
        defense_pool_record_count = _float_or_zero(stats.get("defense_pool_record_count"))
        ranker_applied = _float_or_zero(stats.get("ranker_applied"))
        if real_record_count > 0.0:
            real_rows += 1
            real_records.append(real_record_count)
            real_similarities.append(real_similarity)
            real_match_results.append(real_match_result)
        if exact_count > 0.0:
            exact_rows += 1
        if similar_count > 0.0:
            similar_rows += 1
        if "weight_entropy_normalized" in stats:
            weight_entropies.append(weight_entropy)
        pair = pair_by_key.get((attack_id, defense_id))
        attack_success = None
        alignment_abs_error = None
        if pair is not None and _is_number(pair.get("attack_success")) and "real_match_result_mean" in stats:
            attack_success = float(pair["attack_success"])
            alignment_abs_error = round(abs(attack_success - real_match_result), 12)
            if real_record_count > 0.0:
                alignment_errors.append(alignment_abs_error)
        rows.append(
            {
                "attack_id": attack_id,
                "defense_id": defense_id,
                "has_belief_domain_stats": bool(stats),
                "real_record_count": real_record_count,
                "real_exact_record_count": exact_count,
                "real_similar_record_count": similar_count,
                "real_similarity_mean": real_similarity,
                "real_match_result_mean": real_match_result if "real_match_result_mean" in stats else None,
                "defense_pool_record_count": defense_pool_record_count,
                "ranker_applied": ranker_applied,
                "candidate_count": candidate_count,
                "weight_entropy_normalized": weight_entropy if "weight_entropy_normalized" in stats else None,
                "oracle_attack_success": attack_success,
                "alignment_abs_error": alignment_abs_error,
            }
        )
    report = {
        "schema_version": "belief_real_distribution_validation_report.v1",
        "module": "BeliefRealDistributionValidationReport",
        "round_dir": str(root),
        "candidates": len(candidates),
        "belief_domain_stats_rows": stats_rows,
        "real_distribution_rows": real_rows,
        "real_distribution_coverage": 0.0 if not candidates else real_rows / len(candidates),
        "exact_real_rows": exact_rows,
        "similar_real_rows": similar_rows,
        "mean_real_record_count": round(_mean(real_records), 12),
        "mean_real_similarity": round(_mean(real_similarities), 12),
        "mean_real_match_result": round(_mean(real_match_results), 12),
        "mean_weight_entropy_normalized": round(_mean(weight_entropies), 12),
        "oracle_alignment_rows": len(alignment_errors),
        "oracle_alignment_mae": round(_mean(alignment_errors), 12),
        "candidate_rows": rows,
    }
    report["red_line_violations"] = _belief_real_distribution_red_lines(
        report,
        min_real_coverage=min_real_coverage,
        min_mean_real_records=min_mean_real_records,
        min_mean_real_similarity=min_mean_real_similarity,
        max_oracle_alignment_mae=max_oracle_alignment_mae,
    )
    return report


def build_data_engineering_validation_report(
    round_dirs: Sequence[Path | str],
    *,
    min_metadata_coverage: float = 1.0,
    min_core_table_coverage: float = 1.0,
    min_artifact_hash_coverage: float = 1.0,
) -> dict[str, Any]:
    roots = tuple(Path(path) for path in round_dirs)
    round_reports = [_data_engineering_round_report(root) for root in roots]
    rounds = len(round_reports)
    metadata_files = sum(1 for row in round_reports if row["has_run_metadata"])
    artifact_refs = sum(int(row["artifact_refs"]) for row in round_reports)
    artifact_verified = sum(int(row["artifact_verified"]) for row in round_reports)
    artifact_missing = sum(int(row["artifact_missing_count"]) for row in round_reports)
    artifact_mismatch = sum(int(row["artifact_hash_mismatch_count"]) for row in round_reports)
    expected_tables = rounds * len(CORE_TABLE_FILES)
    table_files_found = sum(int(row["core_table_files_found"]) for row in round_reports)
    table_empty = sum(int(row["core_table_empty_count"]) for row in round_reports)
    table_schema_mismatch = sum(int(row["core_table_schema_mismatch_count"]) for row in round_reports)
    report = {
        "schema_version": "data_engineering_validation_report.v1",
        "module": "DataEngineeringValidationReport",
        "round_dirs": [str(path) for path in roots],
        "rounds": rounds,
        "metadata_files": metadata_files,
        "metadata_coverage": 1.0 if rounds <= 0 else metadata_files / rounds,
        "artifact_refs": artifact_refs,
        "artifact_verified": artifact_verified,
        "artifact_missing_count": artifact_missing,
        "artifact_hash_mismatch_count": artifact_mismatch,
        "artifact_hash_coverage": 1.0 if artifact_refs <= 0 else artifact_verified / artifact_refs,
        "core_table_files_expected": expected_tables,
        "core_table_files_found": table_files_found,
        "core_table_coverage": 1.0 if expected_tables <= 0 else table_files_found / expected_tables,
        "core_table_empty_count": table_empty,
        "core_table_schema_mismatch_count": table_schema_mismatch,
        "round_reports": round_reports,
    }
    report["red_line_violations"] = _data_engineering_red_lines(
        report,
        min_metadata_coverage=min_metadata_coverage,
        min_core_table_coverage=min_core_table_coverage,
        min_artifact_hash_coverage=min_artifact_hash_coverage,
    )
    return report


def build_underdog_residual_validation_report(
    round_dirs: Sequence[Path | str],
    *,
    min_attack_residual_coverage: float = 0.95,
    min_defense_residual_coverage: float = 0.95,
    min_mean_attack_residual_bonus: float = 0.0,
    min_mean_defense_residual_bonus: float = 0.0,
) -> dict[str, Any]:
    roots = tuple(Path(path) for path in round_dirs)
    round_reports = [_underdog_residual_round_report(root) for root in roots]
    attack_rows = sum(int(row["attack_rows"]) for row in round_reports)
    attack_underdog_rows = sum(int(row["attack_underdog_rows"]) for row in round_reports)
    attack_residual_rows = sum(int(row["attack_residual_rows"]) for row in round_reports)
    defense_rows = sum(int(row["defense_rows"]) for row in round_reports)
    defense_underdog_rows = sum(int(row["defense_underdog_rows"]) for row in round_reports)
    defense_residual_rows = sum(int(row["defense_residual_rows"]) for row in round_reports)
    attack_gaps = _flatten_round_values(round_reports, "attack_underdog_gaps")
    attack_bonuses = _flatten_round_values(round_reports, "attack_residual_bonuses")
    attack_objectives = _flatten_round_values(round_reports, "attack_objective_scores")
    defense_gaps = _flatten_round_values(round_reports, "defense_underdog_gaps")
    defense_bonuses = _flatten_round_values(round_reports, "defense_residual_bonuses")
    defense_objectives = _flatten_round_values(round_reports, "defense_objective_scores")
    report = {
        "schema_version": "underdog_residual_validation_report.v1",
        "module": "UnderdogResidualValidationReport",
        "round_dirs": [str(path) for path in roots],
        "rounds": len(round_reports),
        "attack_rows": attack_rows,
        "attack_underdog_rows": attack_underdog_rows,
        "attack_residual_rows": attack_residual_rows,
        "attack_residual_coverage": 0.0 if attack_underdog_rows <= 0 else attack_residual_rows / attack_underdog_rows,
        "mean_attack_underdog_gap": round(_mean(attack_gaps), 12),
        "mean_attack_residual_bonus": round(_mean(attack_bonuses), 12),
        "mean_attack_objective_score": round(_mean(attack_objectives), 12),
        "defense_rows": defense_rows,
        "defense_underdog_rows": defense_underdog_rows,
        "defense_residual_rows": defense_residual_rows,
        "defense_residual_coverage": 0.0 if defense_underdog_rows <= 0 else defense_residual_rows / defense_underdog_rows,
        "mean_defense_underdog_gap": round(_mean(defense_gaps), 12),
        "mean_defense_residual_bonus": round(_mean(defense_bonuses), 12),
        "mean_defense_objective_score": round(_mean(defense_objectives), 12),
        "round_reports": round_reports,
    }
    report["red_line_violations"] = _underdog_residual_red_lines(
        report,
        min_attack_residual_coverage=min_attack_residual_coverage,
        min_defense_residual_coverage=min_defense_residual_coverage,
        min_mean_attack_residual_bonus=min_mean_attack_residual_bonus,
        min_mean_defense_residual_bonus=min_mean_defense_residual_bonus,
    )
    return report


def build_exploiter_effectiveness_report(
    teacher_jsonl_paths: Sequence[Path | str] = (),
    *,
    training_root: Path | str | None = None,
    min_target_coverage: float = 0.95,
    min_positive_residual_rate: float = 0.50,
    min_trend_delta: float | None = None,
) -> dict[str, Any]:
    paths = _collect_attack_teacher_paths(teacher_jsonl_paths, training_root=training_root)
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            payload = dict(row)
            payload.setdefault("teacher_jsonl", str(path))
            payload.setdefault("round_id", _infer_teacher_round_id(payload, path))
            rows.append(payload)
    target_rows = [row for row in rows if _has_exploiter_target_feedback(row)]
    role_names = sorted({str(row.get("attack_role", "unknown")) for row in rows} | {"main", "exploiter", "underdog"})
    role_stats = {role: _exploiter_role_stats([row for row in rows if str(row.get("attack_role", "unknown")) == role]) for role in role_names}
    anti_meta_rows = [row for row in rows if str(row.get("attack_role", "unknown")) in ANTI_META_ROLES]
    main_rows = [row for row in rows if str(row.get("attack_role", "unknown")) == "main"]
    anti_meta_stats = _exploiter_role_stats(anti_meta_rows)
    main_stats = _exploiter_role_stats(main_rows)
    anti_meta = {
        **anti_meta_stats,
        "roles": sorted(ANTI_META_ROLES),
        "residual_lift_vs_main": round(float(anti_meta_stats["mean_residual"]) - float(main_stats["mean_residual"]), 12),
    }
    round_stats = _exploiter_round_stats(rows)
    trend = _exploiter_round_trend(round_stats)
    report = {
        "schema_version": "exploiter_effectiveness_report.v1",
        "module": "ExploiterEffectivenessReport",
        "teacher_jsonl_paths": tuple(str(path) for path in paths),
        "teacher_rows": len(rows),
        "target_feedback_rows": len(target_rows),
        "target_feedback_coverage": 0.0 if not rows else len(target_rows) / len(rows),
        "role_stats": role_stats,
        "anti_meta": anti_meta,
        "round_stats": round_stats,
        "trend": trend,
    }
    report["red_line_violations"] = _exploiter_red_lines(
        report,
        min_target_coverage=min_target_coverage,
        min_positive_residual_rate=min_positive_residual_rate,
        min_trend_delta=min_trend_delta,
    )
    return report


def build_defense_anti_meta_effectiveness_report(
    teacher_jsonl_paths: Sequence[Path | str] = (),
    *,
    training_root: Path | str | None = None,
    min_feedback_coverage: float = 0.95,
    min_positive_residual_rate: float = 0.50,
    min_mean_residual: float = 0.0,
    min_trend_delta: float | None = None,
) -> dict[str, Any]:
    paths = _collect_defense_teacher_paths(teacher_jsonl_paths, training_root=training_root)
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            payload = dict(row)
            payload.setdefault("teacher_jsonl", str(path))
            payload.setdefault("round_id", _infer_teacher_round_id(payload, path))
            rows.append(payload)
    feedback_rows = [row for row in rows if _has_defense_anti_meta_feedback(row)]
    role_names = sorted({str(row.get("defense_role", "unknown")) for row in rows} | {"main", "anti_meta", "underdog"})
    role_stats = {role: _defense_anti_meta_role_stats([row for row in rows if str(row.get("defense_role", "unknown")) == role]) for role in role_names}
    anti_meta_stats = _defense_anti_meta_role_stats(feedback_rows)
    round_stats = _defense_anti_meta_round_stats(rows)
    trend = _defense_anti_meta_round_trend(round_stats)
    report = {
        "schema_version": "defense_anti_meta_effectiveness_report.v1",
        "module": "DefenseAntiMetaEffectivenessReport",
        "teacher_jsonl_paths": tuple(str(path) for path in paths),
        "teacher_rows": len(rows),
        "anti_meta_feedback_rows": len(feedback_rows),
        "anti_meta_feedback_coverage": 0.0 if not rows else len(feedback_rows) / len(rows),
        "role_stats": role_stats,
        "anti_meta": {
            **anti_meta_stats,
            "roles": role_names,
        },
        "round_stats": round_stats,
        "trend": trend,
    }
    report["red_line_violations"] = _defense_anti_meta_red_lines(
        report,
        min_feedback_coverage=min_feedback_coverage,
        min_positive_residual_rate=min_positive_residual_rate,
        min_mean_residual=min_mean_residual,
        min_trend_delta=min_trend_delta,
    )
    return report


def build_learned_exploiter_validation_report(
    *,
    selfplay_root: Path | str,
    training_root: Path | str | None = None,
    min_rounds: int = 2,
    min_oracle_requests: int = 1,
    require_latest_checkpoints: bool = True,
    min_attack_target_coverage: float = 0.95,
    min_attack_positive_residual_rate: float = 0.50,
    min_attack_trend_delta: float | None = None,
    min_defense_feedback_coverage: float = 0.95,
    min_defense_positive_residual_rate: float = 0.50,
    min_defense_mean_residual: float = 0.0,
    min_defense_trend_delta: float | None = None,
) -> dict[str, Any]:
    root = Path(selfplay_root)
    state = _read_json(root / "orchestrator_state.json")
    effective_training_root = Path(
        training_root
        if training_root is not None
        else state.get("training_dir", root / "training")
    )
    round_records = state.get("rounds")
    if not isinstance(round_records, Sequence) or isinstance(round_records, (str, bytes)):
        round_records = ()
    normalized_rounds = [row for row in round_records if isinstance(row, Mapping)]
    oracle_requests = sum(int(row.get("oracle_requests", 0) or 0) for row in normalized_rounds)
    latest_attack_checkpoint = state.get("latest_attack_proposal_checkpoint")
    latest_defense_checkpoint = state.get("latest_defense_proposal_checkpoint")
    exploiter_report = build_exploiter_effectiveness_report(
        training_root=effective_training_root,
        min_target_coverage=min_attack_target_coverage,
        min_positive_residual_rate=min_attack_positive_residual_rate,
        min_trend_delta=min_attack_trend_delta,
    )
    defense_report = build_defense_anti_meta_effectiveness_report(
        training_root=effective_training_root,
        min_feedback_coverage=min_defense_feedback_coverage,
        min_positive_residual_rate=min_defense_positive_residual_rate,
        min_mean_residual=min_defense_mean_residual,
        min_trend_delta=min_defense_trend_delta,
    )
    report = {
        "schema_version": "learned_exploiter_validation_report.v1",
        "module": "LearnedExploiterValidationReport",
        "selfplay_root": str(root),
        "training_root": str(effective_training_root),
        "rounds": len(normalized_rounds),
        "round_ids": [str(row.get("round_id", "")) for row in normalized_rounds],
        "oracle_requests": oracle_requests,
        "latest_attack_proposal_checkpoint": latest_attack_checkpoint,
        "latest_defense_proposal_checkpoint": latest_defense_checkpoint,
        "exploiter_report": exploiter_report,
        "defense_anti_meta_report": defense_report,
    }
    red_lines = _learned_exploiter_validation_red_lines(
        report,
        min_rounds=min_rounds,
        min_oracle_requests=min_oracle_requests,
        require_latest_checkpoints=require_latest_checkpoints,
    )
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


def build_league_selfplay_health_report(
    round_dirs: Sequence[Path | str],
    *,
    min_attack_pool: int = 1,
    min_defense_pool: int = 1,
    min_total_clusters: int = 2,
    min_payoff_density: float = 0.0,
    required_attack_roles: Sequence[str] = ("main", "exploiter", "underdog"),
    required_defense_roles: Sequence[str] = ("main", "exploiter", "underdog"),
    min_active_pool_fraction: float = 0.0,
    min_new_attack_strength_delta: float | None = None,
    min_new_defense_strength_delta: float | None = None,
) -> dict[str, Any]:
    roots = tuple(Path(path) for path in round_dirs)
    round_reports = [_league_selfplay_round_report(root) for root in roots]
    latest = _latest_league_selfplay_round(round_reports)
    required_attack = tuple(str(role) for role in required_attack_roles)
    required_defense = tuple(str(role) for role in required_defense_roles)
    attack_role_coverage = _role_coverage(latest.get("active_attack_role_counts", {}), required_attack)
    defense_role_coverage = _role_coverage(latest.get("active_defense_role_counts", {}), required_defense)
    report = {
        "schema_version": "league_selfplay_health_report.v1",
        "module": "LeagueSelfPlayHealthReport",
        "round_dirs": [str(path) for path in roots],
        "rounds": len(round_reports),
        "latest_round_dir": str(latest.get("round_dir", "")),
        "latest_round_id": str(latest.get("round_id", "")),
        "latest_iteration": int(latest.get("iteration", 0) or 0),
        "attack_pool": int(latest.get("attack_pool", 0) or 0),
        "defense_pool": int(latest.get("defense_pool", 0) or 0),
        "active_attack_pool": int(latest.get("active_attack_pool", 0) or 0),
        "active_defense_pool": int(latest.get("active_defense_pool", 0) or 0),
        "retired_attack_pool": int(latest.get("retired_attack_pool", 0) or 0),
        "retired_defense_pool": int(latest.get("retired_defense_pool", 0) or 0),
        "active_pool_fraction": float(latest.get("active_pool_fraction", 0.0) or 0.0),
        "historical_attack_pool": int(latest.get("historical_attack_pool", 0) or 0),
        "historical_defense_pool": int(latest.get("historical_defense_pool", 0) or 0),
        "attack_clusters": int(latest.get("attack_clusters", 0) or 0),
        "defense_clusters": int(latest.get("defense_clusters", 0) or 0),
        "total_clusters": int(latest.get("total_clusters", 0) or 0),
        "attack_role_counts": latest.get("attack_role_counts", {}),
        "defense_role_counts": latest.get("defense_role_counts", {}),
        "active_attack_role_counts": latest.get("active_attack_role_counts", {}),
        "active_defense_role_counts": latest.get("active_defense_role_counts", {}),
        "required_attack_roles": list(required_attack),
        "required_defense_roles": list(required_defense),
        "attack_role_coverage": attack_role_coverage,
        "defense_role_coverage": defense_role_coverage,
        "payoff_entries": int(latest.get("payoff_entries", 0) or 0),
        "payoff_games": int(latest.get("payoff_games", 0) or 0),
        "payoff_density": float(latest.get("payoff_density", 0.0) or 0.0),
        "new_attack_pool": int(latest.get("new_attack_pool", 0) or 0),
        "new_defense_pool": int(latest.get("new_defense_pool", 0) or 0),
        "new_attack_strength_delta": float(latest.get("new_attack_strength_delta", 0.0) or 0.0),
        "new_defense_strength_delta": float(latest.get("new_defense_strength_delta", 0.0) or 0.0),
        "best_attack_success": float(latest.get("best_attack_success", 0.0) or 0.0),
        "worst_defense_break_rate": float(latest.get("worst_defense_break_rate", 0.0) or 0.0),
        "oracle_requests": sum(int(row.get("oracle_requests", 0) or 0) for row in round_reports),
        "round_reports": round_reports,
    }
    red_lines = _league_selfplay_health_red_lines(
        report,
        min_attack_pool=min_attack_pool,
        min_defense_pool=min_defense_pool,
        min_total_clusters=min_total_clusters,
        min_payoff_density=min_payoff_density,
        min_active_pool_fraction=min_active_pool_fraction,
        min_new_attack_strength_delta=min_new_attack_strength_delta,
        min_new_defense_strength_delta=min_new_defense_strength_delta,
    )
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


def build_production_readiness_report(
    report_paths: Sequence[str | Path],
    *,
    required_schema_versions: Sequence[str] = (),
    min_clean_report_rate: float = 1.0,
    require_production_ready: bool = True,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in report_paths)
    rows = [_production_readiness_report_row(path) for path in paths]
    readable_rows = [row for row in rows if row["readable"]]
    red_rows = [row for row in readable_rows if row["red_line_violations"]]
    production_ready_checked = [row for row in readable_rows if row["production_ready"] is not None]
    production_ready_false = [row for row in production_ready_checked if row["production_ready"] is False]
    clean_rows = [
        row
        for row in readable_rows
        if not row["red_line_violations"]
        and not (require_production_ready and row["production_ready"] is False)
    ]
    schema_counts: dict[str, int] = {}
    for row in readable_rows:
        for schema_version in row.get("schema_versions", ()):
            if schema_version:
                schema_counts[str(schema_version)] = schema_counts.get(str(schema_version), 0) + 1
    missing_required = [
        str(schema_version)
        for schema_version in required_schema_versions
        if schema_counts.get(str(schema_version), 0) <= 0
    ]
    clean_rate = 0.0 if not rows else len(clean_rows) / len(rows)
    report = {
        "schema_version": "production_readiness_report.v1",
        "module": "ProductionReadinessReport",
        "report_paths": [str(path) for path in paths],
        "reports": len(rows),
        "readable_reports": len(readable_rows),
        "read_error_reports": len(rows) - len(readable_rows),
        "clean_reports": len(clean_rows),
        "red_line_reports": len(red_rows),
        "clean_report_rate": clean_rate,
        "production_ready_checked_reports": len(production_ready_checked),
        "production_ready_reports": sum(1 for row in production_ready_checked if row["production_ready"] is True),
        "production_ready_false_reports": len(production_ready_false),
        "schema_counts": dict(sorted(schema_counts.items())),
        "required_schema_versions": [str(value) for value in required_schema_versions],
        "missing_required_schema_versions": missing_required,
        "report_rows": rows,
    }
    red_lines = _production_readiness_red_lines(
        report,
        min_clean_report_rate=min_clean_report_rate,
        require_production_ready=require_production_ready,
    )
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


def build_v4_conformance_validation_report(report_paths: Sequence[str | Path]) -> dict[str, Any]:
    paths = tuple(Path(path) for path in report_paths)
    rows = [_production_readiness_report_row(path) for path in paths]
    readable_rows = [row for row in rows if row["readable"]]
    schema_counts: dict[str, int] = {}
    for row in readable_rows:
        for schema_version in row.get("schema_versions", ()):
            if schema_version:
                key = str(schema_version)
                schema_counts[key] = schema_counts.get(key, 0) + 1
    requirements = [
        _v4_conformance_requirement_row(requirement, readable_rows)
        for requirement in V4_CONFORMANCE_REQUIREMENTS
    ]
    passed = sum(1 for row in requirements if row["status"] == "pass")
    report = {
        "schema_version": "v4_conformance_validation_report.v1",
        "module": "V4ConformanceValidationReport",
        "report_paths": [str(path) for path in paths],
        "reports": len(rows),
        "readable_reports": len(readable_rows),
        "read_error_reports": len(rows) - len(readable_rows),
        "schema_counts": dict(sorted(schema_counts.items())),
        "requirements_total": len(requirements),
        "passed_requirements": passed,
        "failed_requirements": len(requirements) - passed,
        "requirements": requirements,
        "report_rows": rows,
    }
    red_lines = _v4_conformance_red_lines(report)
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


def build_attack_oracle_failure_validation_report(
    *,
    oracle_output_paths: Sequence[Path | str] = (),
    round_dirs: Sequence[Path | str] = (),
    min_failure_annotation_coverage: float = 1.0,
    min_failure_diagnostic_coverage: float = 1.0,
) -> dict[str, Any]:
    output_paths = tuple(Path(path) for path in oracle_output_paths)
    round_paths = tuple(Path(path) for path in round_dirs)
    rows: list[dict[str, Any]] = []
    candidate_rows = 0
    candidate_risk_rows = 0
    candidate_missing_risk_rows = 0
    for path in output_paths:
        payload = _read_json(path)
        if not payload:
            continue
        rows.append(
            _attack_failure_validation_row(
                payload.get("risk_report"),
                payload.get("diagnostics"),
                source_type="attack_oracle_output",
                source_path=str(path),
                attack_id=payload.get("attack_id"),
                defense_id=payload.get("defense_id"),
            )
        )
    for round_dir in round_paths:
        for candidate in _read_jsonl(round_dir / "candidates.jsonl"):
            candidate_rows += 1
            risk_report = candidate.get("attack_risk_report")
            if isinstance(risk_report, Mapping):
                candidate_risk_rows += 1
                rows.append(
                    _attack_failure_validation_row(
                        risk_report,
                        candidate.get("diagnostics"),
                        source_type="round_candidate",
                        source_path=str(round_dir / "candidates.jsonl"),
                        attack_id=candidate.get("attack_id"),
                        defense_id=candidate.get("defense_id"),
                    )
                )
            else:
                candidate_missing_risk_rows += 1
    failure_rows = [row for row in rows if row["has_failure"]]
    annotated_failure_rows = [row for row in failure_rows if row["has_failure_code"] and row["has_failure_stage"]]
    diagnostic_failure_rows = [row for row in failure_rows if row["has_matching_diagnostic"]]
    normal_rows = [row for row in rows if not row["has_failure"]]
    report = {
        "schema_version": "attack_oracle_failure_validation_report.v1",
        "module": "AttackOracleFailureValidationReport",
        "oracle_output_paths": tuple(str(path) for path in output_paths),
        "round_dirs": tuple(str(path) for path in round_paths),
        "oracle_outputs": len(output_paths),
        "candidate_rows": candidate_rows,
        "candidate_risk_report_rows": candidate_risk_rows,
        "candidate_missing_risk_report_rows": candidate_missing_risk_rows,
        "checked_rows": len(rows),
        "failure_rows": len(failure_rows),
        "annotated_failure_rows": len(annotated_failure_rows),
        "diagnostic_failure_rows": len(diagnostic_failure_rows),
        "failure_annotation_coverage": 1.0 if not failure_rows else len(annotated_failure_rows) / len(failure_rows),
        "failure_diagnostic_coverage": 1.0 if not failure_rows else len(diagnostic_failure_rows) / len(failure_rows),
        "normal_risk_report_rows": len(normal_rows),
        "failure_stage_counts": _counts(row["failure_stage"] for row in failure_rows if row["failure_stage"]),
        "failure_code_counts": _counts(row["failure_code"] for row in failure_rows if row["failure_code"]),
        "validation_rows": rows,
    }
    report["red_line_violations"] = _attack_oracle_failure_red_lines(
        report,
        min_failure_annotation_coverage=min_failure_annotation_coverage,
        min_failure_diagnostic_coverage=min_failure_diagnostic_coverage,
    )
    return report


def build_league_round_report(round_dir: Path, *, date: str) -> DailyTrainingReport:
    root = Path(round_dir)
    summary = _read_json(root / "summary.json")
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    results = _read_jsonl(root / "oracle_results.jsonl")
    active_queries = _read_jsonl(root / "active_queries.jsonl")
    league = _read_json(root / "league_state.json")
    failure_cases = [
        {
            "request_id": row.get("request_id"),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for row in results
        if row.get("status") != "completed"
    ]
    clusters = {
        row.get("diversity_cluster", "default")
        for row in tuple(league.get("attack_pool", ())) + tuple(league.get("defense_pool", ()))
    }
    underdog_candidates = [row for row in candidates if row.get("attack_role") == "underdog"]
    ambiguity_values = [float(row.get("ambiguity_score", 0.0) or 0.0) for row in defenses]
    attack_expected = _nested_float_values(candidates, "attack_risk_report", "expected_match_win")
    attack_worst_case = _nested_float_values(candidates, "attack_risk_report", "worst_case_match_win")
    attack_backup_counts = _nested_float_values(candidates, "attack_risk_report", "backup_attack_count")
    attack_belief_case_counts = _nested_float_values(candidates, "attack_risk_report", "belief_case_count")
    underdog_expected = _nested_float_values(underdog_candidates, "attack_risk_report", "expected_match_win")
    underdog_gaps = _nested_float_values(underdog_candidates, "attack_risk_report", "underdog_gap")
    underdog_residual_bonuses = _nested_float_values(underdog_candidates, "attack_risk_report", "underdog_residual_bonus")
    defense_estimated_break = _nested_float_values(defenses, "defense_risk_report", "estimated_break_rate")
    defense_estimated_survival = _nested_float_values(defenses, "defense_risk_report", "estimated_survival_rate")
    defense_underdog_gaps = _nested_float_values(defenses, "defense_risk_report", "underdog_defense_gap")
    defense_underdog_residual_bonuses = _nested_float_values(defenses, "defense_risk_report", "underdog_residual_bonus")
    defense_hidden_counts = [
        *_nested_float_values(defenses, "defense_risk_report", "hidden_count"),
        *[float(row.get("hidden_count")) for row in defenses if _is_number(row.get("hidden_count"))],
    ]
    defense_backup_counts = _nested_float_values(defenses, "defense_risk_report", "backup_defense_count")
    attack_oracle = {
        "top1": float(summary.get("best_attack_success", 0.0) or 0.0),
        "top5_hit": 1.0 if float(summary.get("best_attack_success", 0.0) or 0.0) > 0.0 else 0.0,
    }
    if attack_expected:
        attack_oracle["belief_expected_mean"] = _mean(attack_expected)
        attack_oracle["top5_mean"] = _mean(sorted(attack_expected, reverse=True)[:5])
    if attack_worst_case:
        attack_oracle["belief_worst_case_mean"] = _mean(attack_worst_case)
        attack_oracle["belief_worst_case_min"] = min(attack_worst_case)
    if attack_backup_counts:
        attack_oracle["backup_attack_mean"] = _mean(attack_backup_counts)
    if attack_belief_case_counts:
        attack_oracle["belief_case_mean"] = _mean(attack_belief_case_counts)
    if underdog_gaps:
        attack_oracle["underdog_gap_mean"] = _mean(underdog_gaps)
        attack_oracle["underdog_gap_max"] = max(underdog_gaps)
    if underdog_residual_bonuses:
        attack_oracle["underdog_residual_bonus_mean"] = _mean(underdog_residual_bonuses)
    defense_oracle = {
        "attack_success": float(summary.get("worst_defense_break_rate", 0.0) or 0.0),
        "ambiguity": sum(ambiguity_values) / len(ambiguity_values) if ambiguity_values else 0.0,
    }
    if defense_estimated_break:
        defense_oracle["estimated_break_rate"] = _mean(defense_estimated_break)
        defense_oracle["best_response_break_rate"] = max(defense_estimated_break)
    if defense_estimated_survival:
        defense_oracle["estimated_survival_rate"] = _mean(defense_estimated_survival)
    if defense_hidden_counts:
        defense_oracle["hidden_count_mean"] = _mean(defense_hidden_counts)
    if defense_backup_counts:
        defense_oracle["backup_defense_mean"] = _mean(defense_backup_counts)
    if defense_underdog_gaps:
        defense_oracle["underdog_gap_mean"] = _mean(defense_underdog_gaps)
        defense_oracle["underdog_gap_max"] = max(defense_underdog_gaps)
    if defense_underdog_residual_bonuses:
        defense_oracle["underdog_residual_bonus_mean"] = _mean(defense_underdog_residual_bonuses)
    return DailyTrainingReport(
        date=date,
        sim_games=int(summary.get("oracle_requests", len(results))),
        real_matches=int(summary.get("real_matches", 0) or 0),
        single_model={"brier": 0.0, "ece": 0.0, "auc": 0.0},
        attack_oracle=attack_oracle,
        defense_oracle=defense_oracle,
        league={
            "attack_pool": len(league.get("attack_pool", ())),
            "defense_pool": len(league.get("defense_pool", ())),
            "clusters": len(clusters),
            "active_query_count": len(active_queries),
        },
        underdog={
            "samples": len(underdog_candidates),
            "success_rate": _mean(underdog_expected) if underdog_expected else 0.0,
        },
        active_queries=active_queries,
        failure_cases=failure_cases,
    )


def _active_query_queue_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    queues = sorted({str(row.get("queue", "unknown")) for row in rows})
    stats: dict[str, dict[str, Any]] = {}
    for queue in queues:
        queue_rows = [row for row in rows if str(row.get("queue", "unknown")) == queue]
        success_values = [float(row["attack_success"]) for row in queue_rows if _is_number(row.get("attack_success"))]
        stats[queue] = {
            "queries": len(queue_rows),
            "matched_queries": len(success_values),
            "underdog_queries": sum(1 for row in queue_rows if row.get("query_type") == "underdog"),
            "mean_score": _mean([float(row.get("score", 0.0) or 0.0) for row in queue_rows]),
            "mean_info_gain": _mean([float(row.get("info_gain", 0.0) or 0.0) for row in queue_rows]),
            "mean_decision_impact": _mean([float(row.get("decision_impact", 0.0) or 0.0) for row in queue_rows]),
            "mean_novelty": _mean([float(row.get("novelty", 0.0) or 0.0) for row in queue_rows]),
            "mean_underdog_potential": _mean([float(row.get("underdog_potential", 0.0) or 0.0) for row in queue_rows]),
            "mean_cost": _mean([float(row.get("cost", 0.0) or 0.0) for row in queue_rows]),
            "mean_attack_success": _mean(success_values),
        }
    return stats


def _active_query_feedback_red_lines(
    report: Mapping[str, Any],
    *,
    min_matched_query_coverage: float,
    max_oracle_result_error_rate: float,
    min_real_query_count: int,
) -> list[str]:
    violations: list[str] = []
    queries = int(report.get("queries", 0) or 0)
    if queries <= 0:
        violations.append("no_active_queries")
    if float(report.get("matched_query_coverage", 0.0) or 0.0) < float(min_matched_query_coverage):
        violations.append("active_query_feedback_coverage_low")
    if int(report.get("real_queries", 0) or 0) < int(min_real_query_count):
        violations.append("real_query_count_low")
    if int(report.get("real_queries", 0) or 0) > int(report.get("matched_real_queries", 0) or 0):
        violations.append("real_query_feedback_missing")
    if float(report.get("oracle_result_error_rate", 0.0) or 0.0) > float(max_oracle_result_error_rate):
        violations.append("oracle_result_errors")
    return violations


def _active_real_dispatch_report_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "readable": False,
            "read_error": "missing_report",
            "schema_version": None,
            "module": None,
        }
    try:
        payload = _read_json(path)
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "read_error": str(exc),
            "schema_version": None,
            "module": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "readable": True,
        "read_error": None,
        "schema_version": payload.get("schema_version"),
        "module": payload.get("module"),
        "round_dir": payload.get("round_dir"),
        "out_dir": payload.get("out_dir"),
        "queued_queries": int(payload.get("queued_queries", 0) or 0),
        "dispatchable_queries": int(payload.get("dispatchable_queries", 0) or 0),
        "skipped_queries": int(payload.get("skipped_queries", 0) or 0),
        "skipped_query_reasons": payload.get("skipped_query_reasons", {}),
        "dispatched_pairs": int(payload.get("dispatched_pairs", 0) or 0),
        "oracle_requests": int(payload.get("oracle_requests", 0) or 0),
        "oracle_result_errors": int(payload.get("oracle_result_errors", 0) or 0),
        "completion_rate": float(payload.get("completion_rate", 0.0) or 0.0),
        "attack_teacher_rows": int(payload.get("attack_teacher_rows", 0) or 0),
        "defense_teacher_rows": int(payload.get("defense_teacher_rows", 0) or 0),
        "teacher_feedback_complete": bool(payload.get("teacher_feedback_complete")),
        "real_query_queue_validated": bool(payload.get("real_query_queue_validated")),
    }


def _active_real_dispatch_validation_red_lines(
    report: Mapping[str, Any],
    *,
    min_reports: int,
    min_dispatched_pairs: int,
    min_completion_rate: float,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("readable_reports", 0) or 0) < int(min_reports):
        violations.append("active_real_dispatch_reports_low")
    if int(report.get("read_error_reports", 0) or 0) > 0:
        violations.append("dispatch_validation_report_read_error")
    if int(report.get("queued_queries", 0) or 0) <= 0:
        violations.append("no_active_real_queries")
    if int(report.get("dispatched_pairs", 0) or 0) < int(min_dispatched_pairs):
        violations.append("active_real_dispatched_pairs_low")
    if int(report.get("oracle_result_errors", 0) or 0) > 0:
        violations.append("active_real_oracle_result_errors")
    if float(report.get("completion_rate", 0.0) or 0.0) < float(min_completion_rate):
        violations.append("active_real_completion_rate_low")
    if int(report.get("teacher_feedback_complete_reports", 0) or 0) < int(report.get("readable_reports", 0) or 0):
        violations.append("active_real_teacher_feedback_incomplete")
    if int(report.get("real_query_queue_validated_reports", 0) or 0) < int(report.get("readable_reports", 0) or 0):
        violations.append("active_real_queue_not_validated")
    return _dedupe_strings(violations)


def _merge_count_mappings(values: Sequence[object]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, count in value.items():
            merged[str(key)] = merged.get(str(key), 0) + int(count or 0)
    return dict(sorted(merged.items()))


def red_line_violations(report: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if report.get("failure_cases"):
        violations.append("oracle_result_errors")
    single_model = report.get("single_model", {})
    if isinstance(single_model, Mapping) and float(single_model.get("ece", 0.0) or 0.0) > 0.20:
        violations.append("single_model_ece_high")
    defense = report.get("defense_oracle", {})
    if isinstance(defense, Mapping) and float(defense.get("attack_success", 0.0) or 0.0) >= 0.95:
        violations.append("defense_oracle_break_rate_high")
    attack = report.get("attack_oracle", {})
    if isinstance(attack, Mapping):
        top1 = float(attack.get("top1", 0.0) or 0.0)
        top5_mean = attack.get("top5_mean")
        if top5_mean is not None and top1 + 0.05 < float(top5_mean or 0.0):
            violations.append("attack_top1_below_top5_mean")
    league = report.get("league", {})
    if isinstance(league, Mapping) and int(league.get("clusters", 0) or 0) <= 1 and int(league.get("attack_pool", 0) or 0) > 1:
        violations.append("league_cluster_collapse")
    return violations


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain JSON object lines")
        rows.append(payload)
    return rows


def _data_engineering_round_report(root: Path) -> dict[str, Any]:
    metadata_path = root / "run_metadata.json"
    metadata_payload: dict[str, Any] = {}
    metadata_error: str | None = None
    if metadata_path.exists():
        try:
            metadata_payload = _read_json(metadata_path)
        except Exception as exc:  # pragma: no cover - exercised through report red-line behavior.
            metadata_error = str(exc)
    artifact_rows = _validate_run_metadata_artifacts(root, metadata_payload)
    table_rows, table_files_found, table_empty_count, table_schema_mismatch_count, missing_tables = _validate_core_tables(root)
    return {
        "round_dir": str(root),
        "has_run_metadata": metadata_path.exists() and metadata_error is None,
        "metadata_error": metadata_error,
        "run_id": str(metadata_payload.get("run_id", "")) if metadata_payload else "",
        "metadata_schema_version": str(metadata_payload.get("schema_version", "")) if metadata_payload else "",
        "artifact_refs": len(artifact_rows),
        "artifact_verified": sum(1 for row in artifact_rows if row["verified"]),
        "artifact_missing_count": sum(1 for row in artifact_rows if row["missing"]),
        "artifact_hash_mismatch_count": sum(1 for row in artifact_rows if row["hash_mismatch"]),
        "artifact_rows": artifact_rows,
        "core_table_files_expected": len(CORE_TABLE_FILES),
        "core_table_files_found": table_files_found,
        "core_table_missing": missing_tables,
        "core_table_empty_count": table_empty_count,
        "core_table_schema_mismatch_count": table_schema_mismatch_count,
        "table_rows": table_rows,
    }


def _validate_run_metadata_artifacts(root: Path, metadata_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifact_refs = list(metadata_payload.get("input_artifacts", ()) or ()) + list(
        metadata_payload.get("output_artifacts", ()) or ()
    )
    for ref in artifact_refs:
        if not isinstance(ref, Mapping):
            continue
        recorded_path = Path(str(ref.get("path", "")))
        actual_path = _resolve_artifact_path(root, recorded_path)
        missing = not actual_path.exists()
        sha256 = _sha256_file(actual_path) if not missing else ""
        expected_sha = str(ref.get("sha256", ""))
        hash_mismatch = (not missing) and bool(expected_sha) and sha256 != expected_sha
        rows.append(
            {
                "path": str(ref.get("path", "")),
                "kind": str(ref.get("kind", "")),
                "role": str(ref.get("role", "")),
                "resolved_path": str(actual_path),
                "missing": missing,
                "hash_mismatch": hash_mismatch,
                "verified": (not missing) and (not hash_mismatch),
            }
        )
    return rows


def _resolve_artifact_path(root: Path, recorded_path: Path) -> Path:
    if recorded_path.is_absolute() or recorded_path.exists():
        return recorded_path
    candidate = root / recorded_path.name
    if candidate.exists():
        return candidate
    return recorded_path


def _validate_core_tables(root: Path) -> tuple[dict[str, int], int, int, int, list[str]]:
    table_dir = root / "tables"
    table_rows: dict[str, int] = {table: 0 for table in CORE_TABLE_FILES.values()}
    files_found = 0
    empty_count = 0
    schema_mismatch_count = 0
    missing_tables: list[str] = []
    for filename, table_name in CORE_TABLE_FILES.items():
        path = table_dir / filename
        if not path.exists():
            missing_tables.append(table_name)
            continue
        files_found += 1
        rows = _read_jsonl(path)
        table_rows[table_name] = len(rows)
        if not rows:
            empty_count += 1
        if any(row.get("schema_version") != "core_tables.v1" or row.get("table") != table_name for row in rows):
            schema_mismatch_count += 1
    return table_rows, files_found, empty_count, schema_mismatch_count, missing_tables


def _underdog_residual_round_report(root: Path) -> dict[str, Any]:
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    attack_underdogs = [row for row in candidates if _is_attack_underdog_residual_row(row)]
    attack_residual_rows = [
        row
        for row in attack_underdogs
        if _has_risk_numbers(row, "attack_risk_report", "underdog_gap", "underdog_residual_bonus")
    ]
    attack_gaps = _risk_float_values(attack_residual_rows, "attack_risk_report", "underdog_gap")
    attack_bonuses = _risk_float_values(attack_residual_rows, "attack_risk_report", "underdog_residual_bonus")
    attack_objectives = [
        value
        for value in (
            _first_risk_float(row, "attack_risk_report", "objective_score", "expected_match_win")
            for row in attack_residual_rows
        )
        if value is not None
    ]
    defense_underdogs = [row for row in defenses if _is_defense_underdog_residual_row(row)]
    defense_residual_rows = [
        row
        for row in defense_underdogs
        if _has_risk_numbers(row, "defense_risk_report", "underdog_defense_gap", "underdog_residual_bonus")
    ]
    defense_gaps = _risk_float_values(defense_residual_rows, "defense_risk_report", "underdog_defense_gap")
    defense_bonuses = _risk_float_values(defense_residual_rows, "defense_risk_report", "underdog_residual_bonus")
    defense_objectives = [
        value
        for value in (_defense_objective_score(row) for row in defense_residual_rows)
        if value is not None
    ]
    return {
        "round_dir": str(root),
        "attack_rows": len(candidates),
        "attack_underdog_rows": len(attack_underdogs),
        "attack_residual_rows": len(attack_residual_rows),
        "attack_residual_coverage": 0.0 if not attack_underdogs else len(attack_residual_rows) / len(attack_underdogs),
        "attack_underdog_gaps": attack_gaps,
        "attack_residual_bonuses": attack_bonuses,
        "attack_objective_scores": attack_objectives,
        "mean_attack_underdog_gap": round(_mean(attack_gaps), 12),
        "mean_attack_residual_bonus": round(_mean(attack_bonuses), 12),
        "mean_attack_objective_score": round(_mean(attack_objectives), 12),
        "defense_rows": len(defenses),
        "defense_underdog_rows": len(defense_underdogs),
        "defense_residual_rows": len(defense_residual_rows),
        "defense_residual_coverage": 0.0 if not defense_underdogs else len(defense_residual_rows) / len(defense_underdogs),
        "defense_underdog_gaps": defense_gaps,
        "defense_residual_bonuses": defense_bonuses,
        "defense_objective_scores": defense_objectives,
        "mean_defense_underdog_gap": round(_mean(defense_gaps), 12),
        "mean_defense_residual_bonus": round(_mean(defense_bonuses), 12),
        "mean_defense_objective_score": round(_mean(defense_objectives), 12),
    }


def _is_attack_underdog_residual_row(row: Mapping[str, Any]) -> bool:
    if str(row.get("attack_role", "")).lower() == "underdog":
        return True
    risk = row.get("attack_risk_report")
    return isinstance(risk, Mapping) and (
        _positive_float(risk.get("underdog_gap")) or _positive_float(risk.get("underdog_residual_bonus"))
    )


def _is_defense_underdog_residual_row(row: Mapping[str, Any]) -> bool:
    role = str(row.get("defense_role", row.get("role", ""))).lower()
    if role == "underdog":
        return True
    risk = row.get("defense_risk_report")
    return isinstance(risk, Mapping) and (
        _positive_float(risk.get("underdog_defense_gap")) or _positive_float(risk.get("underdog_residual_bonus"))
    )


def _has_risk_numbers(row: Mapping[str, Any], parent: str, *keys: str) -> bool:
    risk = row.get(parent)
    if not isinstance(risk, Mapping):
        return False
    return all(_coerce_float(risk.get(key)) is not None for key in keys)


def _risk_float_values(rows: Sequence[Mapping[str, Any]], parent: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _first_risk_float(row, parent, key)
        if value is not None:
            values.append(value)
    return values


def _first_risk_float(row: Mapping[str, Any], parent: str, *keys: str) -> float | None:
    risk = row.get(parent)
    if not isinstance(risk, Mapping):
        return None
    for key in keys:
        value = _coerce_float(risk.get(key))
        if value is not None:
            return value
    return None


def _defense_objective_score(row: Mapping[str, Any]) -> float | None:
    value = _first_risk_float(row, "defense_risk_report", "objective_score", "estimated_survival_rate", "survival_rate")
    if value is not None:
        return value
    break_rate = _first_risk_float(row, "defense_risk_report", "estimated_break_rate", "break_rate")
    if break_rate is None:
        return None
    return 1.0 - break_rate


def _positive_float(value: object) -> bool:
    number = _coerce_float(value)
    return number is not None and number > 0.0


def _coerce_float(value: object) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _flatten_round_values(round_reports: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for report in round_reports:
        raw = report.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        values.extend(float(value) for value in raw if _coerce_float(value) is not None)
    return values


def _underdog_residual_red_lines(
    report: Mapping[str, Any],
    *,
    min_attack_residual_coverage: float,
    min_defense_residual_coverage: float,
    min_mean_attack_residual_bonus: float,
    min_mean_defense_residual_bonus: float,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("rounds", 0) or 0) <= 0:
        violations.append("no_round_dirs")
    if int(report.get("attack_underdog_rows", 0) or 0) <= 0:
        violations.append("attack_underdog_rows_missing")
    if int(report.get("defense_underdog_rows", 0) or 0) <= 0:
        violations.append("defense_underdog_rows_missing")
    if float(report.get("attack_residual_coverage", 0.0) or 0.0) < float(min_attack_residual_coverage):
        violations.append("attack_residual_coverage_low")
    if float(report.get("defense_residual_coverage", 0.0) or 0.0) < float(min_defense_residual_coverage):
        violations.append("defense_residual_coverage_low")
    attack_bonus = float(report.get("mean_attack_residual_bonus", 0.0) or 0.0)
    defense_bonus = float(report.get("mean_defense_residual_bonus", 0.0) or 0.0)
    if attack_bonus <= 0.0 or attack_bonus < float(min_mean_attack_residual_bonus):
        violations.append("attack_residual_bonus_non_positive")
    if defense_bonus <= 0.0 or defense_bonus < float(min_mean_defense_residual_bonus):
        violations.append("defense_residual_bonus_non_positive")
    return violations


def _collect_attack_teacher_paths(
    teacher_jsonl_paths: Sequence[Path | str],
    *,
    training_root: Path | str | None,
) -> tuple[Path, ...]:
    paths = [Path(path) for path in teacher_jsonl_paths]
    if training_root is not None:
        root = Path(training_root)
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(sorted(root.glob("round_*/attack_teacher.jsonl")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _collect_defense_teacher_paths(
    teacher_jsonl_paths: Sequence[Path | str],
    *,
    training_root: Path | str | None,
) -> tuple[Path, ...]:
    paths = [Path(path) for path in teacher_jsonl_paths]
    if training_root is not None:
        root = Path(training_root)
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(sorted(root.glob("round_*/defense_teacher.jsonl")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _has_exploiter_target_feedback(row: Mapping[str, Any]) -> bool:
    return (
        row.get("target_defense_id") is not None
        and row.get("target_defense_hash") is not None
        and _is_number(row.get("target_baseline_break_rate"))
        and _is_number(row.get("exploiter_residual_target"))
    )


def _infer_teacher_round_id(row: Mapping[str, Any], path: Path) -> str:
    round_id = row.get("round_id")
    if round_id is not None:
        return str(round_id)
    group_id = row.get("teacher_group_id")
    if isinstance(group_id, str) and group_id:
        prefix = group_id.split(":", 1)[0]
        if prefix:
            return prefix
    for part in reversed(path.parts):
        if part.startswith("round_"):
            return part
    return "unknown"


def _exploiter_role_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = [_float_or_zero(row.get("exploiter_residual_target")) for row in rows if _is_number(row.get("exploiter_residual_target"))]
    attack_success = [_float_or_zero(row.get("attack_success")) for row in rows if _is_number(row.get("attack_success"))]
    baseline_break = [_float_or_zero(row.get("target_baseline_break_rate")) for row in rows if _is_number(row.get("target_baseline_break_rate"))]
    role_weights = [_float_or_zero(row.get("role_weight")) for row in rows if _is_number(row.get("role_weight"))]
    target_rows = [row for row in rows if _has_exploiter_target_feedback(row)]
    return {
        "samples": len(rows),
        "target_feedback_rows": len(target_rows),
        "target_feedback_coverage": 0.0 if not rows else len(target_rows) / len(rows),
        "mean_attack_success": round(_mean(attack_success), 12),
        "mean_baseline_break_rate": round(_mean(baseline_break), 12),
        "mean_residual": round(_mean(residuals), 12),
        "positive_residual_rate": _positive_rate(residuals),
        "mean_role_weight": round(_mean(role_weights), 12),
        "source_counts": _source_counts(rows),
    }


def _exploiter_round_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    round_ids = sorted({str(row.get("round_id", "unknown")) for row in rows})
    for round_id in round_ids:
        round_rows = [row for row in rows if str(row.get("round_id", "unknown")) == round_id]
        role_names = sorted({str(row.get("attack_role", "unknown")) for row in round_rows} | {"main", "exploiter", "underdog"})
        anti_meta_rows = [row for row in round_rows if str(row.get("attack_role", "unknown")) in ANTI_META_ROLES]
        main_rows = [row for row in round_rows if str(row.get("attack_role", "unknown")) == "main"]
        anti_meta_stats = _exploiter_role_stats(anti_meta_rows)
        main_stats = _exploiter_role_stats(main_rows)
        stats[round_id] = {
            "teacher_rows": len(round_rows),
            "target_feedback_rows": sum(1 for row in round_rows if _has_exploiter_target_feedback(row)),
            "role_stats": {
                role: _exploiter_role_stats([row for row in round_rows if str(row.get("attack_role", "unknown")) == role])
                for role in role_names
            },
            "anti_meta": {
                **anti_meta_stats,
                "roles": sorted(ANTI_META_ROLES),
                "residual_lift_vs_main": round(float(anti_meta_stats["mean_residual"]) - float(main_stats["mean_residual"]), 12),
            },
        }
    return stats


def _exploiter_round_trend(round_stats: Mapping[str, Any]) -> dict[str, Any]:
    round_ids: list[str] = []
    residuals: list[float] = []
    positive_rates: list[float] = []
    for round_id in sorted(round_stats):
        payload = round_stats.get(round_id)
        if not isinstance(payload, Mapping):
            continue
        anti_meta = payload.get("anti_meta")
        if not isinstance(anti_meta, Mapping) or int(anti_meta.get("samples", 0) or 0) <= 0:
            continue
        round_ids.append(str(round_id))
        residuals.append(round(float(anti_meta.get("mean_residual", 0.0) or 0.0), 12))
        positive_rates.append(round(float(anti_meta.get("positive_residual_rate", 0.0) or 0.0), 12))
    first = residuals[0] if residuals else 0.0
    last = residuals[-1] if residuals else 0.0
    delta = round(last - first, 12) if len(residuals) >= 2 else 0.0
    slope = round(delta / max(len(residuals) - 1, 1), 12) if len(residuals) >= 2 else 0.0
    return {
        "rounds": round_ids,
        "anti_meta_mean_residuals": residuals,
        "anti_meta_positive_residual_rates": positive_rates,
        "first_anti_meta_mean_residual": first,
        "last_anti_meta_mean_residual": last,
        "delta_anti_meta_mean_residual": delta,
        "slope_per_round": slope,
        "improving": delta > 0.0,
    }


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _positive_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value > 0.0) / len(values)


def _exploiter_red_lines(
    report: Mapping[str, Any],
    *,
    min_target_coverage: float,
    min_positive_residual_rate: float,
    min_trend_delta: float | None,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("teacher_rows", 0) or 0) <= 0:
        violations.append("no_attack_teacher_rows")
    if float(report.get("target_feedback_coverage", 0.0) or 0.0) < float(min_target_coverage):
        violations.append("target_feedback_coverage_low")
    anti_meta = report.get("anti_meta")
    if not isinstance(anti_meta, Mapping) or int(anti_meta.get("samples", 0) or 0) <= 0:
        violations.append("no_anti_meta_samples")
        return violations
    if float(anti_meta.get("mean_residual", 0.0) or 0.0) <= 0.0:
        violations.append("anti_meta_residual_non_positive")
    if float(anti_meta.get("positive_residual_rate", 0.0) or 0.0) < float(min_positive_residual_rate):
        violations.append("anti_meta_positive_rate_low")
    if min_trend_delta is not None:
        trend = report.get("trend")
        if isinstance(trend, Mapping) and len(trend.get("rounds", ()) or ()) >= 2:
            delta = float(trend.get("delta_anti_meta_mean_residual", 0.0) or 0.0)
            if delta <= float(min_trend_delta):
                violations.append("anti_meta_residual_trend_non_positive")
    return violations


def _has_defense_anti_meta_feedback(row: Mapping[str, Any]) -> bool:
    return (
        _is_number(row.get("survival_rate"))
        and _is_number(row.get("meta_attack_success"))
        and _is_number(row.get("anti_meta_residual_target"))
    )


def _defense_anti_meta_role_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feedback_rows = [row for row in rows if _has_defense_anti_meta_feedback(row)]
    residuals = [_float_or_zero(row.get("anti_meta_residual_target")) for row in feedback_rows]
    survival_rates = [_float_or_zero(row.get("survival_rate")) for row in feedback_rows]
    meta_attack_success = [_float_or_zero(row.get("meta_attack_success")) for row in feedback_rows]
    role_weights = [_float_or_zero(row.get("role_weight")) for row in rows if _is_number(row.get("role_weight"))]
    return {
        "samples": len(rows),
        "anti_meta_feedback_rows": len(feedback_rows),
        "anti_meta_feedback_coverage": 0.0 if not rows else len(feedback_rows) / len(rows),
        "mean_survival_rate": round(_mean(survival_rates), 12),
        "mean_meta_attack_success": round(_mean(meta_attack_success), 12),
        "mean_residual": round(_mean(residuals), 12),
        "mean_survival_lift": round(_mean(residuals), 12),
        "positive_residual_rate": _positive_rate(residuals),
        "mean_role_weight": round(_mean(role_weights), 12),
        "source_counts": _source_counts(rows),
    }


def _defense_anti_meta_round_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    round_ids = sorted({str(row.get("round_id", "unknown")) for row in rows})
    for round_id in round_ids:
        round_rows = [row for row in rows if str(row.get("round_id", "unknown")) == round_id]
        role_names = sorted({str(row.get("defense_role", "unknown")) for row in round_rows} | {"main", "anti_meta", "underdog"})
        feedback_rows = [row for row in round_rows if _has_defense_anti_meta_feedback(row)]
        stats[round_id] = {
            "teacher_rows": len(round_rows),
            "anti_meta_feedback_rows": len(feedback_rows),
            "role_stats": {
                role: _defense_anti_meta_role_stats([row for row in round_rows if str(row.get("defense_role", "unknown")) == role])
                for role in role_names
            },
            "anti_meta": {
                **_defense_anti_meta_role_stats(feedback_rows),
                "roles": role_names,
            },
        }
    return stats


def _defense_anti_meta_round_trend(round_stats: Mapping[str, Any]) -> dict[str, Any]:
    round_ids: list[str] = []
    residuals: list[float] = []
    positive_rates: list[float] = []
    for round_id in sorted(round_stats):
        payload = round_stats.get(round_id)
        if not isinstance(payload, Mapping):
            continue
        anti_meta = payload.get("anti_meta")
        if not isinstance(anti_meta, Mapping) or int(anti_meta.get("anti_meta_feedback_rows", 0) or 0) <= 0:
            continue
        round_ids.append(str(round_id))
        residuals.append(round(float(anti_meta.get("mean_residual", 0.0) or 0.0), 12))
        positive_rates.append(round(float(anti_meta.get("positive_residual_rate", 0.0) or 0.0), 12))
    first = residuals[0] if residuals else 0.0
    last = residuals[-1] if residuals else 0.0
    delta = round(last - first, 12) if len(residuals) >= 2 else 0.0
    slope = round(delta / max(len(residuals) - 1, 1), 12) if len(residuals) >= 2 else 0.0
    return {
        "rounds": round_ids,
        "anti_meta_mean_residuals": residuals,
        "anti_meta_positive_residual_rates": positive_rates,
        "first_anti_meta_mean_residual": first,
        "last_anti_meta_mean_residual": last,
        "delta_anti_meta_mean_residual": delta,
        "slope_per_round": slope,
        "improving": delta > 0.0,
    }


def _defense_anti_meta_red_lines(
    report: Mapping[str, Any],
    *,
    min_feedback_coverage: float,
    min_positive_residual_rate: float,
    min_mean_residual: float,
    min_trend_delta: float | None,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("teacher_rows", 0) or 0) <= 0:
        violations.append("no_defense_teacher_rows")
    if float(report.get("anti_meta_feedback_coverage", 0.0) or 0.0) < float(min_feedback_coverage):
        violations.append("anti_meta_feedback_coverage_low")
    anti_meta = report.get("anti_meta")
    if not isinstance(anti_meta, Mapping) or int(anti_meta.get("anti_meta_feedback_rows", 0) or 0) <= 0:
        violations.append("no_defense_anti_meta_feedback")
        return violations
    if float(anti_meta.get("mean_residual", 0.0) or 0.0) <= float(min_mean_residual):
        violations.append("defense_anti_meta_residual_non_positive")
    if float(anti_meta.get("positive_residual_rate", 0.0) or 0.0) < float(min_positive_residual_rate):
        violations.append("defense_anti_meta_positive_rate_low")
    if min_trend_delta is not None:
        trend = report.get("trend")
        if isinstance(trend, Mapping) and len(trend.get("rounds", ()) or ()) >= 2:
            delta = float(trend.get("delta_anti_meta_mean_residual", 0.0) or 0.0)
            if delta <= float(min_trend_delta):
                violations.append("defense_anti_meta_residual_trend_non_positive")
    return violations


def _learned_exploiter_validation_red_lines(
    report: Mapping[str, Any],
    *,
    min_rounds: int,
    min_oracle_requests: int,
    require_latest_checkpoints: bool,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("rounds", 0) or 0) < int(min_rounds):
        violations.append("validation_rounds_low")
    if int(report.get("oracle_requests", 0) or 0) < int(min_oracle_requests):
        violations.append("oracle_requests_low")
    if require_latest_checkpoints:
        if not report.get("latest_attack_proposal_checkpoint"):
            violations.append("missing_latest_attack_checkpoint")
        if not report.get("latest_defense_proposal_checkpoint"):
            violations.append("missing_latest_defense_checkpoint")
    exploiter_report = report.get("exploiter_report")
    if isinstance(exploiter_report, Mapping):
        violations.extend(f"attack_{value}" for value in exploiter_report.get("red_line_violations", ()) or ())
    else:
        violations.append("attack_report_missing")
    defense_report = report.get("defense_anti_meta_report")
    if isinstance(defense_report, Mapping):
        violations.extend(f"defense_{value}" for value in defense_report.get("red_line_violations", ()) or ())
    else:
        violations.append("defense_report_missing")
    return _dedupe_strings(violations)


def _league_selfplay_round_report(root: Path) -> dict[str, Any]:
    summary = _read_json(root / "summary.json")
    league = _read_json(root / "league_state.json")
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    oracle_pairs = _read_jsonl(root / "oracle_pairs.jsonl")
    attacks = _mapping_rows(league.get("attack_pool"))
    defense_pool = _mapping_rows(league.get("defense_pool"))
    payoffs = _mapping_rows(league.get("payoffs")) or oracle_pairs
    iteration = int(league.get("iteration", summary.get("iteration", 0)) or 0)
    active_attacks = [row for row in attacks if _strategy_active(row)]
    active_defenses = [row for row in defense_pool if _strategy_active(row)]
    active_total = len(active_attacks) + len(active_defenses)
    pool_total = len(attacks) + len(defense_pool)
    attack_clusters = _strategy_clusters(attacks)
    defense_clusters = _strategy_clusters(defense_pool)
    payoff_denominator = len(attacks) * len(defense_pool)
    new_attacks = _created_in_iteration(attacks, iteration)
    new_defenses = _created_in_iteration(defense_pool, iteration)
    prior_attacks = _created_before_iteration(attacks, iteration)
    prior_defenses = _created_before_iteration(defense_pool, iteration)
    return {
        "round_dir": str(root),
        "round_id": str(summary.get("round_id", root.name)),
        "has_league_state": bool(league),
        "iteration": iteration,
        "oracle_requests": int(summary.get("oracle_requests", 0) or 0),
        "best_attack_success": float(summary.get("best_attack_success", 0.0) or 0.0),
        "worst_defense_break_rate": float(summary.get("worst_defense_break_rate", 0.0) or 0.0),
        "attack_pool": len(attacks),
        "defense_pool": len(defense_pool),
        "active_attack_pool": len(active_attacks),
        "active_defense_pool": len(active_defenses),
        "retired_attack_pool": len(attacks) - len(active_attacks),
        "retired_defense_pool": len(defense_pool) - len(active_defenses),
        "active_pool_fraction": 0.0 if pool_total <= 0 else active_total / pool_total,
        "historical_attack_pool": len(_historical_strategies(attacks, iteration)),
        "historical_defense_pool": len(_historical_strategies(defense_pool, iteration)),
        "attack_clusters": len(attack_clusters),
        "defense_clusters": len(defense_clusters),
        "total_clusters": len(attack_clusters | defense_clusters),
        "attack_role_counts": _counts(str(row.get("role", "unknown")) for row in attacks),
        "defense_role_counts": _counts(str(row.get("role", "unknown")) for row in defense_pool),
        "active_attack_role_counts": _counts(str(row.get("role", "unknown")) for row in active_attacks),
        "active_defense_role_counts": _counts(str(row.get("role", "unknown")) for row in active_defenses),
        "candidate_attack_role_counts": _counts(str(row.get("attack_role", "unknown")) for row in candidates),
        "candidate_defense_role_counts": _counts(str(row.get("defense_role", row.get("role", "unknown"))) for row in candidates),
        "scored_defense_role_counts": _counts(str(row.get("defense_role", row.get("role", "unknown"))) for row in defenses),
        "payoff_entries": len(payoffs),
        "payoff_games": sum(int(row.get("games", 0) or 0) for row in payoffs),
        "payoff_density": 0.0 if payoff_denominator <= 0 else len(payoffs) / payoff_denominator,
        "mean_payoff_attack_success": round(_mean([float(row.get("attack_success", 0.0) or 0.0) for row in payoffs]), 12),
        "new_attack_pool": len(new_attacks),
        "new_defense_pool": len(new_defenses),
        "new_attack_mean_strength": round(_strategy_mean_strength(new_attacks), 12),
        "new_defense_mean_strength": round(_strategy_mean_strength(new_defenses), 12),
        "prior_attack_mean_strength": round(_strategy_mean_strength(prior_attacks), 12),
        "prior_defense_mean_strength": round(_strategy_mean_strength(prior_defenses), 12),
        "new_attack_strength_delta": round(_new_strategy_strength_delta(new_attacks, prior_attacks), 12),
        "new_defense_strength_delta": round(_new_strategy_strength_delta(new_defenses, prior_defenses), 12),
    }


def _latest_league_selfplay_round(round_reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not round_reports:
        return {}
    indexed = tuple(enumerate(round_reports))
    return max(indexed, key=lambda item: (int(item[1].get("iteration", 0) or 0), item[0]))[1]


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _strategy_active(row: Mapping[str, Any]) -> bool:
    return row.get("active", True) is not False


def _strategy_clusters(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    clusters: set[str] = set()
    for index, row in enumerate(rows):
        cluster = row.get("diversity_cluster")
        if cluster is None:
            cluster = row.get("plan_hash", f"missing-{index}")
        clusters.add(str(cluster))
    return clusters


def _strategy_created_iteration(row: Mapping[str, Any], fallback: int) -> int:
    value = row.get("created_iteration", fallback)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return fallback


def _created_in_iteration(rows: Sequence[Mapping[str, Any]], iteration: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if _strategy_created_iteration(row, iteration) == iteration]


def _created_before_iteration(rows: Sequence[Mapping[str, Any]], iteration: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if _strategy_created_iteration(row, iteration) < iteration]


def _historical_strategies(rows: Sequence[Mapping[str, Any]], iteration: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("role", "")) == "historical" or _strategy_created_iteration(row, iteration) < iteration
    ]


def _strategy_mean_strength(rows: Sequence[Mapping[str, Any]]) -> float:
    return _mean([float(row.get("strength", 0.0) or 0.0) for row in rows])


def _new_strategy_strength_delta(
    new_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
) -> float:
    if not new_rows or not prior_rows:
        return 0.0
    return _strategy_mean_strength(new_rows) - _strategy_mean_strength(prior_rows)


def _role_coverage(role_counts: object, required_roles: Sequence[str]) -> float:
    required = tuple(str(role) for role in required_roles)
    if not required:
        return 1.0
    if not isinstance(role_counts, Mapping):
        return 0.0
    present = sum(1 for role in required if int(role_counts.get(role, 0) or 0) > 0)
    return present / len(required)


def _league_selfplay_health_red_lines(
    report: Mapping[str, Any],
    *,
    min_attack_pool: int,
    min_defense_pool: int,
    min_total_clusters: int,
    min_payoff_density: float,
    min_active_pool_fraction: float,
    min_new_attack_strength_delta: float | None,
    min_new_defense_strength_delta: float | None,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("rounds", 0) or 0) <= 0:
        violations.append("no_round_dirs")
    round_reports = report.get("round_reports", ())
    if isinstance(round_reports, Sequence) and not isinstance(round_reports, (str, bytes)):
        if any(isinstance(row, Mapping) and not row.get("has_league_state") for row in round_reports):
            violations.append("league_state_missing")
    if int(report.get("attack_pool", 0) or 0) < int(min_attack_pool):
        violations.append("attack_pool_too_small")
    if int(report.get("defense_pool", 0) or 0) < int(min_defense_pool):
        violations.append("defense_pool_too_small")
    if int(report.get("total_clusters", 0) or 0) < int(min_total_clusters):
        violations.append("league_cluster_collapse")
    if float(report.get("attack_role_coverage", 0.0) or 0.0) < 1.0:
        violations.append("attack_role_coverage_low")
    if float(report.get("defense_role_coverage", 0.0) or 0.0) < 1.0:
        violations.append("defense_role_coverage_low")
    if float(report.get("payoff_density", 0.0) or 0.0) < float(min_payoff_density):
        violations.append("payoff_density_low")
    if int(report.get("payoff_entries", 0) or 0) <= 0:
        violations.append("no_payoffs")
    if float(report.get("active_pool_fraction", 0.0) or 0.0) < float(min_active_pool_fraction):
        violations.append("active_pool_fraction_low")
    if (
        min_new_attack_strength_delta is not None
        and float(report.get("new_attack_strength_delta", 0.0) or 0.0) < float(min_new_attack_strength_delta)
    ):
        violations.append("new_attack_strength_delta_low")
    if (
        min_new_defense_strength_delta is not None
        and float(report.get("new_defense_strength_delta", 0.0) or 0.0) < float(min_new_defense_strength_delta)
    ):
        violations.append("new_defense_strength_delta_low")
    return _dedupe_strings(violations)


def _production_readiness_report_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "readable": False,
            "schema_version": None,
            "schema_versions": [],
            "module": None,
            "production_ready": None,
            "red_line_violations": [],
            "read_error": "missing_report",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("report must contain a JSON object")
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "schema_version": None,
            "schema_versions": [],
            "module": None,
            "production_ready": None,
            "red_line_violations": [],
            "read_error": str(exc),
        }
    red_lines = payload.get("red_line_violations", ())
    if isinstance(red_lines, Sequence) and not isinstance(red_lines, (str, bytes)):
        red_line_values = _dedupe_strings([str(value) for value in red_lines])
    else:
        red_line_values = []
    production_ready = payload.get("production_ready")
    production_ready_value = production_ready if isinstance(production_ready, bool) else None
    schema_version = None if payload.get("schema_version") is None else str(payload.get("schema_version"))
    schema_versions = _production_readiness_schema_versions(payload)
    return {
        "path": str(path),
        "exists": True,
        "readable": True,
        "schema_version": schema_version,
        "schema_versions": schema_versions,
        "module": None if payload.get("module") is None else str(payload.get("module")),
        "production_ready": production_ready_value,
        "red_line_violations": red_line_values,
        "read_error": None,
    }


def _production_readiness_schema_versions(payload: Mapping[str, Any]) -> list[str]:
    versions: list[str] = []
    top_level_schema = payload.get("schema_version")
    if top_level_schema is not None:
        versions.append(str(top_level_schema))

    ingestions = payload.get("ingestions")
    if isinstance(ingestions, Sequence) and not isinstance(ingestions, (str, bytes)):
        for ingestion in ingestions:
            if isinstance(ingestion, Mapping) and ingestion.get("schema_version") is not None:
                versions.append(str(ingestion["schema_version"]))

    drift = payload.get("drift")
    if isinstance(drift, Mapping) and drift.get("schema_version") is not None:
        versions.append(str(drift["schema_version"]))

    return _dedupe_strings(versions)


def _production_readiness_red_lines(
    report: Mapping[str, Any],
    *,
    min_clean_report_rate: float,
    require_production_ready: bool,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("reports", 0) or 0) <= 0:
        violations.append("no_reports")
    if int(report.get("read_error_reports", 0) or 0) > 0:
        violations.append("report_read_error")
    if report.get("missing_required_schema_versions"):
        violations.append("required_schema_missing")
    if int(report.get("red_line_reports", 0) or 0) > 0:
        violations.append("red_line_reports_present")
    if require_production_ready and int(report.get("production_ready_false_reports", 0) or 0) > 0:
        violations.append("production_ready_false")
    if float(report.get("clean_report_rate", 0.0) or 0.0) < float(min_clean_report_rate):
        violations.append("clean_report_rate_low")
    return _dedupe_strings(violations)


def _v4_conformance_requirement_row(
    requirement: Mapping[str, Any],
    report_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    requirement_id = str(requirement.get("requirement_id", "unknown"))
    required_schemas = tuple(str(value) for value in requirement.get("required_schema_versions", ()))
    evidence_rows = [
        row
        for row in report_rows
        if any(schema in set(str(value) for value in row.get("schema_versions", ())) for schema in required_schemas)
    ]
    evidence_schema_versions = _dedupe_strings(
        [
            str(schema)
            for row in evidence_rows
            for schema in row.get("schema_versions", ())
            if str(schema) in required_schemas
        ]
    )
    missing_schema_versions = [
        schema
        for schema in required_schemas
        if schema not in evidence_schema_versions
    ]
    child_red_lines = _dedupe_strings(
        [
            str(violation)
            for row in evidence_rows
            for violation in row.get("red_line_violations", ())
        ]
    )
    production_not_ready = [
        str(row.get("path"))
        for row in evidence_rows
        if row.get("production_ready") is False
    ]
    status = "pass"
    if missing_schema_versions or child_red_lines or production_not_ready:
        status = "fail"
    return {
        "requirement_id": requirement_id,
        "title": str(requirement.get("title", requirement_id)),
        "required_schema_versions": list(required_schemas),
        "evidence_schema_versions": evidence_schema_versions,
        "missing_schema_versions": missing_schema_versions,
        "evidence_count": len(evidence_rows),
        "report_paths": [str(row.get("path")) for row in evidence_rows],
        "child_red_line_violations": child_red_lines,
        "production_not_ready_paths": production_not_ready,
        "status": status,
    }


def _v4_conformance_red_lines(report: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if int(report.get("reports", 0) or 0) <= 0:
        violations.append("no_reports")
    if int(report.get("read_error_reports", 0) or 0) > 0:
        violations.append("report_read_error")
    requirements = report.get("requirements", ())
    if isinstance(requirements, Sequence) and not isinstance(requirements, (str, bytes)):
        for row in requirements:
            if not isinstance(row, Mapping):
                continue
            requirement_id = str(row.get("requirement_id", "unknown"))
            if row.get("missing_schema_versions"):
                violations.append(f"{requirement_id}_evidence_missing")
            if row.get("child_red_line_violations"):
                violations.append(f"{requirement_id}_red_lines_present")
            if row.get("production_not_ready_paths"):
                violations.append(f"{requirement_id}_production_not_ready")
    return _dedupe_strings(violations)


def _attack_failure_validation_row(
    risk_report_value: object,
    diagnostics_value: object,
    *,
    source_type: str,
    source_path: str,
    attack_id: object,
    defense_id: object,
) -> dict[str, Any]:
    risk_report = risk_report_value if isinstance(risk_report_value, Mapping) else {}
    diagnostics = (
        diagnostics_value
        if isinstance(diagnostics_value, Sequence) and not isinstance(diagnostics_value, (str, bytes))
        else ()
    )
    failure = risk_report.get("failure")
    failure_code = risk_report.get("failure_code")
    failure_stage = risk_report.get("failure_stage")
    has_failure = failure is not None or failure_code is not None or failure_stage is not None
    matching_diagnostic = _diagnostics_match_failure(diagnostics, failure_code, failure_stage)
    return {
        "source_type": source_type,
        "source_path": source_path,
        "attack_id": None if attack_id is None else str(attack_id),
        "defense_id": None if defense_id is None else str(defense_id),
        "has_risk_report": isinstance(risk_report_value, Mapping),
        "has_failure": has_failure,
        "failure": None if failure is None else str(failure),
        "failure_code": None if failure_code is None else str(failure_code),
        "failure_stage": None if failure_stage is None else str(failure_stage),
        "has_failure_code": failure_code is not None,
        "has_failure_stage": failure_stage is not None,
        "diagnostic_count": len(diagnostics),
        "has_matching_diagnostic": matching_diagnostic,
    }


def _diagnostics_match_failure(diagnostics: object, failure_code: object, failure_stage: object) -> bool:
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        return False
    expected_code = None if failure_code is None else str(failure_code)
    expected_stage = None if failure_stage is None else str(failure_stage)
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        code_ok = expected_code is not None and str(diagnostic.get("code")) == expected_code
        stage_ok = expected_stage is None or str(diagnostic.get("stage")) == expected_stage
        if code_ok and stage_ok:
            return True
    return False


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _attack_oracle_failure_red_lines(
    report: Mapping[str, Any],
    *,
    min_failure_annotation_coverage: float,
    min_failure_diagnostic_coverage: float,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("checked_rows", 0) or 0) <= 0:
        violations.append("no_attack_oracle_rows")
        return violations
    if int(report.get("candidate_missing_risk_report_rows", 0) or 0) > 0:
        violations.append("candidate_risk_report_missing")
    failure_rows = int(report.get("failure_rows", 0) or 0)
    rows = report.get("validation_rows", ())
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        if any(isinstance(row, Mapping) and row.get("has_failure") and not row.get("has_failure_code") for row in rows):
            violations.append("failure_code_missing")
        if any(isinstance(row, Mapping) and row.get("has_failure") and not row.get("has_failure_stage") for row in rows):
            violations.append("failure_stage_missing")
    if failure_rows > 0:
        if float(report.get("failure_annotation_coverage", 0.0) or 0.0) < float(min_failure_annotation_coverage):
            violations.append("failure_annotation_coverage_low")
        if float(report.get("failure_diagnostic_coverage", 0.0) or 0.0) < float(min_failure_diagnostic_coverage):
            violations.append("failure_diagnostic_coverage_low")
    return violations


def _nested_float_values(rows: list[dict[str, Any]], parent: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        payload = row.get(parent)
        if not isinstance(payload, Mapping):
            continue
        value = payload.get(key)
        if _is_number(value):
            values.append(float(value))
    return values


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_from_first_number(*values: object) -> int:
    for value in values:
        if _is_number(value):
            return int(value)
    return 0


def _float_or_zero(value: object) -> float:
    return float(value) if _is_number(value) else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _belief_domain_stats_mapping(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(item) for key, item in value.items() if _is_number(item)}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    stats: dict[str, float] = {}
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        key = item[0]
        stat_value = item[1]
        if _is_number(stat_value):
            stats[str(key)] = float(stat_value)
    return stats


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mask_explanation_red_lines(
    report: Mapping[str, Any],
    *,
    min_hidden_explanation_coverage: float,
) -> list[str]:
    violations: list[str] = []
    defenses = int(report.get("defenses", 0) or 0)
    if defenses <= 0:
        violations.append("no_scored_defenses")
        return violations
    if int(report.get("risk_report_rows", 0) or 0) < defenses:
        violations.append("defense_risk_report_missing")
    if int(report.get("mask_explanation_rows", 0) or 0) < defenses:
        violations.append("mask_explanation_missing")
    if int(report.get("defenses_with_no_hidden_slots", 0) or 0) > 0:
        violations.append("no_hidden_slots")
    if float(report.get("hidden_explanation_coverage", 0.0) or 0.0) < float(min_hidden_explanation_coverage):
        violations.append("hidden_slot_explanation_coverage_low")
    if int(report.get("learned_mask_score_rows", 0) or 0) < defenses:
        violations.append("learned_mask_score_missing")
    if int(report.get("counter_attack_risk_rows", 0) or 0) < defenses:
        violations.append("counter_attack_risk_missing")
    return violations


def _belief_real_distribution_red_lines(
    report: Mapping[str, Any],
    *,
    min_real_coverage: float,
    min_mean_real_records: float,
    min_mean_real_similarity: float,
    max_oracle_alignment_mae: float,
) -> list[str]:
    violations: list[str] = []
    candidates = int(report.get("candidates", 0) or 0)
    if candidates <= 0:
        violations.append("no_candidate_rows")
        return violations
    if int(report.get("belief_domain_stats_rows", 0) or 0) < candidates:
        violations.append("belief_domain_stats_missing")
    if float(report.get("real_distribution_coverage", 0.0) or 0.0) < float(min_real_coverage):
        violations.append("real_distribution_coverage_low")
    if float(report.get("mean_real_record_count", 0.0) or 0.0) < float(min_mean_real_records):
        violations.append("real_record_count_low")
    if float(report.get("mean_real_similarity", 0.0) or 0.0) < float(min_mean_real_similarity):
        violations.append("real_similarity_low")
    if (
        int(report.get("oracle_alignment_rows", 0) or 0) > 0
        and float(report.get("oracle_alignment_mae", 0.0) or 0.0) > float(max_oracle_alignment_mae)
    ):
        violations.append("real_oracle_alignment_error_high")
    return violations


def _data_engineering_red_lines(
    report: Mapping[str, Any],
    *,
    min_metadata_coverage: float,
    min_core_table_coverage: float,
    min_artifact_hash_coverage: float,
) -> list[str]:
    violations: list[str] = []
    if int(report.get("rounds", 0) or 0) <= 0:
        violations.append("no_round_dirs")
        return violations
    if float(report.get("metadata_coverage", 0.0) or 0.0) < float(min_metadata_coverage):
        violations.append("metadata_coverage_low")
    if int(report.get("metadata_files", 0) or 0) < int(report.get("rounds", 0) or 0):
        violations.append("run_metadata_missing")
    if int(report.get("artifact_missing_count", 0) or 0) > 0:
        violations.append("artifact_missing")
    if int(report.get("artifact_hash_mismatch_count", 0) or 0) > 0:
        violations.append("artifact_hash_mismatch")
    if float(report.get("artifact_hash_coverage", 0.0) or 0.0) < float(min_artifact_hash_coverage):
        violations.append("artifact_hash_coverage_low")
    if int(report.get("core_table_files_found", 0) or 0) < int(report.get("core_table_files_expected", 0) or 0):
        violations.append("core_table_missing")
    if int(report.get("core_table_empty_count", 0) or 0) > 0:
        violations.append("core_table_empty")
    if int(report.get("core_table_schema_mismatch_count", 0) or 0) > 0:
        violations.append("core_table_schema_mismatch")
    if float(report.get("core_table_coverage", 0.0) or 0.0) < float(min_core_table_coverage):
        violations.append("core_table_coverage_low")
    return _dedupe_strings(violations)
