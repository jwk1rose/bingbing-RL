"""真实平台适配层，对应 tex §1235-1283、§1494-1497。

这里放 backend client、battle request codec、真实 oracle 评估、资源加载和真实校准。
算法层通过本层获得真实平台反馈，不直接构造 backend 协议。
"""

from .backend import OracleBackendClient, OracleBackendSimulator, is_oracle_backend_ready
from .backend_codec import (
    PlanBattleScore,
    build_plan_battle_requests,
    result_to_attack_win_rate,
    score_plan_battle_results,
)
from .calibration import (
    RealCalibrationModel,
    RealCalibrationSampleBuildSummary,
    RealCalibrationIngestionSummary,
    RealMetaDB,
    RealMetaObservationMatch,
    RealMetaRecord,
    VersionDriftReport,
    build_real_calibration_features,
    build_real_calibration_samples_from_artifacts,
    build_real_calibration_validation_report,
    build_version_drift_report,
    ingest_active_real_query_feedback,
    ingest_league_round_real_meta,
    real_meta_observation_similarity,
    time_decay_weight,
)
from .oracle import OracleBatchEvaluator, OracleEvaluationRecord
from .resources import (
    DEFAULT_ORACLE_EXCLUDED_HERO_IDS,
    HeroResource,
    HeroResourceBundle,
    RuntimeResourceRules,
    load_decoded_runtime_rules,
    load_hero_resource_bundle,
    load_peak_arena_camp_hero_ids,
    load_unique_legend_equip_ids,
)

__all__ = [
    "DEFAULT_ORACLE_EXCLUDED_HERO_IDS",
    "HeroResource",
    "HeroResourceBundle",
    "OracleBackendClient",
    "OracleBackendSimulator",
    "OracleBatchEvaluator",
    "OracleEvaluationRecord",
    "PlanBattleScore",
    "RealCalibrationModel",
    "RealCalibrationSampleBuildSummary",
    "RealCalibrationIngestionSummary",
    "RealMetaDB",
    "RealMetaObservationMatch",
    "RealMetaRecord",
    "RuntimeResourceRules",
    "VersionDriftReport",
    "build_plan_battle_requests",
    "build_real_calibration_features",
    "build_real_calibration_samples_from_artifacts",
    "build_real_calibration_validation_report",
    "build_version_drift_report",
    "ingest_active_real_query_feedback",
    "ingest_league_round_real_meta",
    "is_oracle_backend_ready",
    "load_decoded_runtime_rules",
    "load_hero_resource_bundle",
    "load_peak_arena_camp_hero_ids",
    "load_unique_legend_equip_ids",
    "result_to_attack_win_rate",
    "real_meta_observation_similarity",
    "score_plan_battle_results",
    "time_decay_weight",
]
