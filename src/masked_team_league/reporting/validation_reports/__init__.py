"""验证报告集合，对齐 tex 中的红线与生产就绪检查。

当前 builders 模块保留共享读取、聚合和红线 helper；对外只暴露稳定的
`build_*_report` 函数，后续继续拆分单个 schema 时不影响调用方。
"""

from __future__ import annotations

from .builders import (
    build_active_query_feedback_report,
    build_active_real_query_dispatch_validation_report,
    build_attack_oracle_failure_validation_report,
    build_belief_real_distribution_validation_report,
    build_data_engineering_validation_report,
    build_defense_anti_meta_effectiveness_report,
    build_exploiter_effectiveness_report,
    build_league_selfplay_health_report,
    build_learned_exploiter_validation_report,
    build_mask_explanation_validation_report,
    build_production_readiness_report,
    build_underdog_residual_validation_report,
    build_v4_conformance_validation_report,
)

__all__ = [
    "build_active_query_feedback_report",
    "build_active_real_query_dispatch_validation_report",
    "build_attack_oracle_failure_validation_report",
    "build_belief_real_distribution_validation_report",
    "build_data_engineering_validation_report",
    "build_defense_anti_meta_effectiveness_report",
    "build_exploiter_effectiveness_report",
    "build_league_selfplay_health_report",
    "build_learned_exploiter_validation_report",
    "build_mask_explanation_validation_report",
    "build_production_readiness_report",
    "build_underdog_residual_validation_report",
    "build_v4_conformance_validation_report",
]
