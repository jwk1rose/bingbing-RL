"""报告层兼容 facade。

新代码优先从 `round_reports` 或 `validation_reports` 导入。保留本模块是为了
让已有调用在重构期间不需要一次性改完。
"""

from __future__ import annotations

from .round_reports import build_league_round_report, red_line_violations
from .validation_reports import (
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
    "build_league_round_report",
    "build_league_selfplay_health_report",
    "build_learned_exploiter_validation_report",
    "build_mask_explanation_validation_report",
    "build_production_readiness_report",
    "build_underdog_residual_validation_report",
    "build_v4_conformance_validation_report",
    "red_line_violations",
]
