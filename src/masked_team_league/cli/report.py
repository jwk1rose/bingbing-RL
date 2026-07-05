"""报告、红线校验与生产就绪 CLI 分组。"""

from __future__ import annotations

from masked_team_league.cli._dispatch import CommandSpec, dispatch_group


COMMANDS = {
    "round": CommandSpec("report_league_round", "汇总单轮联赛日报式报告。"),
    "active-query-feedback": CommandSpec("report_active_query_feedback", "校验 active/real-query 反馈。"),
    "active-real-dispatch": CommandSpec(
        "report_active_real_query_dispatch_validation",
        "校验 active real-query 派发完成率。",
    ),
    "attack-oracle-failure": CommandSpec(
        "report_attack_oracle_failure_validation",
        "校验 AttackOracle 失败注释与诊断。",
    ),
    "belief-real-distribution": CommandSpec(
        "report_belief_real_distribution_validation",
        "校验 belief 使用真实分布的证据。",
    ),
    "data-engineering": CommandSpec("report_data_engineering_validation", "校验 core tables 与 metadata 覆盖。"),
    "defense-anti-meta": CommandSpec(
        "report_defense_anti_meta_effectiveness",
        "汇总防守反 meta 残差反馈质量。",
    ),
    "exploiter": CommandSpec("report_exploiter_effectiveness", "汇总 exploiter 目标残差反馈。"),
    "league-health": CommandSpec("report_league_selfplay_health", "校验联赛池、角色、payoff 与保留策略。"),
    "learned-exploiter": CommandSpec("report_learned_exploiter_validation", "校验学习型 exploiter 产线证据。"),
    "mask-explanation": CommandSpec("report_mask_explanation_validation", "校验 mask 风险解释覆盖。"),
    "production-readiness": CommandSpec("report_production_readiness", "聚合生产就绪红线。"),
    "real-calibration": CommandSpec("report_real_calibration_validation", "校验真实校准 holdout 表现。"),
    "underdog-residual": CommandSpec("report_underdog_residual_validation", "校验 underdog 残差奖励证据。"),
    "v4-conformance": CommandSpec("report_v4_conformance_validation", "校验 v4 规范符合度证据。"),
}


def main() -> int:
    return dispatch_group(description="Masked Team League reporting commands.", commands=COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
