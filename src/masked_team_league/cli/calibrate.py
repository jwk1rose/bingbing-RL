"""真实平台校准与主动查询 CLI 分组。"""

from __future__ import annotations

from masked_team_league.cli._dispatch import CommandSpec, dispatch_group


COMMANDS = {
    "ingest-real": CommandSpec("ingest_real_calibration", "写入 RealMetaDB 并生成版本漂移报告。"),
    "build-real-samples": CommandSpec("build_real_calibration_samples", "从联赛和主动查询产物构建真实校准样本。"),
    "fit-real-feature": CommandSpec("fit_real_feature_calibration", "拟合真实分布特征校准器。"),
    "dispatch-active-real": CommandSpec("dispatch_active_real_queries", "派发 active real-query 并写入 teacher JSONL。"),
}


def main() -> int:
    return dispatch_group(description="Masked Team League real-platform calibration commands.", commands=COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
