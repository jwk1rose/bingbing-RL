"""训练与训练调度 CLI 分组。"""

from __future__ import annotations

from masked_team_league.cli._dispatch import CommandSpec, dispatch_group


COMMANDS = {
    "single-team": CommandSpec("train_single_team_model", "训练 SingleTeamWinrateModel。"),
    "attack-proposal": CommandSpec("train_attack_proposal", "训练攻击提案网络。"),
    "defense-proposal": CommandSpec("train_defense_proposal", "训练防守阵容提案网络。"),
    "mask-selection": CommandSpec("train_mask_selection", "训练 mask 选择网络。"),
    "belief-ranker": CommandSpec("train_belief_ranker", "训练 BeliefEngine 排序器。"),
    "schedule": CommandSpec("run_training_schedule", "生成或执行 v4 训练计划。"),
    "scheduler-daemon": CommandSpec("run_training_scheduler_daemon", "循环运行训练调度器。"),
    "build-belief-ranker-dataset": CommandSpec(
        "build_belief_ranker_dataset",
        "从联赛轮次产物构建 belief-ranker 数据集。",
    ),
    "build-split-manifest": CommandSpec("build_split_manifest", "生成数据切分 manifest。"),
    "select-checkpoint": CommandSpec("select_model_checkpoint", "从 checkpoint registry 选择最佳模型。"),
}


def main() -> int:
    return dispatch_group(description="Masked Team League training commands.", commands=COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
