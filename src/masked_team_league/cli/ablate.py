"""消融实验 CLI 分组。"""

from __future__ import annotations

from masked_team_league.cli._dispatch import CommandSpec, dispatch_group


COMMANDS = {
    "suite": CommandSpec("run_ablation_suite", "从联赛轮次产物构建 v4 消融套件报告。"),
    "experiments": CommandSpec("run_v4_ablation_experiments", "运行 v4 消融实验并输出报告。"),
}


def main() -> int:
    return dispatch_group(description="Masked Team League ablation commands.", commands=COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
