"""联赛轮次日报入口。

日报依赖 validation builders 中的共享 JSON 读取和红线逻辑。单独暴露这个模块，
是为了让“日常 round 报告”和“离线 validation report”在导入层面分开。
"""

from __future__ import annotations

from .validation_reports.builders import build_league_round_report, red_line_violations

__all__ = ["build_league_round_report", "red_line_violations"]
