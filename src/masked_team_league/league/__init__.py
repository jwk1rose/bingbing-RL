"""League 与运行编排层，对应 tex §1169-1234、§1284-1380。

这里放 ActivePerception、LeagueManager、单轮 round runner 和 self-play。
本层负责串联 oracle、真实平台和训练反馈，不承载底层合法性或报告实现。
"""

from .active_feedback import ActiveRealQueryDispatchSummary, dispatch_active_real_queries
from .active_perception import ActivePerceptionScheduler, Query, SchedulerConfig, SchedulerOutput
from .manager import LeagueManager, PayoffEntry, StrategyRecord

__all__ = [
    "ActivePerceptionScheduler",
    "ActiveRealQueryDispatchSummary",
    "LeagueManager",
    "PayoffEntry",
    "Query",
    "SchedulerConfig",
    "SchedulerOutput",
    "StrategyRecord",
    "dispatch_active_real_queries",
]
