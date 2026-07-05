"""在线 best-response 层，对应 tex §1054-1168。

这里放 AttackOracle、DefenseOracle、mask search 和 oracle 输出契约。
训练循环、真实 backend 和报告聚合不属于本层。
"""

from .attack import AttackCandidateSource, AttackOracle, AttackOracleConfig, AttackOracleOutput
from .defense import DefenseOracle, DefenseOracleConfig, DefenseOracleOutput, DefenseRosterSource
from .mask_search import MaskSearcher, MaskSlotScoreProvider, enumerate_legal_masks, legal_team_mask_patterns

__all__ = [
    "AttackCandidateSource",
    "AttackOracle",
    "AttackOracleConfig",
    "AttackOracleOutput",
    "DefenseOracle",
    "DefenseOracleConfig",
    "DefenseOracleOutput",
    "DefenseRosterSource",
    "MaskSearcher",
    "MaskSlotScoreProvider",
    "enumerate_legal_masks",
    "legal_team_mask_patterns",
]
