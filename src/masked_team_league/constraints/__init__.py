"""硬约束层，对应 tex §360-470、§1693-1714。

这里负责结构合法性、隐藏槽位补全、MRV/forward checking 和 legal action mask。
网络和搜索模块必须通过本层获得合法候选，不能自行信任自由输出。
"""

from .diagnostics import LEGAL_DIAGNOSTIC_SCHEMA_VERSION, LegalDiagnostic, LegalReport
from .engine import ConstraintEngine

__all__ = [
    "ConstraintEngine",
    "LEGAL_DIAGNOSTIC_SCHEMA_VERSION",
    "LegalDiagnostic",
    "LegalReport",
]
