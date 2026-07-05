"""报告与红线验证层，对应 tex §1440-1467、§2381-2402。

这里放稳定 JSON contract、日报、validation report、production readiness 和 ablation。
运行时模块只产出结构化 artifact；跨 artifact 的解释和红线判断放在这里。
"""

from .contracts import (
    REQUIRED_RUNTIME_TRAINING_SCHEMA_VERSIONS,
    OutputContract,
    failure_diagnostics,
    jsonable,
    output_contract_registry,
)

__all__ = [
    "OutputContract",
    "REQUIRED_RUNTIME_TRAINING_SCHEMA_VERSIONS",
    "failure_diagnostics",
    "jsonable",
    "output_contract_registry",
]
