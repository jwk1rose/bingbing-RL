"""生成层，对应 tex §665-993、§1737-1802。

这里放合法 proposal 生成、Attack/Defense 自回归网络、teacher 数据、mask selection。
硬合法性仍由 constraints 层提供，网络只负责 proposal 和估值。
"""

from .legal_generator import GenerationGoal, LegalPlanGenerator
from .legal_masked import AttackNetInput, AttackNetOutput, LegalMaskedAttackGenerator
from .proposal_networks import (
    AttackGenerationNetwork,
    DefenseRosterGenerationNetwork,
    GenerationContextEncoder,
    GenerationContextOutput,
    MaskSelectionNetwork,
    ProposalDistillationLoss,
    ProposalNetworkConfig,
    ProposalNetworkOutput,
    ProposalSequence,
    apply_legal_action_mask,
    beam_search_proposal_tokens,
    build_causal_attention_mask,
    proposal_distillation_loss,
    sample_proposal_tokens,
)

__all__ = [
    "AttackGenerationNetwork",
    "AttackNetInput",
    "AttackNetOutput",
    "DefenseRosterGenerationNetwork",
    "GenerationContextEncoder",
    "GenerationContextOutput",
    "GenerationGoal",
    "LegalMaskedAttackGenerator",
    "LegalPlanGenerator",
    "MaskSelectionNetwork",
    "ProposalDistillationLoss",
    "ProposalNetworkConfig",
    "ProposalNetworkOutput",
    "ProposalSequence",
    "apply_legal_action_mask",
    "beam_search_proposal_tokens",
    "build_causal_attention_mask",
    "proposal_distillation_loss",
    "sample_proposal_tokens",
]
