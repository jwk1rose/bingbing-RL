"""Belief 层，对应 tex §994-1053、§1803-1824。

这里负责从 mask observation 构造合法补全集合，并用真实分布、defense pool 和 ranker 加权。
AttackOracle 只能消费 BeliefOutput，不应绕过本层偷看完整防守。
"""

from .engine import BeliefEngine, BeliefOutput, BeliefRanker
from .ranker import (
    BeliefRankerDatasetBuildResult,
    BeliefRankerTrainingHistory,
    BeliefRankerTrainingSample,
    BeliefRankerVocab,
    TorchBeliefRanker,
    TorchBeliefRankerAdapter,
    build_belief_ranker_dataset_from_rounds,
    evaluate_belief_ranker,
    load_belief_ranker_checkpoint,
    load_belief_ranker_samples_jsonl,
    save_belief_ranker_checkpoint,
    train_belief_ranker,
)

__all__ = [
    "BeliefEngine",
    "BeliefOutput",
    "BeliefRanker",
    "BeliefRankerDatasetBuildResult",
    "BeliefRankerTrainingHistory",
    "BeliefRankerTrainingSample",
    "BeliefRankerVocab",
    "TorchBeliefRanker",
    "TorchBeliefRankerAdapter",
    "build_belief_ranker_dataset_from_rounds",
    "evaluate_belief_ranker",
    "load_belief_ranker_checkpoint",
    "load_belief_ranker_samples_jsonl",
    "save_belief_ranker_checkpoint",
    "train_belief_ranker",
]
