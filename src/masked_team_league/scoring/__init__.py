"""评分与搜索工具层，对应 tex §472-556、§1091-1101。

这里放 BO3/BO5 胜率、资源成本、surrogate scorer、缓存和 successive halving。
Oracle 可以组合这些工具，但具体工具不反向依赖 oracle。
"""

from .cache import MatchupCacheKey, SimulationCache, SimulationResult, SurrogateSimulator
from .halving import HalvingStage, HalvingTrace, successive_halving
from .match import (
    RankedItem,
    all_win_probability,
    clip_probability,
    diversity_select,
    independent_match_loss_probability,
    jaccard_similarity,
    match_win_probability,
    plan_cost,
    team_cost,
)
from .surrogate import HeuristicSurrogateScorer, SurrogatePrediction, SurrogateScorer

__all__ = [
    "HalvingStage",
    "HalvingTrace",
    "HeuristicSurrogateScorer",
    "MatchupCacheKey",
    "RankedItem",
    "SimulationCache",
    "SimulationResult",
    "SurrogatePrediction",
    "SurrogateScorer",
    "SurrogateSimulator",
    "all_win_probability",
    "clip_probability",
    "diversity_select",
    "independent_match_loss_probability",
    "jaccard_similarity",
    "match_win_probability",
    "plan_cost",
    "successive_halving",
    "team_cost",
]
