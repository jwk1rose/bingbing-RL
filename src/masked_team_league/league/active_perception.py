from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    query_id: str
    query_type: str
    info_gain: float
    decision_impact: float
    meta_frequency: float
    novelty: float
    underdog_potential: float
    cost: float


@dataclass(frozen=True)
class SchedulerConfig:
    lambda_info_gain: float = 1.0
    lambda_decision_impact: float = 1.0
    lambda_meta_frequency: float = 0.2
    lambda_novelty: float = 0.5
    lambda_underdog: float = 0.7
    lambda_cost: float = 0.1


@dataclass(frozen=True)
class SchedulerOutput:
    sim_queue: tuple[Query, ...]
    real_query_queue: tuple[Query, ...]
    scores: tuple[tuple[str, float], ...]


class ActivePerceptionScheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def schedule(self, queries: tuple[Query, ...], *, sim_keep: int, real_keep: int = 0) -> SchedulerOutput:
        scored = [(query, self.score(query)) for query in queries]
        scored.sort(key=lambda item: item[1], reverse=True)
        sim_queue = tuple(query for query, _score in scored[:sim_keep])
        real_query_queue = tuple(query for query, _score in scored[sim_keep : sim_keep + real_keep])
        return SchedulerOutput(
            sim_queue=sim_queue,
            real_query_queue=real_query_queue,
            scores=tuple((query.query_id, score) for query, score in scored),
        )

    def score(self, query: Query) -> float:
        c = self.config
        return (
            c.lambda_info_gain * query.info_gain
            + c.lambda_decision_impact * query.decision_impact
            + c.lambda_meta_frequency * query.meta_frequency
            + c.lambda_novelty * query.novelty
            + c.lambda_underdog * query.underdog_potential
            - c.lambda_cost * query.cost
        )
