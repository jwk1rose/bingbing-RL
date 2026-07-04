from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class HalvingStage:
    games_each: int
    keep: int


@dataclass(frozen=True)
class HalvingTrace(Generic[T]):
    stage_index: int
    games_each: int
    kept: tuple[T, ...]
    scores: tuple[tuple[str, float], ...]


def successive_halving(
    items: list[T],
    *,
    stages: tuple[HalvingStage, ...],
    evaluate: Callable[[T, int], float],
    key: Callable[[T], str],
) -> tuple[list[T], tuple[HalvingTrace[T], ...]]:
    current = list(items)
    traces: list[HalvingTrace[T]] = []
    for stage_index, stage in enumerate(stages):
        scored = [(item, evaluate(item, stage.games_each)) for item in current]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        current = [item for item, _score in scored[: stage.keep]]
        traces.append(
            HalvingTrace(
                stage_index=stage_index,
                games_each=stage.games_each,
                kept=tuple(current),
                scores=tuple((key(item), score) for item, score in scored[: stage.keep]),
            )
        )
        if len(current) <= 1:
            break
    return current, tuple(traces)
