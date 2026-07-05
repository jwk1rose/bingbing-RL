from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


DAILY_TRAINING_REPORT_SCHEMA_VERSION = "daily_training_report.v1"


def brier_score(labels: Sequence[float], scores: Sequence[float]) -> float:
    _validate_same_length(labels, scores)
    if not labels:
        return 0.0
    return sum((float(score) - float(label)) ** 2 for label, score in zip(labels, scores)) / len(labels)


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    _validate_same_length(labels, scores)
    positives = [(score, index) for index, (label, score) in enumerate(zip(labels, scores)) if float(label) >= 0.5]
    negatives = [(score, index) for index, (label, score) in enumerate(zip(labels, scores)) if float(label) < 0.5]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for pos_score, _pos_index in positives:
        for neg_score, _neg_index in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def expected_calibration_error(labels: Sequence[float], scores: Sequence[float], *, bins: int = 10) -> float:
    _validate_same_length(labels, scores)
    if not labels:
        return 0.0
    if bins <= 0:
        raise ValueError("bins must be positive")
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        rows = [
            (float(label), float(score))
            for label, score in zip(labels, scores)
            if lower <= float(score) < upper or (bin_index == bins - 1 and float(score) == 1.0)
        ]
        if not rows:
            continue
        accuracy = sum(label for label, _score in rows) / len(rows)
        confidence = sum(score for _label, score in rows) / len(rows)
        ece += len(rows) / total * abs(accuracy - confidence)
    return ece


def precision_at_k(labels: Sequence[float], scores: Sequence[float], *, k: int) -> float:
    _validate_same_length(labels, scores)
    if k <= 0 or not labels:
        return 0.0
    rows = _top_k(labels, scores, k)
    return sum(1.0 for label, _score in rows if float(label) >= 0.5) / len(rows)


def recall_at_k(labels: Sequence[float], scores: Sequence[float], *, k: int) -> float:
    _validate_same_length(labels, scores)
    positives = sum(1.0 for label in labels if float(label) >= 0.5)
    if k <= 0 or positives <= 0:
        return 0.0
    rows = _top_k(labels, scores, k)
    hits = sum(1.0 for label, _score in rows if float(label) >= 0.5)
    return hits / positives


@dataclass(frozen=True)
class DailyTrainingReport:
    date: str
    sim_games: int
    real_matches: int
    single_model: Mapping[str, float]
    attack_oracle: Mapping[str, float]
    defense_oracle: Mapping[str, float]
    league: Mapping[str, float | int]
    underdog: Mapping[str, float | int]
    active_queries: Sequence[Mapping[str, Any]]
    failure_cases: Sequence[Mapping[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = DAILY_TRAINING_REPORT_SCHEMA_VERSION
        payload["module"] = "DailyTrainingReport"
        return payload


def _top_k(labels: Sequence[float], scores: Sequence[float], k: int) -> list[tuple[float, float]]:
    rows = sorted(zip(labels, scores), key=lambda item: float(item[1]), reverse=True)
    return [(float(label), float(score)) for label, score in rows[: min(k, len(rows))]]


def _validate_same_length(labels: Sequence[float], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
