from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ModelCheckpointRecord:
    checkpoint_id: str
    model_type: str
    model_path: str
    metrics_path: str
    created_at: float
    dataset_hash: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", {str(key): float(value) for key, value in self.metrics.items()})

    def metric(self, name: str) -> float | None:
        value = self.metrics.get(name)
        return None if value is None else float(value)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "metrics_path": self.metrics_path,
            "created_at": self.created_at,
            "dataset_hash": self.dataset_hash,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "ModelCheckpointRecord":
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("checkpoint metrics must be an object")
        return cls(
            checkpoint_id=str(payload["checkpoint_id"]),
            model_type=str(payload["model_type"]),
            model_path=str(payload["model_path"]),
            metrics_path=str(payload["metrics_path"]),
            created_at=float(payload["created_at"]),
            dataset_hash=str(payload.get("dataset_hash", "")),
            metrics={str(key): float(value) for key, value in metrics.items()},
        )


class CheckpointRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: list[ModelCheckpointRecord] = []
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            records = payload.get("checkpoints", payload if isinstance(payload, list) else [])
            if not isinstance(records, list):
                raise ValueError("checkpoint registry must contain a checkpoints list")
            self._records = [ModelCheckpointRecord.from_json_dict(record) for record in records]

    def all(self) -> tuple[ModelCheckpointRecord, ...]:
        return tuple(self._records)

    def add(self, record: ModelCheckpointRecord) -> None:
        self._records = [row for row in self._records if row.checkpoint_id != record.checkpoint_id]
        self._records.append(record)
        self.save()

    def latest(self, model_type: str | None = None) -> ModelCheckpointRecord | None:
        rows = self._filter(model_type)
        if not rows:
            return None
        return max(rows, key=lambda record: record.created_at)

    def best(
        self,
        *,
        metric: str,
        mode: str = "min",
        model_type: str | None = None,
    ) -> ModelCheckpointRecord | None:
        rows = [record for record in self._filter(model_type) if record.metric(metric) is not None]
        if not rows:
            return None
        if mode == "min":
            return min(rows, key=lambda record: record.metric(metric) or float("inf"))
        if mode == "max":
            return max(rows, key=lambda record: record.metric(metric) or float("-inf"))
        raise ValueError("mode must be 'min' or 'max'")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoints": [record.to_json_dict() for record in self._records]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _filter(self, model_type: str | None) -> tuple[ModelCheckpointRecord, ...]:
        if model_type is None:
            return tuple(self._records)
        return tuple(record for record in self._records if record.model_type == model_type)
