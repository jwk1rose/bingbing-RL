from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from .checkpointing import CheckpointRegistry, ModelCheckpointRecord


@dataclass(frozen=True)
class SplitFileRecord:
    split: str
    path: str
    rows: int
    bytes: int
    sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "path": self.path,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "SplitFileRecord":
        return cls(
            split=str(payload["split"]),
            path=str(payload["path"]),
            rows=int(payload["rows"]),
            bytes=int(payload["bytes"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True)
class DatasetSplitManifest:
    dataset_id: str
    version: str
    created_at: float
    dataset_hash: str
    files: tuple[SplitFileRecord, ...]
    metadata: Mapping[str, object]

    @property
    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.files:
            counts[record.split] = counts.get(record.split, 0) + record.rows
        return counts

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "created_at": self.created_at,
            "dataset_hash": self.dataset_hash,
            "split_counts": self.split_counts,
            "files": [record.to_json_dict() for record in self.files],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "DatasetSplitManifest":
        files = payload.get("files", ())
        if not isinstance(files, Sequence):
            raise ValueError("split manifest files must be a sequence")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("split manifest metadata must be an object")
        return cls(
            dataset_id=str(payload["dataset_id"]),
            version=str(payload["version"]),
            created_at=float(payload["created_at"]),
            dataset_hash=str(payload["dataset_hash"]),
            files=tuple(SplitFileRecord.from_json_dict(item) for item in files if isinstance(item, Mapping)),
            metadata={str(key): value for key, value in metadata.items()},
        )


def build_jsonl_split_manifest(
    splits: Mapping[str, Sequence[str | Path]],
    *,
    dataset_id: str,
    version: str = "unknown",
    metadata: Mapping[str, object] | None = None,
    created_at: float | None = None,
) -> DatasetSplitManifest:
    records: list[SplitFileRecord] = []
    for split, paths in sorted(splits.items(), key=lambda item: item[0]):
        for path in paths:
            records.append(_scan_jsonl_file(str(split), Path(path)))
    dataset_hash = _manifest_dataset_hash(dataset_id=dataset_id, version=version, files=tuple(records), metadata=metadata or {})
    return DatasetSplitManifest(
        dataset_id=dataset_id,
        version=version,
        created_at=time.time() if created_at is None else float(created_at),
        dataset_hash=dataset_hash,
        files=tuple(records),
        metadata=dict(metadata or {}),
    )


def write_split_manifest(path: str | Path, manifest: DatasetSplitManifest) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split_manifest(path: str | Path) -> DatasetSplitManifest:
    return DatasetSplitManifest.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def select_best_checkpoint(
    registry_path: str | Path,
    *,
    metric: str,
    mode: str = "min",
    model_type: str | None = None,
    dataset_hash: str | None = None,
    out_path: str | Path | None = None,
) -> ModelCheckpointRecord:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
    records = [
        record
        for record in CheckpointRegistry(registry_path).all()
        if (model_type is None or record.model_type == model_type)
        and (dataset_hash is None or record.dataset_hash == dataset_hash)
        and record.metric(metric) is not None
    ]
    if not records:
        raise ValueError("no checkpoint matches the selection criteria")
    selected = (min if mode == "min" else max)(records, key=lambda record: record.metric(metric) or 0.0)
    if out_path is not None:
        _write_selection(out_path, selected, metric=metric, mode=mode, model_type=model_type, dataset_hash=dataset_hash)
    return selected


def _scan_jsonl_file(split: str, path: Path) -> SplitFileRecord:
    digest = hashlib.sha256()
    rows = 0
    total_bytes = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            total_bytes += len(line)
            if line.strip():
                rows += 1
    return SplitFileRecord(split=split, path=str(path), rows=rows, bytes=total_bytes, sha256=digest.hexdigest())


def _manifest_dataset_hash(
    *,
    dataset_id: str,
    version: str,
    files: tuple[SplitFileRecord, ...],
    metadata: Mapping[str, object],
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "version": version,
        "files": [record.to_json_dict() for record in files],
        "metadata": dict(metadata),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_selection(
    path: str | Path,
    record: ModelCheckpointRecord,
    *,
    metric: str,
    mode: str,
    model_type: str | None,
    dataset_hash: str | None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection": {
            "metric": metric,
            "mode": mode,
            "model_type": model_type,
            "dataset_hash": dataset_hash,
            "selected_at": time.time(),
        },
        "checkpoint": record.to_json_dict(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
