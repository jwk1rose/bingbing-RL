from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..domain import ResultMetadata


RUN_METADATA_SCHEMA_VERSION = "run_metadata.v1"


@dataclass(frozen=True)
class RunArtifactRef:
    path: str
    kind: str
    role: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: str | Path, *, kind: str, role: str) -> "RunArtifactRef":
        file_path = Path(path)
        return cls(
            path=str(file_path),
            kind=str(kind),
            role=str(role),
            sha256=_sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunArtifactRef":
        return cls(
            path=str(payload["path"]),
            kind=str(payload["kind"]),
            role=str(payload["role"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RunMetadataManifest:
    run_id: str
    created_at: float
    code_version: str
    model_version: str
    data_version: str
    simulator_version: str
    league_iteration: int
    random_seed: int
    generation_config_hash: str
    calibration_version: str
    input_artifacts: tuple[RunArtifactRef, ...] = ()
    output_artifacts: tuple[RunArtifactRef, ...] = ()
    metrics: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    extra: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = RUN_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        object.__setattr__(self, "output_artifacts", tuple(self.output_artifacts))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "extra", {str(key): str(value) for key, value in self.extra.items()})

    @classmethod
    def from_result_metadata(
        cls,
        *,
        run_id: str,
        metadata: ResultMetadata,
        created_at: float,
        code_version: str,
        input_artifacts: tuple[RunArtifactRef, ...] = (),
        output_artifacts: tuple[RunArtifactRef, ...] = (),
        metrics: Mapping[str, float | int | str | bool] | None = None,
        extra: Mapping[str, str] | None = None,
    ) -> "RunMetadataManifest":
        return cls(
            run_id=run_id,
            created_at=float(created_at),
            code_version=code_version,
            model_version=metadata.model_version,
            data_version=metadata.data_version,
            simulator_version=metadata.simulator_version,
            league_iteration=int(metadata.league_iteration),
            random_seed=int(metadata.random_seed),
            generation_config_hash=metadata.generation_config_hash,
            calibration_version=metadata.calibration_version,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            metrics=dict(metrics or {}),
            extra=dict(extra or {}),
        )

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "RunMetadataManifest":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != RUN_METADATA_SCHEMA_VERSION:
            raise ValueError(f"unsupported run metadata schema_version: {schema_version}")
        return cls(
            run_id=str(payload["run_id"]),
            created_at=float(payload["created_at"]),
            code_version=str(payload["code_version"]),
            model_version=str(payload["model_version"]),
            data_version=str(payload["data_version"]),
            simulator_version=str(payload["simulator_version"]),
            league_iteration=int(payload["league_iteration"]),
            random_seed=int(payload["random_seed"]),
            generation_config_hash=str(payload["generation_config_hash"]),
            calibration_version=str(payload["calibration_version"]),
            input_artifacts=tuple(RunArtifactRef.from_dict(item) for item in payload.get("input_artifacts", ())),
            output_artifacts=tuple(RunArtifactRef.from_dict(item) for item in payload.get("output_artifacts", ())),
            metrics=dict(payload.get("metrics") or {}),
            extra={str(key): str(value) for key, value in (payload.get("extra") or {}).items()},
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_version": self.code_version,
            "model_version": self.model_version,
            "data_version": self.data_version,
            "simulator_version": self.simulator_version,
            "league_iteration": self.league_iteration,
            "random_seed": self.random_seed,
            "generation_config_hash": self.generation_config_hash,
            "calibration_version": self.calibration_version,
            "input_artifacts": [artifact.to_json_dict() for artifact in self.input_artifacts],
            "output_artifacts": [artifact.to_json_dict() for artifact in self.output_artifacts],
            "metrics": dict(sorted(self.metrics.items())),
            "extra": dict(sorted(self.extra.items())),
        }


def hash_generation_config(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical_json(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_run_metadata_manifest(manifest: RunMetadataManifest, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_run_metadata_manifest(path: str | Path) -> RunMetadataManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run metadata manifest must be a JSON object")
    return RunMetadataManifest.from_json_dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_json(item) for item in value)
    return value
