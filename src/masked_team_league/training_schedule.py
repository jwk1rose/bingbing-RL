from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

from .models import ResultMetadata
from .run_metadata import RunArtifactRef, RunMetadataManifest, hash_generation_config, write_run_metadata_manifest


TRAINING_SCHEDULE_SCHEMA_VERSION = "training_schedule.v1"
TRAINING_RUN_SUMMARY_SCHEMA_VERSION = "training_run_summary.v1"
RECURRING_TRAINING_SCHEDULER_STATE_SCHEMA_VERSION = "recurring_training_scheduler_state.v1"


@dataclass(frozen=True)
class ScheduledTrainingJob:
    job_id: str
    stage: str
    command: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    resources: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", dict(self.resources or {}))

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "ScheduledTrainingJob":
        return cls(
            job_id=str(payload["job_id"]),
            stage=str(payload["stage"]),
            command=tuple(str(item) for item in payload.get("command", ())),
            inputs=tuple(str(item) for item in payload.get("inputs", ())),
            outputs=tuple(str(item) for item in payload.get("outputs", ())),
            depends_on=tuple(str(item) for item in payload.get("depends_on", ())),
            resources=dict(payload.get("resources") or {}),
        )


@dataclass(frozen=True)
class TrainingSchedule:
    schedule_id: str
    root_dir: str
    registry_path: str
    created_at: float
    jobs: tuple[ScheduledTrainingJob, ...]
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRAINING_SCHEDULE_SCHEMA_VERSION,
            "module": "TrainingSchedule",
            "schedule_id": self.schedule_id,
            "root_dir": self.root_dir,
            "registry_path": self.registry_path,
            "created_at": self.created_at,
            "jobs": [job.to_json_dict() for job in self.jobs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "TrainingSchedule":
        jobs = payload.get("jobs", ())
        if not isinstance(jobs, Sequence):
            raise ValueError("training schedule jobs must be a sequence")
        return cls(
            schedule_id=str(payload["schedule_id"]),
            root_dir=str(payload["root_dir"]),
            registry_path=str(payload["registry_path"]),
            created_at=float(payload["created_at"]),
            jobs=tuple(ScheduledTrainingJob.from_json_dict(item) for item in jobs if isinstance(item, Mapping)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrainingJobStatus:
    job_id: str
    status: str
    returncode: int | None
    started_at: float | None
    finished_at: float | None
    stdout_path: str | None = None
    stderr_path: str | None = None
    resource_before: Mapping[str, object] | None = None
    resource_after: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TrainingResourceSnapshot:
    timestamp: float
    cpu_count: int
    loadavg_1m: float
    loadavg_5m: float
    loadavg_15m: float
    memory_total_mb: float
    memory_available_mb: float
    gpu_count: int = 0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    schedule_id: str
    executed: bool
    jobs: tuple[TrainingJobStatus, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRAINING_RUN_SUMMARY_SCHEMA_VERSION,
            "module": "TrainingRunSummary",
            "schedule_id": self.schedule_id,
            "executed": self.executed,
            "jobs": [asdict(job) for job in self.jobs],
        }


@dataclass(frozen=True)
class RecurringTrainingIteration:
    iteration: int
    schedule_id: str
    run_dir: str
    schedule_path: str
    status_path: str
    status: str
    started_at: float
    finished_at: float
    failed_jobs: tuple[str, ...] = ()
    red_line_violations: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecurringTrainingSchedulerState:
    scheduler_id: str
    root_dir: str
    stopped: bool
    stop_reason: str | None
    iterations: tuple[RecurringTrainingIteration, ...]
    next_run_at: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECURRING_TRAINING_SCHEDULER_STATE_SCHEMA_VERSION,
            "module": "RecurringTrainingSchedulerState",
            "scheduler_id": self.scheduler_id,
            "root_dir": self.root_dir,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "next_run_at": self.next_run_at,
            "iterations": [iteration.to_json_dict() for iteration in self.iterations],
        }


def build_v4_training_schedule(
    *,
    schedule_id: str,
    root_dir: str | Path,
    heroes_json: str | Path,
    decoded_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    single_team_train_jsonl: str | Path | None = None,
    single_team_holdout_jsonl: str | Path | None = None,
    belief_train_jsonl: str | Path | None = None,
    belief_holdout_jsonl: str | Path | None = None,
    belief_round_dirs: Sequence[str | Path] | None = None,
    belief_round_holdout_fraction: float = 0.1,
    attack_teacher_jsonl: str | Path | None = None,
    defense_teacher_jsonl: str | Path | None = None,
    mask_teacher_jsonl: str | Path | None = None,
    mask_explanation_round_dirs: Sequence[str | Path] | None = None,
    mask_explanation_validation_report: str | Path | None = None,
    mask_min_hidden_explanation_coverage: float = 0.95,
    belief_real_validation_round_dirs: Sequence[str | Path] | None = None,
    belief_real_distribution_report: str | Path | None = None,
    belief_min_real_coverage: float = 0.50,
    belief_min_mean_real_records: float = 1.0,
    belief_min_mean_real_similarity: float = 0.25,
    belief_max_oracle_alignment_mae: float = 0.35,
    exploiter_training_root: str | Path | None = None,
    exploiter_effectiveness_report: str | Path | None = None,
    exploiter_min_target_coverage: float = 0.95,
    exploiter_min_positive_residual_rate: float = 0.50,
    exploiter_min_trend_delta: float | None = None,
    defense_anti_meta_training_root: str | Path | None = None,
    defense_anti_meta_effectiveness_report: str | Path | None = None,
    defense_anti_meta_min_feedback_coverage: float = 0.95,
    defense_anti_meta_min_positive_residual_rate: float = 0.50,
    defense_anti_meta_min_mean_residual: float = 0.0,
    defense_anti_meta_min_trend_delta: float | None = None,
    learned_exploiter_selfplay_root: str | Path | None = None,
    learned_exploiter_training_root: str | Path | None = None,
    learned_exploiter_validation_report: str | Path | None = None,
    learned_exploiter_min_rounds: int = 2,
    learned_exploiter_min_oracle_requests: int = 1,
    learned_exploiter_require_latest_checkpoints: bool = True,
    learned_exploiter_min_attack_trend_delta: float | None = None,
    learned_exploiter_min_defense_trend_delta: float | None = None,
    attack_oracle_failure_round_dirs: Sequence[str | Path] | None = None,
    attack_oracle_failure_output_jsons: Sequence[str | Path] | None = None,
    attack_oracle_failure_validation_report: str | Path | None = None,
    attack_oracle_min_failure_annotation_coverage: float = 1.0,
    attack_oracle_min_failure_diagnostic_coverage: float = 1.0,
    active_query_feedback_round_dirs: Sequence[str | Path] | None = None,
    active_query_feedback_report: str | Path | None = None,
    active_query_min_matched_coverage: float = 1.0,
    active_query_max_oracle_error_rate: float = 0.0,
    active_query_min_real_query_count: int = 0,
    active_real_dispatch_validation_jsons: Sequence[str | Path] | None = None,
    active_real_dispatch_validation_report: str | Path | None = None,
    active_real_dispatch_min_reports: int = 1,
    active_real_dispatch_min_dispatched_pairs: int = 1,
    active_real_dispatch_min_completion_rate: float = 1.0,
    active_real_feedback_dirs: Sequence[str | Path] | None = None,
    build_real_calibration_samples_jsonl: str | Path | None = None,
    build_real_calibration_samples_report: str | Path | None = None,
    data_engineering_round_dirs: Sequence[str | Path] | None = None,
    data_engineering_validation_report: str | Path | None = None,
    data_engineering_min_metadata_coverage: float = 1.0,
    data_engineering_min_core_table_coverage: float = 1.0,
    data_engineering_min_artifact_hash_coverage: float = 1.0,
    underdog_residual_round_dirs: Sequence[str | Path] | None = None,
    underdog_residual_validation_report: str | Path | None = None,
    underdog_min_attack_residual_coverage: float = 0.95,
    underdog_min_defense_residual_coverage: float = 0.95,
    underdog_min_mean_attack_residual_bonus: float = 0.0,
    underdog_min_mean_defense_residual_bonus: float = 0.0,
    league_health_round_dirs: Sequence[str | Path] | None = None,
    league_selfplay_health_report: str | Path | None = None,
    league_health_min_attack_pool: int = 1,
    league_health_min_defense_pool: int = 1,
    league_health_min_total_clusters: int = 2,
    league_health_min_payoff_density: float = 0.0,
    league_health_required_attack_roles: Sequence[str] = ("main", "exploiter", "underdog"),
    league_health_required_defense_roles: Sequence[str] = ("main", "exploiter", "underdog"),
    league_health_min_active_pool_fraction: float = 0.0,
    league_health_min_new_attack_strength_delta: float | None = None,
    league_health_min_new_defense_strength_delta: float | None = None,
    production_readiness_report_paths: Sequence[str | Path] | None = None,
    production_readiness_include_scheduled_reports: bool = False,
    production_readiness_report: str | Path | None = None,
    production_readiness_required_schema_versions: Sequence[str] | None = None,
    production_readiness_min_clean_report_rate: float = 1.0,
    production_readiness_require_production_ready: bool = True,
    v4_conformance_report_paths: Sequence[str | Path] | None = None,
    v4_conformance_include_scheduled_reports: bool = False,
    v4_conformance_validation_report: str | Path | None = None,
    real_round_dirs: Sequence[str | Path] | None = None,
    real_meta_db_jsonl: str | Path | None = None,
    real_calibration_report: str | Path | None = None,
    real_rank_segment: str = "unknown",
    real_server: str = "oracle_backend",
    real_season: str | None = None,
    real_timestamp: float | None = None,
    drift_baseline_season: str | None = None,
    drift_current_season: str | None = None,
    drift_delta_threshold: float = 0.15,
    drift_min_overlap: float = 0.20,
    real_calibration_validation_samples_jsonl: Sequence[str | Path] | None = None,
    real_calibration_validation_model_json: str | Path | None = None,
    real_calibration_validation_report: str | Path | None = None,
    real_calibration_validation_min_samples: int = 100,
    real_calibration_min_brier_improvement: float = 0.0,
    real_calibration_min_ece_improvement: float = 0.0,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    model_dim: int = 256,
    proposal_model_dim: int = 256,
    heads: int = 8,
    layers: int = 2,
    seed: int = 0,
    camp_group: int = 3,
) -> TrainingSchedule:
    root = Path(root_dir)
    registry = Path(registry_path) if registry_path is not None else root / "checkpoint_registry.json"
    manifests = root / "manifests"
    models = root / "models"
    selections = root / "selections"
    jobs: list[ScheduledTrainingJob] = []
    common = _resource_args(heroes_json=heroes_json, decoded_dir=decoded_dir, camp_group=camp_group)

    if single_team_train_jsonl is not None:
        train = Path(single_team_train_jsonl)
        holdout = Path(single_team_holdout_jsonl) if single_team_holdout_jsonl is not None else None
        manifest = manifests / "single_team_split_manifest.json"
        split_command = [
            sys.executable,
            "scripts/build_split_manifest.py",
            "--split",
            f"train={train}",
            "--dataset-id",
            f"{schedule_id}:single_team",
            "--version",
            schedule_id,
            "--out-json",
            str(manifest),
        ]
        if holdout is not None:
            split_command.extend(["--split", f"holdout={holdout}"])
        jobs.append(
            ScheduledTrainingJob(
                job_id="single-team-split-manifest",
                stage="split_manifest",
                command=tuple(split_command),
                inputs=tuple(str(path) for path in (train, holdout) if path is not None),
                outputs=(str(manifest),),
            )
        )
        value_model = models / "single_team_value.pt"
        value_metrics = models / "single_team_value.metrics.json"
        train_command = [
            sys.executable,
            "scripts/train_single_team_model.py",
            "--samples-jsonl",
            str(train),
            "--out-model",
            str(value_model),
            "--out-metrics",
            str(value_metrics),
            "--registry",
            str(registry),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--model-dim",
            str(model_dim),
            "--heads",
            str(heads),
            "--layers",
            str(layers),
            "--seed",
            str(seed),
            *common,
        ]
        if holdout is not None:
            train_command.extend(["--holdout-jsonl", str(holdout), "--calibrate"])
        train_command.extend(_device_args(device))
        jobs.append(
            ScheduledTrainingJob(
                job_id="single-team-train",
                stage="train_single_team_value",
                command=tuple(train_command),
                inputs=tuple(str(path) for path in (train, holdout, manifest) if path is not None),
                outputs=(str(value_model), str(value_metrics), str(registry)),
                depends_on=("single-team-split-manifest",),
                resources=_resources(device=device),
            )
        )
        jobs.append(_selection_job("single-team-select-best", registry, selections / "single_team_value.json", "single_team_value", "holdout_auc" if holdout else "auc", "max", ("single-team-train",)))

    round_belief_dirs = tuple(Path(path) for path in (belief_round_dirs or ()))
    if round_belief_dirs and belief_train_jsonl is not None:
        raise ValueError("belief_round_dirs and belief_train_jsonl are mutually exclusive")
    belief_dataset_dependency: tuple[str, ...] = ()
    belief_manifest: Path | None = None
    if round_belief_dirs:
        dataset_dir = root / "datasets" / "belief_ranker"
        train = dataset_dir / "belief_ranker_train.jsonl"
        holdout = dataset_dir / "belief_ranker_holdout.jsonl"
        belief_manifest = dataset_dir / "split_manifest.json"
        command = [
            sys.executable,
            "scripts/build_belief_ranker_dataset.py",
            "--out-dir",
            str(dataset_dir),
            "--holdout-fraction",
            str(belief_round_holdout_fraction),
            "--seed",
            str(seed),
            "--dataset-id",
            f"{schedule_id}:belief_ranker_rounds",
        ]
        for round_dir in round_belief_dirs:
            command.extend(["--round-dir", str(round_dir)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="belief-ranker-build-dataset",
                stage="build_belief_ranker_dataset",
                command=tuple(command),
                inputs=tuple(str(path) for path in round_belief_dirs),
                outputs=(str(train), str(holdout), str(belief_manifest)),
            )
        )
        belief_train_jsonl = train
        belief_holdout_jsonl = holdout
        belief_dataset_dependency = ("belief-ranker-build-dataset",)

    if belief_train_jsonl is not None:
        train = Path(belief_train_jsonl)
        holdout = Path(belief_holdout_jsonl) if belief_holdout_jsonl is not None else None
        holdout_for_training = (
            None if round_belief_dirs and float(belief_round_holdout_fraction) <= 0.0 else holdout
        )
        if belief_manifest is None:
            manifest = manifests / "belief_ranker_split_manifest.json"
            command = [
                sys.executable,
                "scripts/build_split_manifest.py",
                "--split",
                f"train={train}",
                "--dataset-id",
                f"{schedule_id}:belief_ranker",
                "--version",
                schedule_id,
                "--out-json",
                str(manifest),
            ]
            if holdout is not None:
                command.extend(["--split", f"holdout={holdout}"])
            jobs.append(
                ScheduledTrainingJob(
                    job_id="belief-ranker-split-manifest",
                    stage="split_manifest",
                    command=tuple(command),
                    inputs=tuple(str(path) for path in (train, holdout) if path is not None),
                    outputs=(str(manifest),),
                )
            )
            belief_dataset_dependency = ("belief-ranker-split-manifest",)
        else:
            manifest = belief_manifest
        checkpoint = models / "belief_ranker.pt"
        train_command = [
            sys.executable,
            "scripts/train_belief_ranker.py",
            "--samples-jsonl",
            str(train),
            "--out-checkpoint",
            str(checkpoint),
            "--registry",
            str(registry),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--model-dim",
            str(model_dim),
            "--seed",
            str(seed),
            *common,
        ]
        if holdout_for_training is not None:
            train_command.extend(["--holdout-jsonl", str(holdout_for_training)])
        train_command.extend(_device_args(device))
        jobs.append(
            ScheduledTrainingJob(
                job_id="belief-ranker-train",
                stage="train_belief_ranker",
                command=tuple(train_command),
                inputs=tuple(str(path) for path in (train, holdout_for_training, manifest) if path is not None),
                outputs=(str(checkpoint), str(checkpoint.with_suffix(".metrics.json")), str(registry)),
                depends_on=belief_dataset_dependency,
                resources=_resources(device=device),
            )
        )
        jobs.append(
            _selection_job(
                "belief-ranker-select-best",
                registry,
                selections / "belief_ranker.json",
                "belief_ranker",
                "holdout_top1_accuracy" if holdout_for_training else "train_top1_accuracy",
                "max",
                ("belief-ranker-train",),
            )
        )

    if attack_teacher_jsonl is not None:
        jobs.extend(
            _proposal_jobs(
                job_prefix="attack-proposal",
                train_script="scripts/train_attack_proposal.py",
                model_type="attack_proposal",
                teacher_jsonl=Path(attack_teacher_jsonl),
                checkpoint=models / "attack_proposal.pt",
                selection_json=selections / "attack_proposal.json",
                registry=registry,
                common_args=common,
                epochs=epochs,
                lr=lr,
                model_dim=proposal_model_dim,
                heads=heads,
                layers=layers,
                seed=seed,
                device=device,
            )
        )

    if defense_teacher_jsonl is not None:
        jobs.extend(
            _proposal_jobs(
                job_prefix="defense-proposal",
                train_script="scripts/train_defense_proposal.py",
                model_type="defense_proposal",
                teacher_jsonl=Path(defense_teacher_jsonl),
                checkpoint=models / "defense_proposal.pt",
                selection_json=selections / "defense_proposal.json",
                registry=registry,
                common_args=common,
                epochs=epochs,
                lr=lr,
                model_dim=proposal_model_dim,
                heads=heads,
                layers=layers,
                seed=seed,
                device=device,
            )
        )

    if mask_teacher_jsonl is not None:
        checkpoint = models / "mask_selection.pt"
        train_job_id = "mask-selection-train"
        train_command = [
            sys.executable,
            "scripts/train_mask_selection.py",
            "--teacher-jsonl",
            str(Path(mask_teacher_jsonl)),
            "--out-checkpoint",
            str(checkpoint),
            "--registry",
            str(registry),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--hidden-dim",
            str(proposal_model_dim),
            "--seed",
            str(seed),
            *_device_args(device),
        ]
        jobs.append(
            ScheduledTrainingJob(
                job_id=train_job_id,
                stage="train_mask_selection",
                command=tuple(train_command),
                inputs=(str(Path(mask_teacher_jsonl)),),
                outputs=(str(checkpoint), str(checkpoint.with_suffix(".metrics.json")), str(registry)),
                resources=_resources(device=device),
            )
        )
        jobs.append(
            _selection_job(
                "mask-selection-select-best",
                registry,
                selections / "mask_selection.json",
                "mask_selection",
                "train_loss",
                "min",
                (train_job_id,),
            )
        )

    if attack_teacher_jsonl is not None or exploiter_training_root is not None:
        report = Path(exploiter_effectiveness_report) if exploiter_effectiveness_report is not None else root / "reports" / "exploiter_effectiveness_report.json"
        command = [
            sys.executable,
            "scripts/report_exploiter_effectiveness.py",
            "--out-report",
            str(report),
            "--min-target-coverage",
            str(exploiter_min_target_coverage),
            "--min-positive-residual-rate",
            str(exploiter_min_positive_residual_rate),
        ]
        inputs: list[str] = []
        if attack_teacher_jsonl is not None:
            command.extend(["--teacher-jsonl", str(Path(attack_teacher_jsonl))])
            inputs.append(str(Path(attack_teacher_jsonl)))
        if exploiter_training_root is not None:
            command.extend(["--training-root", str(Path(exploiter_training_root))])
            inputs.append(str(Path(exploiter_training_root)))
        if exploiter_min_trend_delta is not None:
            command.extend(["--min-trend-delta", str(exploiter_min_trend_delta)])
        depends_on = ("attack-proposal-train",) if any(job.job_id == "attack-proposal-train" for job in jobs) else _terminal_job_ids(jobs)
        jobs.append(
            ScheduledTrainingJob(
                job_id="exploiter-effectiveness-report",
                stage="exploiter_effectiveness_report",
                command=tuple(command),
                inputs=tuple(inputs),
                outputs=(str(report),),
                depends_on=depends_on,
            )
        )

    if defense_teacher_jsonl is not None or defense_anti_meta_training_root is not None:
        report = (
            Path(defense_anti_meta_effectiveness_report)
            if defense_anti_meta_effectiveness_report is not None
            else root / "reports" / "defense_anti_meta_effectiveness_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_defense_anti_meta_effectiveness.py",
            "--out-report",
            str(report),
            "--min-feedback-coverage",
            str(defense_anti_meta_min_feedback_coverage),
            "--min-positive-residual-rate",
            str(defense_anti_meta_min_positive_residual_rate),
            "--min-mean-residual",
            str(defense_anti_meta_min_mean_residual),
        ]
        inputs: list[str] = []
        if defense_teacher_jsonl is not None:
            command.extend(["--teacher-jsonl", str(Path(defense_teacher_jsonl))])
            inputs.append(str(Path(defense_teacher_jsonl)))
        if defense_anti_meta_training_root is not None:
            command.extend(["--training-root", str(Path(defense_anti_meta_training_root))])
            inputs.append(str(Path(defense_anti_meta_training_root)))
        if defense_anti_meta_min_trend_delta is not None:
            command.extend(["--min-trend-delta", str(defense_anti_meta_min_trend_delta)])
        depends_on = ("defense-proposal-train",) if any(job.job_id == "defense-proposal-train" for job in jobs) else _terminal_job_ids(jobs)
        jobs.append(
            ScheduledTrainingJob(
                job_id="defense-anti-meta-effectiveness-report",
                stage="defense_anti_meta_effectiveness_report",
                command=tuple(command),
                inputs=tuple(inputs),
                outputs=(str(report),),
                depends_on=depends_on,
            )
        )

    if learned_exploiter_selfplay_root is not None:
        selfplay_root = Path(learned_exploiter_selfplay_root)
        training_root_for_validation = (
            Path(learned_exploiter_training_root)
            if learned_exploiter_training_root is not None
            else (
                Path(exploiter_training_root)
                if exploiter_training_root is not None
                else Path(defense_anti_meta_training_root)
                if defense_anti_meta_training_root is not None
                else None
            )
        )
        report = (
            Path(learned_exploiter_validation_report)
            if learned_exploiter_validation_report is not None
            else root / "reports" / "learned_exploiter_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_learned_exploiter_validation.py",
            "--selfplay-root",
            str(selfplay_root),
            "--out-report",
            str(report),
            "--min-rounds",
            str(learned_exploiter_min_rounds),
            "--min-oracle-requests",
            str(learned_exploiter_min_oracle_requests),
            "--min-attack-target-coverage",
            str(exploiter_min_target_coverage),
            "--min-attack-positive-residual-rate",
            str(exploiter_min_positive_residual_rate),
            "--min-defense-feedback-coverage",
            str(defense_anti_meta_min_feedback_coverage),
            "--min-defense-positive-residual-rate",
            str(defense_anti_meta_min_positive_residual_rate),
            "--min-defense-mean-residual",
            str(defense_anti_meta_min_mean_residual),
        ]
        inputs = [str(selfplay_root)]
        if training_root_for_validation is not None:
            command.extend(["--training-root", str(training_root_for_validation)])
            inputs.append(str(training_root_for_validation))
        if not learned_exploiter_require_latest_checkpoints:
            command.append("--no-require-latest-checkpoints")
        if learned_exploiter_min_attack_trend_delta is not None:
            command.extend(["--min-attack-trend-delta", str(learned_exploiter_min_attack_trend_delta)])
        if learned_exploiter_min_defense_trend_delta is not None:
            command.extend(["--min-defense-trend-delta", str(learned_exploiter_min_defense_trend_delta)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="learned-exploiter-validation-report",
                stage="learned_exploiter_validation_report",
                command=tuple(command),
                inputs=tuple(inputs),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    attack_failure_round_paths = tuple(Path(path) for path in (attack_oracle_failure_round_dirs or ()))
    attack_failure_output_paths = tuple(Path(path) for path in (attack_oracle_failure_output_jsons or ()))
    if attack_failure_round_paths or attack_failure_output_paths:
        report = (
            Path(attack_oracle_failure_validation_report)
            if attack_oracle_failure_validation_report is not None
            else root / "reports" / "attack_oracle_failure_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_attack_oracle_failure_validation.py",
            "--out-report",
            str(report),
            "--min-failure-annotation-coverage",
            str(attack_oracle_min_failure_annotation_coverage),
            "--min-failure-diagnostic-coverage",
            str(attack_oracle_min_failure_diagnostic_coverage),
        ]
        for path in attack_failure_output_paths:
            command.extend(["--oracle-output-json", str(path)])
        for path in attack_failure_round_paths:
            command.extend(["--round-dir", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="attack-oracle-failure-validation-report",
                stage="attack_oracle_failure_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in (*attack_failure_output_paths, *attack_failure_round_paths)),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    active_query_round_paths = tuple(Path(path) for path in (active_query_feedback_round_dirs or ()))
    if active_query_round_paths:
        report = (
            Path(active_query_feedback_report)
            if active_query_feedback_report is not None
            else root / "reports" / "active_query_feedback_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_active_query_feedback.py",
            "--out-report",
            str(report),
            "--min-matched-query-coverage",
            str(active_query_min_matched_coverage),
            "--max-oracle-result-error-rate",
            str(active_query_max_oracle_error_rate),
            "--min-real-query-count",
            str(active_query_min_real_query_count),
        ]
        for path in active_query_round_paths:
            command.extend(["--round-dir", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="active-query-feedback-report",
                stage="active_query_feedback_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in active_query_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    active_real_dispatch_validation_paths = tuple(Path(path) for path in (active_real_dispatch_validation_jsons or ()))
    if active_real_dispatch_validation_paths:
        report = (
            Path(active_real_dispatch_validation_report)
            if active_real_dispatch_validation_report is not None
            else root / "reports" / "active_real_query_dispatch_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_active_real_query_dispatch_validation.py",
            "--out-report",
            str(report),
            "--min-reports",
            str(active_real_dispatch_min_reports),
            "--min-dispatched-pairs",
            str(active_real_dispatch_min_dispatched_pairs),
            "--min-completion-rate",
            str(active_real_dispatch_min_completion_rate),
        ]
        for path in active_real_dispatch_validation_paths:
            command.extend(["--validation-json", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="active-real-dispatch-validation-report",
                stage="active_real_query_dispatch_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in active_real_dispatch_validation_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    data_engineering_round_paths = tuple(Path(path) for path in (data_engineering_round_dirs or ()))
    if data_engineering_round_paths:
        report = (
            Path(data_engineering_validation_report)
            if data_engineering_validation_report is not None
            else root / "reports" / "data_engineering_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_data_engineering_validation.py",
            "--out-report",
            str(report),
            "--min-metadata-coverage",
            str(data_engineering_min_metadata_coverage),
            "--min-core-table-coverage",
            str(data_engineering_min_core_table_coverage),
            "--min-artifact-hash-coverage",
            str(data_engineering_min_artifact_hash_coverage),
        ]
        for path in data_engineering_round_paths:
            command.extend(["--round-dir", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="data-engineering-validation-report",
                stage="data_engineering_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in data_engineering_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    underdog_residual_round_paths = tuple(Path(path) for path in (underdog_residual_round_dirs or ()))
    if underdog_residual_round_paths:
        report = (
            Path(underdog_residual_validation_report)
            if underdog_residual_validation_report is not None
            else root / "reports" / "underdog_residual_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_underdog_residual_validation.py",
            "--out-report",
            str(report),
            "--min-attack-residual-coverage",
            str(underdog_min_attack_residual_coverage),
            "--min-defense-residual-coverage",
            str(underdog_min_defense_residual_coverage),
            "--min-mean-attack-residual-bonus",
            str(underdog_min_mean_attack_residual_bonus),
            "--min-mean-defense-residual-bonus",
            str(underdog_min_mean_defense_residual_bonus),
        ]
        for path in underdog_residual_round_paths:
            command.extend(["--round-dir", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="underdog-residual-validation-report",
                stage="underdog_residual_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in underdog_residual_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    league_health_round_paths = tuple(Path(path) for path in (league_health_round_dirs or ()))
    if league_health_round_paths:
        report = (
            Path(league_selfplay_health_report)
            if league_selfplay_health_report is not None
            else root / "reports" / "league_selfplay_health_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_league_selfplay_health.py",
            "--out-report",
            str(report),
            "--min-attack-pool",
            str(league_health_min_attack_pool),
            "--min-defense-pool",
            str(league_health_min_defense_pool),
            "--min-total-clusters",
            str(league_health_min_total_clusters),
            "--min-payoff-density",
            str(league_health_min_payoff_density),
            "--min-active-pool-fraction",
            str(league_health_min_active_pool_fraction),
        ]
        for role in league_health_required_attack_roles:
            command.extend(["--required-attack-role", str(role)])
        for role in league_health_required_defense_roles:
            command.extend(["--required-defense-role", str(role)])
        if league_health_min_new_attack_strength_delta is not None:
            command.extend(["--min-new-attack-strength-delta", str(league_health_min_new_attack_strength_delta)])
        if league_health_min_new_defense_strength_delta is not None:
            command.extend(["--min-new-defense-strength-delta", str(league_health_min_new_defense_strength_delta)])
        for round_dir in league_health_round_paths:
            command.extend(["--round-dir", str(round_dir)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="league-selfplay-health-report",
                stage="league_selfplay_health_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in league_health_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    mask_explanation_round_paths = tuple(Path(path) for path in (mask_explanation_round_dirs or ()))
    if mask_explanation_round_paths:
        report = (
            Path(mask_explanation_validation_report)
            if mask_explanation_validation_report is not None
            else root / "reports" / "mask_explanation_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_mask_explanation_validation.py",
            "--out-report",
            str(report),
            "--min-hidden-explanation-coverage",
            str(mask_min_hidden_explanation_coverage),
        ]
        for round_dir in mask_explanation_round_paths:
            command.extend(["--round-dir", str(round_dir)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="mask-explanation-validation-report",
                stage="mask_explanation_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in mask_explanation_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    belief_real_round_paths = tuple(Path(path) for path in (belief_real_validation_round_dirs or ()))
    if belief_real_round_paths:
        report = (
            Path(belief_real_distribution_report)
            if belief_real_distribution_report is not None
            else root / "reports" / "belief_real_distribution_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_belief_real_distribution_validation.py",
            "--out-report",
            str(report),
            "--min-real-coverage",
            str(belief_min_real_coverage),
            "--min-mean-real-records",
            str(belief_min_mean_real_records),
            "--min-mean-real-similarity",
            str(belief_min_mean_real_similarity),
            "--max-oracle-alignment-mae",
            str(belief_max_oracle_alignment_mae),
        ]
        for round_dir in belief_real_round_paths:
            command.extend(["--round-dir", str(round_dir)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="belief-real-distribution-validation-report",
                stage="belief_real_distribution_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in belief_real_round_paths),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    real_round_paths = tuple(Path(path) for path in (real_round_dirs or ()))
    active_real_feedback_paths = tuple(Path(path) for path in (active_real_feedback_dirs or ()))
    if real_round_paths or active_real_feedback_paths:
        if real_meta_db_jsonl is None:
            raise ValueError("real_meta_db_jsonl is required when real calibration input dirs are supplied")
        if real_season is None:
            raise ValueError("real_season is required when real calibration input dirs are supplied")
        db_jsonl = Path(real_meta_db_jsonl)
        report = Path(real_calibration_report) if real_calibration_report is not None else root / "reports" / "real_calibration_report.json"
        command = [sys.executable, "scripts/ingest_real_calibration.py"]
        for round_dir in real_round_paths:
            command.extend(["--round-dir", str(round_dir)])
        for feedback_dir in active_real_feedback_paths:
            command.extend(["--active-real-feedback-dir", str(feedback_dir)])
        command.extend(
            [
                "--db-jsonl",
                str(db_jsonl),
                "--out-report",
                str(report),
                "--rank-segment",
                real_rank_segment,
                "--server",
                real_server,
                "--season",
                real_season,
                "--drift-delta-threshold",
                str(drift_delta_threshold),
                "--drift-min-overlap",
                str(drift_min_overlap),
            ]
        )
        if real_timestamp is not None:
            command.extend(["--timestamp", str(real_timestamp)])
        if drift_baseline_season is not None:
            command.extend(["--drift-baseline-season", drift_baseline_season])
        if drift_current_season is not None:
            command.extend(["--drift-current-season", drift_current_season])
        jobs.append(
            ScheduledTrainingJob(
                job_id="real-calibration-ingest",
                stage="real_calibration",
                command=tuple(command),
                inputs=tuple(str(path) for path in (*real_round_paths, *active_real_feedback_paths)),
                outputs=(str(db_jsonl), str(report)),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    built_real_calibration_sample_jsonl: Path | None = None
    if build_real_calibration_samples_jsonl is not None:
        if not real_round_paths and not active_real_feedback_paths:
            raise ValueError("real round dirs or active real feedback dirs are required to build real calibration samples")
        if real_season is None:
            raise ValueError("real_season is required when building real calibration samples")
        sample_jsonl = Path(build_real_calibration_samples_jsonl)
        sample_report = (
            Path(build_real_calibration_samples_report)
            if build_real_calibration_samples_report is not None
            else root / "reports" / "real_calibration_samples_report.json"
        )
        command = [
            sys.executable,
            "scripts/build_real_calibration_samples.py",
            "--out-jsonl",
            str(sample_jsonl),
            "--out-report",
            str(sample_report),
            "--rank-segment",
            real_rank_segment,
            "--server",
            real_server,
            "--season",
            real_season,
        ]
        for round_dir in real_round_paths:
            command.extend(["--round-dir", str(round_dir)])
        for feedback_dir in active_real_feedback_paths:
            command.extend(["--active-real-feedback-dir", str(feedback_dir)])
        if real_timestamp is not None:
            command.extend(["--timestamp", str(real_timestamp)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="real-calibration-sample-build",
                stage="real_calibration_sample_build",
                command=tuple(command),
                inputs=tuple(str(path) for path in (*real_round_paths, *active_real_feedback_paths)),
                outputs=(str(sample_jsonl), str(sample_report)),
                depends_on=_terminal_job_ids(jobs),
            )
        )
        built_real_calibration_sample_jsonl = sample_jsonl

    real_calibration_validation_sample_paths = tuple(
        Path(path) for path in (real_calibration_validation_samples_jsonl or ())
    )
    if not real_calibration_validation_sample_paths and built_real_calibration_sample_jsonl is not None:
        real_calibration_validation_sample_paths = (built_real_calibration_sample_jsonl,)
    if real_calibration_validation_sample_paths:
        if real_calibration_validation_model_json is None:
            raise ValueError("real_calibration_validation_model_json is required when validation samples are supplied")
        model_json = Path(real_calibration_validation_model_json)
        report = (
            Path(real_calibration_validation_report)
            if real_calibration_validation_report is not None
            else root / "reports" / "real_calibration_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_real_calibration_validation.py",
            "--calibration-json",
            str(model_json),
            "--out-report",
            str(report),
            "--min-samples",
            str(real_calibration_validation_min_samples),
            "--min-brier-improvement",
            str(real_calibration_min_brier_improvement),
            "--min-ece-improvement",
            str(real_calibration_min_ece_improvement),
        ]
        for path in real_calibration_validation_sample_paths:
            command.extend(["--samples-jsonl", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="real-calibration-validation-report",
                stage="real_calibration_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in (*real_calibration_validation_sample_paths, model_json)),
                outputs=(str(report),),
                depends_on=_terminal_job_ids(jobs),
            )
        )

    v4_conformance_paths = _dedupe_paths(
        tuple(Path(path) for path in (v4_conformance_report_paths or ()))
        + (_scheduled_report_output_paths(jobs) if v4_conformance_include_scheduled_reports else ())
    )
    if v4_conformance_paths:
        report = (
            Path(v4_conformance_validation_report)
            if v4_conformance_validation_report is not None
            else root / "reports" / "v4_conformance_validation_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_v4_conformance_validation.py",
            "--out-report",
            str(report),
        ]
        for path in v4_conformance_paths:
            command.extend(["--report-json", str(path)])
        jobs.append(
            ScheduledTrainingJob(
                job_id="v4-conformance-validation-report",
                stage="v4_conformance_validation_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in v4_conformance_paths),
                outputs=(str(report),),
                depends_on=_producer_job_ids_for_outputs(jobs, v4_conformance_paths) or _terminal_job_ids(jobs),
            )
        )

    production_readiness_paths = _dedupe_paths(
        tuple(Path(path) for path in (production_readiness_report_paths or ()))
        + (_scheduled_report_output_paths(jobs) if production_readiness_include_scheduled_reports else ())
    )
    if production_readiness_paths:
        report = (
            Path(production_readiness_report)
            if production_readiness_report is not None
            else root / "reports" / "production_readiness_report.json"
        )
        command = [
            sys.executable,
            "scripts/report_production_readiness.py",
            "--out-report",
            str(report),
            "--min-clean-report-rate",
            str(production_readiness_min_clean_report_rate),
        ]
        for path in production_readiness_paths:
            command.extend(["--report-json", str(path)])
        for schema_version in production_readiness_required_schema_versions or ():
            command.extend(["--required-schema-version", str(schema_version)])
        if not production_readiness_require_production_ready:
            command.append("--no-require-production-ready")
        jobs.append(
            ScheduledTrainingJob(
                job_id="production-readiness-report",
                stage="production_readiness_report",
                command=tuple(command),
                inputs=tuple(str(path) for path in production_readiness_paths),
                outputs=(str(report),),
                depends_on=_producer_job_ids_for_outputs(jobs, production_readiness_paths) or _terminal_job_ids(jobs),
            )
        )

    return TrainingSchedule(
        schedule_id=schedule_id,
        root_dir=str(root),
        registry_path=str(registry),
        created_at=time.time(),
        jobs=tuple(jobs),
        metadata={
            "version": "v4",
            "decoded_dir": None if decoded_dir is None else str(decoded_dir),
            "seed": seed,
            "camp_group": camp_group,
        },
    )


def write_training_schedule(path: str | Path, schedule: TrainingSchedule) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schedule.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_training_schedule(path: str | Path) -> TrainingSchedule:
    return TrainingSchedule.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run_training_schedule(
    schedule: TrainingSchedule,
    *,
    execute: bool,
    cwd: str | Path | None = None,
    status_path: str | Path | None = None,
    monitor_resources: bool = False,
    resource_snapshot_fn: Callable[[], TrainingResourceSnapshot] | None = None,
) -> TrainingRunSummary:
    statuses: list[TrainingJobStatus] = []
    log_dir = Path(schedule.root_dir) / "logs"
    if execute:
        log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_fn = resource_snapshot_fn or collect_training_resource_snapshot
    completed: set[str] = set()
    for job in schedule.jobs:
        missing = [dependency for dependency in job.depends_on if dependency not in completed]
        if missing:
            raise ValueError(f"job {job.job_id} has unmet dependencies: {missing}")
        resource_before = snapshot_fn().to_json_dict() if monitor_resources else None
        if not execute:
            resource_after = snapshot_fn().to_json_dict() if monitor_resources else None
            statuses.append(
                TrainingJobStatus(
                    job_id=job.job_id,
                    status="dry_run",
                    returncode=None,
                    started_at=None,
                    finished_at=None,
                    resource_before=resource_before,
                    resource_after=resource_after,
                )
            )
            completed.add(job.job_id)
            continue
        started_at = time.time()
        result = subprocess.run(job.command, check=False, capture_output=True, text=True, cwd=cwd)
        finished_at = time.time()
        resource_after = snapshot_fn().to_json_dict() if monitor_resources else None
        stdout_path = log_dir / f"{job.job_id}.stdout.log"
        stderr_path = log_dir / f"{job.job_id}.stderr.log"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        status = "completed" if result.returncode == 0 else "failed"
        statuses.append(
            TrainingJobStatus(
                job_id=job.job_id,
                status=status,
                returncode=result.returncode,
                started_at=started_at,
                finished_at=finished_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                resource_before=resource_before,
                resource_after=resource_after,
            )
        )
        if result.returncode != 0:
            break
        completed.add(job.job_id)
    summary = TrainingRunSummary(schedule_id=schedule.schedule_id, executed=execute, jobs=tuple(statuses))
    if status_path is not None:
        output_path = Path(status_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_training_run_metadata(schedule, summary, status_path=status_path)
    return summary


def build_scheduler_red_line_check(
    *,
    extra_report_paths: Sequence[str | Path] = (),
    include_iteration_reports: bool = True,
) -> Callable[[Path, TrainingRunSummary], tuple[str, ...]]:
    report_paths = tuple(Path(path) for path in extra_report_paths)

    def _check(run_dir: Path, _summary: TrainingRunSummary) -> tuple[str, ...]:
        violations: list[str] = []
        for path in report_paths:
            violations.extend(_red_line_violations_from_report(path))
        if include_iteration_reports:
            report_dir = Path(run_dir) / "reports"
            violations.extend(_red_line_violations_from_report(report_dir / "real_calibration_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "real_calibration_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "daily_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "active_query_feedback_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "active_real_query_dispatch_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "data_engineering_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "underdog_residual_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "league_selfplay_health_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "attack_oracle_failure_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "exploiter_effectiveness_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "defense_anti_meta_effectiveness_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "learned_exploiter_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "production_readiness_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "v4_conformance_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "mask_explanation_validation_report.json"))
            violations.extend(_red_line_violations_from_report(report_dir / "belief_real_distribution_validation_report.json"))
        return _dedupe(violations)

    return _check


def _write_training_run_metadata(
    schedule: TrainingSchedule,
    summary: TrainingRunSummary,
    *,
    status_path: str | Path | None,
) -> None:
    root = Path(schedule.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_paths: list[str | Path] = [item for job in schedule.jobs for item in job.inputs]
    output_paths: list[str | Path] = [item for job in schedule.jobs for item in job.outputs]
    if status_path is not None:
        output_paths.append(status_path)
    for job in summary.jobs:
        if job.stdout_path is not None:
            output_paths.append(job.stdout_path)
        if job.stderr_path is not None:
            output_paths.append(job.stderr_path)
    metadata = ResultMetadata(
        model_version=str(schedule.metadata.get("model_version", "none")),
        data_version=str(schedule.metadata.get("data_version", schedule.metadata.get("version", "unknown"))),
        simulator_version=str(schedule.metadata.get("simulator_version", "training_schedule")),
        league_iteration=int(schedule.metadata.get("league_iteration", 0) or 0),
        random_seed=int(schedule.metadata.get("seed", 0) or 0),
        generation_config_hash=hash_generation_config(schedule.to_json_dict()),
        calibration_version=str(schedule.metadata.get("calibration_version", "none")),
    )
    manifest = RunMetadataManifest.from_result_metadata(
        run_id=schedule.schedule_id,
        metadata=metadata,
        created_at=schedule.created_at,
        code_version=str(schedule.metadata.get("code_version", "local")),
        input_artifacts=_artifact_refs(input_paths, role="input"),
        output_artifacts=_artifact_refs(output_paths, role="output"),
        metrics={
            "jobs": len(summary.jobs),
            "completed_jobs": sum(1 for job in summary.jobs if job.status in {"completed", "dry_run"}),
            "failed_jobs": sum(1 for job in summary.jobs if job.status == "failed"),
            "executed": int(summary.executed),
        },
        extra={"runner": "run_training_schedule"},
    )
    write_run_metadata_manifest(manifest, root / "run_metadata.json")


def run_recurring_training_scheduler(
    *,
    scheduler_id: str,
    root_dir: str | Path,
    schedule_factory: Callable[[int, Path], TrainingSchedule],
    iterations: int,
    interval_seconds: float,
    execute: bool,
    cwd: str | Path | None = None,
    state_path: str | Path | None = None,
    monitor_resources: bool = False,
    stop_on_failure: bool = True,
    stop_on_red_line: bool = True,
    red_line_check_fn: Callable[[Path, TrainingRunSummary], Sequence[str]] | None = None,
    sleep_fn: Callable[[float], object] = time.sleep,
    now_fn: Callable[[], float] = time.time,
    run_schedule_fn: Callable[..., TrainingRunSummary] = run_training_schedule,
) -> RecurringTrainingSchedulerState:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if interval_seconds < 0.0:
        raise ValueError("interval_seconds must be non-negative")
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_state_path = Path(state_path) if state_path is not None else root / "scheduler_state.json"
    records: list[RecurringTrainingIteration] = []
    stopped = False
    stop_reason: str | None = None
    next_run_at: float | None = None
    for iteration in range(1, iterations + 1):
        started_at = float(now_fn())
        run_dir = root / f"iteration_{iteration:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        schedule = schedule_factory(iteration, run_dir)
        schedule_path = run_dir / "schedule.json"
        status_path = run_dir / "status.json"
        write_training_schedule(schedule_path, schedule)
        summary = run_schedule_fn(
            schedule,
            execute=execute,
            cwd=cwd,
            status_path=status_path,
            monitor_resources=monitor_resources,
        )
        failed_jobs = tuple(job.job_id for job in summary.jobs if job.status == "failed")
        red_lines = tuple(red_line_check_fn(run_dir, summary)) if red_line_check_fn is not None else ()
        status = "completed"
        if failed_jobs:
            status = "failed"
        elif red_lines:
            status = "red_line"
        finished_at = float(now_fn())
        records.append(
            RecurringTrainingIteration(
                iteration=iteration,
                schedule_id=schedule.schedule_id,
                run_dir=str(run_dir),
                schedule_path=str(schedule_path),
                status_path=str(status_path),
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                failed_jobs=failed_jobs,
                red_line_violations=red_lines,
            )
        )
        if failed_jobs and stop_on_failure:
            stopped = True
            stop_reason = "failed_jobs"
        elif red_lines and stop_on_red_line:
            stopped = True
            stop_reason = "red_line_violations"
        is_last = iteration >= iterations
        next_run_at = None if stopped or is_last else finished_at + float(interval_seconds)
        state = RecurringTrainingSchedulerState(
            scheduler_id=scheduler_id,
            root_dir=str(root),
            stopped=stopped,
            stop_reason=stop_reason,
            iterations=tuple(records),
            next_run_at=next_run_at,
        )
        output_state_path.parent.mkdir(parents=True, exist_ok=True)
        output_state_path.write_text(
            json.dumps(state.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if stopped or is_last:
            return state
        if interval_seconds > 0.0:
            sleep_fn(float(interval_seconds))
    return RecurringTrainingSchedulerState(
        scheduler_id=scheduler_id,
        root_dir=str(root),
        stopped=stopped,
        stop_reason=stop_reason,
        iterations=tuple(records),
        next_run_at=next_run_at,
    )


def collect_training_resource_snapshot(
    *,
    now: float | None = None,
    cpu_count: int | None = None,
    loadavg: tuple[float, float, float] | None = None,
    meminfo_path: str | Path = "/proc/meminfo",
    nvidia_smi_output: str | None = None,
) -> TrainingResourceSnapshot:
    if loadavg is None:
        try:
            loadavg = os.getloadavg()
        except OSError:
            loadavg = (0.0, 0.0, 0.0)
    total_mb, available_mb = _read_meminfo_mb(Path(meminfo_path))
    gpu_count, gpu_used, gpu_total = _gpu_memory_from_csv(nvidia_smi_output)
    if nvidia_smi_output is None and shutil.which("nvidia-smi"):
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            output = None
        if output is not None and output.returncode == 0:
            gpu_count, gpu_used, gpu_total = _gpu_memory_from_csv(output.stdout)
    return TrainingResourceSnapshot(
        timestamp=float(time.time() if now is None else now),
        cpu_count=int(os.cpu_count() or 0 if cpu_count is None else cpu_count),
        loadavg_1m=float(loadavg[0]),
        loadavg_5m=float(loadavg[1]),
        loadavg_15m=float(loadavg[2]),
        memory_total_mb=total_mb,
        memory_available_mb=available_mb,
        gpu_count=gpu_count,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
    )


def _red_line_violations_from_report(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return ()
    violations: list[str] = []
    values = payload.get("red_line_violations", ())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        violations.extend(str(value) for value in values)
    drift = payload.get("drift")
    if isinstance(drift, Mapping) and bool(drift.get("drift_detected")):
        violations.append("real_calibration_drift_detected")
    return tuple(violations)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _artifact_refs(paths: Sequence[str | Path], *, role: str) -> tuple[RunArtifactRef, ...]:
    refs: list[RunArtifactRef] = []
    for item in paths:
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        refs.append(RunArtifactRef.from_path(path, kind=path.suffix.lstrip(".") or "file", role=role))
    return tuple(refs)


def _read_meminfo_mb(path: Path) -> tuple[float, float]:
    values: dict[str, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = float(parts[1]) / 1024.0
    except OSError:
        return 0.0, 0.0
    return values.get("MemTotal", 0.0), values.get("MemAvailable", 0.0)


def _gpu_memory_from_csv(output: str | None) -> tuple[int, float, float]:
    if not output:
        return 0, 0.0, 0.0
    count = 0
    used = 0.0
    total = 0.0
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            used += float(parts[1])
            total += float(parts[2])
            count += 1
        except ValueError:
            continue
    return count, used, total


def _resource_args(*, heroes_json: str | Path, decoded_dir: str | Path | None, camp_group: int) -> list[str]:
    args = ["--heroes-json", str(heroes_json), "--camp-group", str(camp_group)]
    if decoded_dir is not None:
        args.extend(["--decoded-dir", str(decoded_dir)])
    return args


def _device_args(device: str | None) -> list[str]:
    return [] if device is None else ["--device", str(device)]


def _resources(*, device: str | None) -> Mapping[str, object]:
    return {"device": device or "default"}


def _terminal_job_ids(jobs: Sequence[ScheduledTrainingJob]) -> tuple[str, ...]:
    depended_on = {dependency for job in jobs for dependency in job.depends_on}
    return tuple(job.job_id for job in jobs if job.job_id not in depended_on)


def _selection_job(
    job_id: str,
    registry: Path,
    out_json: Path,
    model_type: str,
    metric: str,
    mode: str,
    depends_on: tuple[str, ...],
) -> ScheduledTrainingJob:
    return ScheduledTrainingJob(
        job_id=job_id,
        stage="select_best_checkpoint",
        command=(
            sys.executable,
            "scripts/select_model_checkpoint.py",
            "--registry",
            str(registry),
            "--metric",
            metric,
            "--mode",
            mode,
            "--model-type",
            model_type,
            "--out-json",
            str(out_json),
        ),
        inputs=(str(registry),),
        outputs=(str(out_json),),
        depends_on=depends_on,
    )


def _proposal_jobs(
    *,
    job_prefix: str,
    train_script: str,
    model_type: str,
    teacher_jsonl: Path,
    checkpoint: Path,
    selection_json: Path,
    registry: Path,
    common_args: Sequence[str],
    epochs: int,
    lr: float,
    model_dim: int,
    heads: int,
    layers: int,
    seed: int,
    device: str | None,
) -> tuple[ScheduledTrainingJob, ScheduledTrainingJob]:
    train_job_id = f"{job_prefix}-train"
    command = [
        sys.executable,
        train_script,
        "--teacher-jsonl",
        str(teacher_jsonl),
        "--out-checkpoint",
        str(checkpoint),
        "--registry",
        str(registry),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--model-dim",
        str(model_dim),
        "--heads",
        str(heads),
        "--layers",
        str(layers),
        "--seed",
        str(seed),
        *common_args,
        *_device_args(device),
    ]
    train = ScheduledTrainingJob(
        job_id=train_job_id,
        stage=f"train_{model_type}",
        command=tuple(command),
        inputs=(str(teacher_jsonl),),
        outputs=(str(checkpoint), str(checkpoint.with_suffix(".metrics.json")), str(registry)),
        resources=_resources(device=device),
    )
    select = _selection_job(
        f"{job_prefix}-select-best",
        registry,
        selection_json,
        model_type,
        "train_loss",
        "min",
        (train_job_id,),
    )
    return train, select


def _scheduled_report_output_paths(jobs: Sequence[ScheduledTrainingJob]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for job in jobs:
        if job.job_id == "production-readiness-report":
            continue
        for output in job.outputs:
            path = Path(output)
            is_report_stage = str(job.stage).endswith("_report")
            is_report_artifact = "report" in path.name and path.suffix == ".json"
            if path.suffix == ".json" and (is_report_stage or is_report_artifact):
                paths.append(path)
    return tuple(paths)


def _producer_job_ids_for_outputs(
    jobs: Sequence[ScheduledTrainingJob],
    output_paths: Sequence[str | Path],
) -> tuple[str, ...]:
    wanted = {str(Path(path)) for path in output_paths}
    return tuple(job.job_id for job in jobs if any(str(Path(output)) in wanted for output in job.outputs))


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)
