from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .active_feedback import dispatch_active_real_queries
from ..constraints import ConstraintEngine
from ..domain import Loadout
from ..generation.proposal_networks import AttackGenerationNetwork, DefenseRosterGenerationNetwork, ProposalNetworkConfig
from ..generation.proposal_training import (
    load_defense_teacher_samples_jsonl,
    load_attack_teacher_samples_jsonl,
    save_proposal_network_checkpoint,
    train_proposal_network,
)
from ..real_platform.oracle import OracleBatchEvaluator
from ..reporting.validation_reports import build_learned_exploiter_validation_report
from .manager import LeagueManager
from .round_runner import LeagueRoundConfig, LeagueRoundRunner


LearnedValidationBuilder = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class SelfPlayOrchestratorConfig:
    rounds: int
    root_dir: Path
    training_dir: Path
    round_config: LeagueRoundConfig
    start_round: int = 1
    proposal_epochs: int = 1
    proposal_lr: float = 1e-3
    proposal_model_dim: int = 256
    proposal_heads: int = 8
    proposal_layers: int = 2
    proposal_device: str | None = None
    attack_proposal_beam_size: int = 8
    defense_proposal_beam_size: int = 8
    candidate_weight_temperature: float = 0.25
    min_candidate_weight: float = 0.05
    dispatch_real_queries: bool = True
    real_query_seed_offset: int = 100_000
    validate_after_each_round: bool = False
    stop_when_validation_ready: bool = False
    validation_min_rounds: int = 2
    validation_min_oracle_requests: int = 1
    validation_require_latest_checkpoints: bool = True
    validation_min_attack_target_coverage: float = 0.95
    validation_min_attack_positive_residual_rate: float = 0.50
    validation_min_attack_trend_delta: float | None = None
    validation_min_defense_feedback_coverage: float = 0.95
    validation_min_defense_positive_residual_rate: float = 0.50
    validation_min_defense_mean_residual: float = 0.0
    validation_min_defense_trend_delta: float | None = None


@dataclass(frozen=True)
class SelfPlayRoundRecord:
    round_id: str
    round_dir: str
    training_dir: str
    candidates: int
    oracle_pairs: int
    oracle_requests: int
    attack_proposal_checkpoint: str | None
    defense_proposal_checkpoint: str | None
    teacher_jsonl: str | None
    defense_teacher_jsonl: str | None
    real_query_feedback_dir: str | None
    real_query_attack_teacher_jsonl: str | None
    real_query_defense_teacher_jsonl: str | None
    real_query_pairs: int
    train_loss: float | None
    defense_train_loss: float | None
    validation_report: str | None = None
    validation_production_ready: bool | None = None
    validation_red_line_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelfPlayValidationRecord:
    round_id: str
    report_path: str
    production_ready: bool
    red_line_violations: tuple[str, ...]


@dataclass(frozen=True)
class SelfPlayRunSummary:
    root_dir: str
    training_dir: str
    rounds: tuple[SelfPlayRoundRecord, ...]
    latest_attack_proposal_checkpoint: str | None
    latest_defense_proposal_checkpoint: str | None
    validation_reports: tuple[SelfPlayValidationRecord, ...] = ()
    production_ready: bool = False
    stop_reason: str | None = None


class SelfPlayOrchestrator:
    def __init__(
        self,
        *,
        loadout_pool: tuple[Loadout, ...],
        evaluator: OracleBatchEvaluator,
        config: SelfPlayOrchestratorConfig,
        league: LeagueManager | None = None,
        learned_validation_builder: LearnedValidationBuilder | None = None,
    ) -> None:
        if config.rounds <= 0:
            raise ValueError("rounds must be positive")
        self.loadout_pool = tuple(loadout_pool)
        self.evaluator = evaluator
        self.config = config
        self.league = league or LeagueManager()
        self.engine = ConstraintEngine(self.loadout_pool)
        self.learned_validation_builder = learned_validation_builder or build_learned_exploiter_validation_report

    def run(self) -> SelfPlayRunSummary:
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.config.training_dir.mkdir(parents=True, exist_ok=True)
        records: list[SelfPlayRoundRecord] = []
        validation_records: list[SelfPlayValidationRecord] = []
        latest_attack_checkpoint: Path | None = None
        latest_defense_checkpoint: Path | None = None
        production_ready = False
        stop_reason: str | None = None
        for offset in range(self.config.rounds):
            round_number = self.config.start_round + offset
            round_id = f"round_{round_number:04d}"
            round_dir = self.config.root_dir / round_id
            round_config = replace(
                self.config.round_config,
                round_id=round_id,
                seed=self.config.round_config.seed + offset,
                attack_proposal_checkpoint=latest_attack_checkpoint,
                attack_proposal_beam_size=self.config.attack_proposal_beam_size,
                attack_proposal_device=self.config.proposal_device,
                defense_proposal_checkpoint=latest_defense_checkpoint,
                defense_proposal_beam_size=self.config.defense_proposal_beam_size,
                defense_proposal_device=self.config.proposal_device,
            )
            round_summary = LeagueRoundRunner(
                loadout_pool=self.loadout_pool,
                evaluator=self.evaluator,
                league=self.league,
                config=round_config,
            ).run(round_dir)
            training_dir = self.config.training_dir / round_id
            training_dir.mkdir(parents=True, exist_ok=True)
            teacher_jsonl = training_dir / "attack_teacher.jsonl"
            teacher_rows = build_attack_teacher_jsonl_from_round(round_dir, teacher_jsonl)
            defense_teacher_jsonl = training_dir / "defense_teacher.jsonl"
            defense_teacher_rows = build_defense_teacher_jsonl_from_round(round_dir, defense_teacher_jsonl)
            real_feedback_dir: Path | None = None
            real_attack_teacher_jsonl: Path | None = None
            real_defense_teacher_jsonl: Path | None = None
            real_query_pairs = 0
            if self.config.dispatch_real_queries and round_config.active_real_keep > 0:
                real_feedback_dir = training_dir / "real_queries"
                real_summary = dispatch_active_real_queries(
                    round_dir,
                    real_feedback_dir,
                    evaluator=self.evaluator,
                    job_prefix=f"{round_id}-realq",
                    base_seed=self.config.round_config.seed + self.config.real_query_seed_offset + offset,
                )
                real_query_pairs = real_summary.dispatched_pairs
                real_attack_teacher_jsonl = real_feedback_dir / "attack_teacher.jsonl"
                real_defense_teacher_jsonl = real_feedback_dir / "defense_teacher.jsonl"
                teacher_rows += _append_jsonl_file(real_attack_teacher_jsonl, teacher_jsonl)
                defense_teacher_rows += _append_jsonl_file(real_defense_teacher_jsonl, defense_teacher_jsonl)
            attack_checkpoint: Path | None = None
            defense_checkpoint: Path | None = None
            train_loss: float | None = None
            defense_train_loss: float | None = None
            if teacher_rows:
                attack_checkpoint, train_loss = self._train_attack_proposal(teacher_jsonl, training_dir, round_id)
                latest_attack_checkpoint = attack_checkpoint
            if defense_teacher_rows:
                defense_checkpoint, defense_train_loss = self._train_defense_proposal(
                    defense_teacher_jsonl,
                    training_dir,
                    round_id,
                )
                latest_defense_checkpoint = defense_checkpoint
            record = SelfPlayRoundRecord(
                round_id=round_id,
                round_dir=str(round_dir),
                training_dir=str(training_dir),
                candidates=round_summary.candidates,
                oracle_pairs=round_summary.oracle_pairs,
                oracle_requests=round_summary.oracle_requests,
                attack_proposal_checkpoint=None if attack_checkpoint is None else str(attack_checkpoint),
                defense_proposal_checkpoint=None if defense_checkpoint is None else str(defense_checkpoint),
                teacher_jsonl=str(teacher_jsonl) if teacher_rows else None,
                defense_teacher_jsonl=str(defense_teacher_jsonl) if defense_teacher_rows else None,
                real_query_feedback_dir=None if real_feedback_dir is None else str(real_feedback_dir),
                real_query_attack_teacher_jsonl=None if real_attack_teacher_jsonl is None else str(real_attack_teacher_jsonl),
                real_query_defense_teacher_jsonl=None if real_defense_teacher_jsonl is None else str(real_defense_teacher_jsonl),
                real_query_pairs=real_query_pairs,
                train_loss=train_loss,
                defense_train_loss=defense_train_loss,
            )
            records.append(record)
            self._write_state(
                records,
                latest_attack_checkpoint,
                latest_defense_checkpoint,
                validation_records=validation_records,
                production_ready=production_ready,
                stop_reason=stop_reason,
            )
            if self.config.validate_after_each_round:
                validation = self._write_validation_report(round_dir)
                validation_records.append(validation)
                record = replace(
                    record,
                    validation_report=validation.report_path,
                    validation_production_ready=validation.production_ready,
                    validation_red_line_violations=validation.red_line_violations,
                )
                records[-1] = record
                production_ready = validation.production_ready
                if validation.production_ready and self.config.stop_when_validation_ready:
                    stop_reason = "validation_ready"
                    self._write_state(
                        records,
                        latest_attack_checkpoint,
                        latest_defense_checkpoint,
                        validation_records=validation_records,
                        production_ready=production_ready,
                        stop_reason=stop_reason,
                    )
                    break
            self._write_state(
                records,
                latest_attack_checkpoint,
                latest_defense_checkpoint,
                validation_records=validation_records,
                production_ready=production_ready,
                stop_reason=stop_reason,
            )
        return SelfPlayRunSummary(
            root_dir=str(self.config.root_dir),
            training_dir=str(self.config.training_dir),
            rounds=tuple(records),
            latest_attack_proposal_checkpoint=None if latest_attack_checkpoint is None else str(latest_attack_checkpoint),
            latest_defense_proposal_checkpoint=None if latest_defense_checkpoint is None else str(latest_defense_checkpoint),
            validation_reports=tuple(validation_records),
            production_ready=production_ready,
            stop_reason=stop_reason,
        )

    def _train_attack_proposal(self, teacher_jsonl: Path, training_dir: Path, round_id: str) -> tuple[Path, float]:
        samples = load_attack_teacher_samples_jsonl(
            teacher_jsonl,
            loadout_pool=self.loadout_pool,
            constraint_engine=self.engine,
            candidate_weight_temperature=self.config.candidate_weight_temperature,
            min_candidate_weight=self.config.min_candidate_weight,
        )
        network = AttackGenerationNetwork(
            ProposalNetworkConfig(
                loadout_count=len(self.loadout_pool),
                model_dim=self.config.proposal_model_dim,
                heads=self.config.proposal_heads,
                layers=self.config.proposal_layers,
                max_slots=self.config.round_config.teams * 5,
            )
        )
        history = train_proposal_network(
            network,
            samples,
            epochs=self.config.proposal_epochs,
            lr=self.config.proposal_lr,
            device=self.config.proposal_device,
            seed=self.config.round_config.seed,
        )
        checkpoint = training_dir / "attack_proposal.pt"
        save_proposal_network_checkpoint(
            checkpoint,
            network,
            history,
            registry_path=self.config.training_dir / "checkpoint_registry.json",
            checkpoint_id=f"attack-proposal-{round_id}",
            dataset_hash=_file_sha256(teacher_jsonl),
            metadata={"teacher_jsonl": str(teacher_jsonl), "round_id": round_id, "samples": len(samples)},
        )
        return checkpoint, float(history.train_losses[-1]) if history.train_losses else 0.0

    def _train_defense_proposal(self, teacher_jsonl: Path, training_dir: Path, round_id: str) -> tuple[Path, float]:
        samples = load_defense_teacher_samples_jsonl(
            teacher_jsonl,
            loadout_pool=self.loadout_pool,
            constraint_engine=self.engine,
            candidate_weight_temperature=self.config.candidate_weight_temperature,
            min_candidate_weight=self.config.min_candidate_weight,
        )
        network = DefenseRosterGenerationNetwork(
            ProposalNetworkConfig(
                loadout_count=len(self.loadout_pool),
                model_dim=self.config.proposal_model_dim,
                heads=self.config.proposal_heads,
                layers=self.config.proposal_layers,
                max_slots=self.config.round_config.teams * 5,
            )
        )
        history = train_proposal_network(
            network,
            samples,
            epochs=self.config.proposal_epochs,
            lr=self.config.proposal_lr,
            device=self.config.proposal_device,
            seed=self.config.round_config.seed,
        )
        checkpoint = training_dir / "defense_proposal.pt"
        save_proposal_network_checkpoint(
            checkpoint,
            network,
            history,
            registry_path=self.config.training_dir / "checkpoint_registry.json",
            checkpoint_id=f"defense-proposal-{round_id}",
            dataset_hash=_file_sha256(teacher_jsonl),
            model_type="defense_proposal",
            metadata={"teacher_jsonl": str(teacher_jsonl), "round_id": round_id, "samples": len(samples)},
        )
        return checkpoint, float(history.train_losses[-1]) if history.train_losses else 0.0

    def _write_state(
        self,
        records: list[SelfPlayRoundRecord],
        latest_attack_checkpoint: Path | None,
        latest_defense_checkpoint: Path | None,
        *,
        validation_records: list[SelfPlayValidationRecord] | None = None,
        production_ready: bool = False,
        stop_reason: str | None = None,
    ) -> None:
        summary = SelfPlayRunSummary(
            root_dir=str(self.config.root_dir),
            training_dir=str(self.config.training_dir),
            rounds=tuple(records),
            latest_attack_proposal_checkpoint=None if latest_attack_checkpoint is None else str(latest_attack_checkpoint),
            latest_defense_proposal_checkpoint=None if latest_defense_checkpoint is None else str(latest_defense_checkpoint),
            validation_reports=tuple(validation_records or ()),
            production_ready=production_ready,
            stop_reason=stop_reason,
        )
        (self.config.root_dir / "orchestrator_state.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_validation_report(self, round_dir: Path) -> SelfPlayValidationRecord:
        report = dict(
            self.learned_validation_builder(
                selfplay_root=self.config.root_dir,
                training_root=self.config.training_dir,
                min_rounds=self.config.validation_min_rounds,
                min_oracle_requests=self.config.validation_min_oracle_requests,
                require_latest_checkpoints=self.config.validation_require_latest_checkpoints,
                min_attack_target_coverage=self.config.validation_min_attack_target_coverage,
                min_attack_positive_residual_rate=self.config.validation_min_attack_positive_residual_rate,
                min_attack_trend_delta=self.config.validation_min_attack_trend_delta,
                min_defense_feedback_coverage=self.config.validation_min_defense_feedback_coverage,
                min_defense_positive_residual_rate=self.config.validation_min_defense_positive_residual_rate,
                min_defense_mean_residual=self.config.validation_min_defense_mean_residual,
                min_defense_trend_delta=self.config.validation_min_defense_trend_delta,
            )
        )
        report_path = round_dir / "learned_exploiter_validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return SelfPlayValidationRecord(
            round_id=round_dir.name,
            report_path=str(report_path),
            production_ready=bool(report.get("production_ready", False)),
            red_line_violations=tuple(str(value) for value in report.get("red_line_violations", ()) or ()),
        )


def build_attack_teacher_jsonl_from_round(round_dir: str | Path, out_path: str | Path) -> int:
    round_path = Path(round_dir)
    candidate_path = round_path / "candidates.jsonl"
    scored_attack_path = round_path / "scored_attacks.jsonl"
    strength_by_attack_id: dict[str, float] = {}
    strength_by_pair: dict[tuple[str, str], float] = {}
    if scored_attack_path.exists():
        with scored_attack_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                attack_id = str(row.get("attack_id"))
                strength = float(row.get("strength", 0.0))
                strength_by_attack_id[attack_id] = strength
                if row.get("defense_id") is not None:
                    strength_by_pair[(attack_id, str(row.get("defense_id")))] = strength
    defense_targets = _load_attack_teacher_defense_targets(round_path / "scored_defenses.jsonl")
    candidate_rows = _read_jsonl(candidate_path)
    main_baseline_by_defense = _main_attack_baseline_by_defense(
        candidate_rows,
        strength_by_attack_id=strength_by_attack_id,
        strength_by_pair=strength_by_pair,
    )
    rows = []
    for row in candidate_rows:
        attack_plan = row.get("attack_plan")
        if attack_plan is None:
            continue
        attack_id = str(row.get("attack_id"))
        defense_id = str(row.get("defense_id"))
        attack_success = strength_by_pair.get(
            (attack_id, defense_id),
            strength_by_attack_id.get(attack_id, float(row.get("surrogate_score") or 0.0)),
        )
        target = defense_targets.get(defense_id, {})
        target_baseline_break_rate = main_baseline_by_defense.get(
            defense_id,
            _float_or(target.get("break_rate"), 0.0),
        )
        exploiter_residual = round(float(attack_success) - target_baseline_break_rate, 12)
        attack_role = str(row.get("attack_role") or "main")
        role_weight = _attack_role_weight(attack_role, exploiter_residual)
        rows.append(
            {
                "teacher_group_id": f"{row.get('round_id')}:{row.get('defense_id')}",
                "defense_id": defense_id,
                "attack_id": attack_id,
                "attack_role": attack_role,
                "rank": row.get("rank"),
                "attack_plan": attack_plan,
                "attack_success": attack_success,
                "gap_target": float(row.get("belief_top1_top2_gap") or 0.0),
                "target_defense_id": defense_id,
                "target_defense_hash": target.get("defense_hash", row.get("defense_hash")),
                "target_defense_strength": target.get("strength"),
                "target_baseline_break_rate": target_baseline_break_rate,
                "exploiter_residual_target": exploiter_residual,
                "role_weight": role_weight,
                "source": "selfplay_orchestrator",
            }
        )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _main_attack_baseline_by_defense(
    candidate_rows: list[dict[str, Any]],
    *,
    strength_by_attack_id: Mapping[str, float],
    strength_by_pair: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for row in candidate_rows:
        if str(row.get("attack_role") or "main") != "main":
            continue
        attack_id = str(row.get("attack_id"))
        defense_id = str(row.get("defense_id"))
        strength = strength_by_pair.get(
            (attack_id, defense_id),
            strength_by_attack_id.get(attack_id, float(row.get("surrogate_score") or 0.0)),
        )
        baselines[defense_id] = max(float(strength), baselines.get(defense_id, 0.0))
    return baselines


def _load_attack_teacher_defense_targets(path: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return targets
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            defense_id = row.get("defense_id")
            if defense_id is None:
                continue
            strength = _float_or(row.get("strength"), 1.0 - _float_or(row.get("break_rate"), 1.0))
            break_rate = _float_or(row.get("break_rate"), 1.0 - strength)
            targets[str(defense_id)] = {
                "defense_hash": row.get("defense_hash"),
                "strength": strength,
                "break_rate": break_rate,
            }
    return targets


def _attack_role_weight(role: str, exploiter_residual: float) -> float:
    role_name = str(role)
    if role_name == "exploiter":
        return round(1.25 + max(0.0, float(exploiter_residual)), 12)
    if role_name == "underdog":
        return round(1.35 + max(0.0, float(exploiter_residual)), 12)
    return 1.0


def build_defense_teacher_jsonl_from_round(round_dir: str | Path, out_path: str | Path) -> int:
    round_path = Path(round_dir)
    scored_defense_path = round_path / "scored_defenses.jsonl"
    rows = []
    if scored_defense_path.exists():
        with scored_defense_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                defense_plan = row.get("defense_plan")
                if defense_plan is None:
                    continue
                targets = _defense_teacher_targets(row)
                rows.append(
                    {
                        "teacher_group_id": f"{row.get('round_id')}:{row.get('defense_role')}",
                        "defense_id": row.get("defense_id"),
                        "defense_role": row.get("defense_role"),
                        "defense_plan": defense_plan,
                        "strength": float(row.get("strength", targets["survival_rate"])),
                        "break_rate": targets["break_rate"],
                        "risk_estimated_break_rate": targets["risk_estimated_break_rate"],
                        "value_target": targets["survival_rate"],
                        "survival_rate": targets["survival_rate"],
                        "meta_attack_success": targets["meta_attack_success"],
                        "anti_meta_residual_target": targets["anti_meta_residual_target"],
                        "gap_target": float(row.get("gap_target", row.get("ambiguity_score", 0.0))),
                        "ambiguity_score": float(row.get("ambiguity_score") or 0.0),
                        "source": "selfplay_orchestrator",
                    }
                )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _defense_teacher_targets(row: Mapping[str, Any]) -> dict[str, float]:
    risk_report = row.get("defense_risk_report")
    risk = risk_report if isinstance(risk_report, Mapping) else {}
    break_rate = _float_or(row.get("break_rate"), _float_or(risk.get("estimated_break_rate"), 1.0))
    estimated_break = _float_or(risk.get("estimated_break_rate"), break_rate)
    survival_rate = _float_or(row.get("survival_rate"), _float_or(risk.get("estimated_survival_rate"), 1.0 - break_rate))
    meta_attack_success = _float_or(row.get("meta_attack_success"), _float_or(risk.get("meta_attack_success"), break_rate))
    anti_meta_residual = round(_float_or(row.get("anti_meta_residual_target"), survival_rate - meta_attack_success), 12)
    return {
        "break_rate": break_rate,
        "risk_estimated_break_rate": estimated_break,
        "survival_rate": survival_rate,
        "meta_attack_success": meta_attack_success,
        "anti_meta_residual_target": anti_meta_residual,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} must contain JSON object lines")
            rows.append(payload)
    return rows


def _float_or(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(fallback)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl_file(source: Path, target: Path) -> int:
    if not source.exists():
        return 0
    rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")
    return len(rows)


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
