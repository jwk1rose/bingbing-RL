from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

from .backend_codec import result_to_attack_win_rate
from .evaluation import match_win_probability
from .metrics import brier_score, expected_calibration_error
from .models import AttackPlan, DefensePlan, Loadout, MatchFormat, Observation, Team, VisibleSlot, observe_defense


REAL_CALIBRATION_INGESTION_SUMMARY_SCHEMA_VERSION = "real_calibration_ingestion_summary.v1"
REAL_CALIBRATION_SAMPLE_BUILD_SUMMARY_SCHEMA_VERSION = "real_calibration_sample_build_summary.v1"
REAL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION = "real_calibration_validation_report.v1"
VERSION_DRIFT_REPORT_SCHEMA_VERSION = "version_drift_report.v1"


@dataclass(frozen=True)
class RealMetaRecord:
    observation_hash: str
    visible_slots: tuple[tuple[VisibleSlot, ...], ...]
    hidden_slots: tuple[tuple[int, int], ...]
    full_defense_if_available: DefensePlan | None
    attack_plan: AttackPlan
    lane_results: tuple[float, ...]
    match_result: float
    unique_equip_stars: tuple[tuple[int, int], ...]
    rank_segment: str
    server: str
    season: str
    timestamp: float

    @classmethod
    def from_match(
        cls,
        *,
        observation: Observation,
        full_defense_if_available: DefensePlan | None,
        attack_plan: AttackPlan,
        lane_results: tuple[float, ...],
        match_result: float,
        rank_segment: str,
        server: str,
        season: str,
        timestamp: float,
    ) -> "RealMetaRecord":
        return cls(
            observation_hash=observation.hash(),
            visible_slots=observation.slots,
            hidden_slots=observation.hidden_slots,
            full_defense_if_available=full_defense_if_available,
            attack_plan=attack_plan,
            lane_results=tuple(float(value) for value in lane_results),
            match_result=float(match_result),
            unique_equip_stars=tuple(sorted(observation.visible_unique_equip_stars)),
            rank_segment=rank_segment,
            server=server,
            season=season,
            timestamp=float(timestamp),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealMetaRecord":
        return cls(
            observation_hash=str(data["observation_hash"]),
            visible_slots=tuple(tuple(_visible_slot_from_dict(slot) for slot in row) for row in data["visible_slots"]),
            hidden_slots=tuple((int(slot[0]), int(slot[1])) for slot in data["hidden_slots"]),
            full_defense_if_available=(
                None
                if data.get("full_defense_if_available") is None
                else _defense_plan_from_dict(data["full_defense_if_available"])
            ),
            attack_plan=_attack_plan_from_dict(data["attack_plan"]),
            lane_results=tuple(float(value) for value in data["lane_results"]),
            match_result=float(data["match_result"]),
            unique_equip_stars=tuple((int(pair[0]), int(pair[1])) for pair in data["unique_equip_stars"]),
            rank_segment=str(data["rank_segment"]),
            server=str(data["server"]),
            season=str(data["season"]),
            timestamp=float(data["timestamp"]),
        )


@dataclass(frozen=True)
class RealMetaObservationMatch:
    record: RealMetaRecord
    similarity: float
    visible_overlap: float
    hidden_overlap: float


class RealMetaDB:
    def __init__(self, path: str | Path | None = None, *, load_existing: bool = True) -> None:
        self.path = None if path is None else Path(path)
        self._records: list[RealMetaRecord] = []
        self._by_observation_hash: dict[str, list[RealMetaRecord]] = {}
        if self.path is not None and load_existing and self.path.exists():
            for record in self._read_jsonl(self.path):
                self._add_in_memory(record)

    def add(self, record: RealMetaRecord) -> None:
        self._add_in_memory(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")

    def _add_in_memory(self, record: RealMetaRecord) -> None:
        self._records.append(record)
        self._by_observation_hash.setdefault(record.observation_hash, []).append(record)

    def all(self) -> tuple[RealMetaRecord, ...]:
        return tuple(self._records)

    def by_observation_hash(self, observation_hash: str) -> tuple[RealMetaRecord, ...]:
        return tuple(self._by_observation_hash.get(observation_hash, ()))

    def by_season(self, season: str) -> tuple[RealMetaRecord, ...]:
        return tuple(record for record in self._records if record.season == season)

    def similar_observations(
        self,
        observation: Observation,
        *,
        min_similarity: float = 0.25,
        max_records: int | None = None,
        include_exact: bool = False,
    ) -> tuple[RealMetaObservationMatch, ...]:
        matches: list[RealMetaObservationMatch] = []
        for record in self._records:
            if not include_exact and record.observation_hash == observation.hash():
                continue
            similarity, visible_overlap, hidden_overlap = real_meta_observation_similarity(observation, record)
            if similarity < float(min_similarity):
                continue
            matches.append(
                RealMetaObservationMatch(
                    record=record,
                    similarity=similarity,
                    visible_overlap=visible_overlap,
                    hidden_overlap=hidden_overlap,
                )
            )
        matches.sort(key=lambda item: (item.similarity, item.record.timestamp), reverse=True)
        if max_records is not None:
            return tuple(matches[: max(0, int(max_records))])
        return tuple(matches)

    @classmethod
    def load(cls, path: str | Path) -> "RealMetaDB":
        return cls(path=path, load_existing=True)

    @staticmethod
    def _read_jsonl(path: Path) -> tuple[RealMetaRecord, ...]:
        records: list[RealMetaRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(RealMetaRecord.from_dict(json.loads(stripped)))
        return tuple(records)


def real_meta_observation_similarity(observation: Observation, record: RealMetaRecord) -> tuple[float, float, float]:
    if observation.format.n_teams != len(record.visible_slots):
        return 0.0, 0.0, 0.0
    query_slots = _visible_slot_signature(observation.slots)
    record_slots = _visible_slot_signature(record.visible_slots)
    visible_overlap = _set_overlap(query_slots, record_slots)
    hero_overlap = _set_overlap(set(observation.visible_heroes), {slot.hero_id for row in record.visible_slots for slot in row if not slot.is_hidden and slot.hero_id is not None})
    equip_overlap = _set_overlap(
        set(observation.visible_unique_equip_ids),
        {slot.unique_equip_id for row in record.visible_slots for slot in row if not slot.is_hidden and slot.unique_equip_id is not None},
    )
    hidden_overlap = _set_jaccard(set(observation.hidden_slots), set(record.hidden_slots))
    similarity = 0.55 * visible_overlap + 0.25 * hero_overlap + 0.10 * equip_overlap + 0.10 * hidden_overlap
    return similarity, visible_overlap, hidden_overlap


@dataclass(frozen=True)
class RealCalibrationIngestionSummary:
    round_dir: str
    db_path: str
    round_id: str
    records_added: int
    skipped_pairs: int
    mean_match_result: float
    season: str
    server: str
    source_kind: str = "league_round_artifact"

    def to_json_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["schema_version"] = REAL_CALIBRATION_INGESTION_SUMMARY_SCHEMA_VERSION
        payload["module"] = "RealCalibrationIngestionSummary"
        return payload


@dataclass(frozen=True)
class RealCalibrationSampleBuildSummary:
    out_jsonl: str
    source_dirs: tuple[str, ...]
    samples_written: int
    skipped_pairs: int
    mean_label: float
    mean_sim_probability: float
    source_kinds: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["schema_version"] = REAL_CALIBRATION_SAMPLE_BUILD_SUMMARY_SCHEMA_VERSION
        payload["module"] = "RealCalibrationSampleBuildSummary"
        return payload


@dataclass(frozen=True)
class VersionDriftReport:
    baseline_season: str
    current_season: str
    baseline_records: int
    current_records: int
    baseline_mean_match_result: float
    current_mean_match_result: float
    match_result_delta: float
    observation_overlap: float
    drift_detected: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["schema_version"] = VERSION_DRIFT_REPORT_SCHEMA_VERSION
        payload["module"] = "VersionDriftReport"
        return payload


def ingest_league_round_real_meta(
    round_dir: str | Path,
    db_path: str | Path,
    *,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> RealCalibrationIngestionSummary:
    root = Path(round_dir)
    summary_payload = _read_json(root / "summary.json")
    round_id = str(summary_payload.get("round_id") or root.name)
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    results = {str(row.get("request_id")): row for row in _read_jsonl(root / "oracle_results.jsonl")}
    defense_by_id = {
        str(row.get("defense_id")): _defense_plan_from_dict(row["defense_plan"])
        for row in defenses
        if row.get("defense_id") is not None and row.get("defense_plan") is not None
    }
    db = RealMetaDB(path=db_path)
    added = 0
    skipped = 0
    match_results: list[float] = []
    for pair_index, row in enumerate(candidates, start=1):
        attack_data = row.get("attack_plan")
        defense = defense_by_id.get(str(row.get("defense_id")))
        if attack_data is None or defense is None:
            skipped += 1
            continue
        attack = _attack_plan_from_dict(attack_data)
        lane_results: list[float] = []
        for lane_idx in range(1, attack.format.n_teams + 1):
            request_id = f"{round_id}-p{pair_index:06d}-r{lane_idx}"
            result = results.get(request_id)
            if result is None:
                break
            lane_results.append(result_to_attack_win_rate(result))
        if len(lane_results) != attack.format.n_teams:
            skipped += 1
            continue
        match_result = match_win_probability(tuple(lane_results), attack.format.win_required)
        db.add(
            RealMetaRecord.from_match(
                observation=observe_defense(defense),
                full_defense_if_available=defense,
                attack_plan=attack,
                lane_results=tuple(lane_results),
                match_result=match_result,
                rank_segment=rank_segment,
                server=server,
                season=season,
                timestamp=timestamp,
            )
        )
        added += 1
        match_results.append(match_result)
    return RealCalibrationIngestionSummary(
        round_dir=str(root),
        db_path=str(db_path),
        round_id=round_id,
        records_added=added,
        skipped_pairs=skipped,
        mean_match_result=sum(match_results) / len(match_results) if match_results else 0.0,
        season=season,
        server=server,
        source_kind="league_round_artifact",
    )


def build_real_calibration_samples_from_artifacts(
    *,
    out_jsonl: str | Path,
    round_dirs: Sequence[str | Path] | None = None,
    active_real_feedback_dirs: Sequence[str | Path] | None = None,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> RealCalibrationSampleBuildSummary:
    rows: list[dict[str, Any]] = []
    skipped = 0
    source_dirs: list[str] = []
    source_kinds: set[str] = set()
    for round_dir in round_dirs or ():
        root = Path(round_dir)
        source_dirs.append(str(root))
        source_kinds.add("league_round_artifact")
        built, skipped_count = _build_real_calibration_samples_from_round_dir(
            root,
            rank_segment=rank_segment,
            server=server,
            season=season,
            timestamp=timestamp,
        )
        rows.extend(built)
        skipped += skipped_count
    for feedback_dir in active_real_feedback_dirs or ():
        root = Path(feedback_dir)
        source_dirs.append(str(root))
        source_kinds.add("active_real_query_feedback")
        built, skipped_count = _build_real_calibration_samples_from_feedback_dir(
            root,
            rank_segment=rank_segment,
            server=server,
            season=season,
            timestamp=timestamp,
        )
        rows.extend(built)
        skipped += skipped_count
    output = Path(out_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return RealCalibrationSampleBuildSummary(
        out_jsonl=str(output),
        source_dirs=tuple(source_dirs),
        samples_written=len(rows),
        skipped_pairs=skipped,
        mean_label=_mean(row["label"] for row in rows),
        mean_sim_probability=_mean(row["sim_probability"] for row in rows),
        source_kinds=tuple(sorted(source_kinds)),
    )


def ingest_active_real_query_feedback(
    feedback_dir: str | Path,
    db_path: str | Path,
    *,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> RealCalibrationIngestionSummary:
    root = Path(feedback_dir)
    summary_payload = _read_json(root / "summary.json")
    source_round_dir = str(summary_payload.get("round_dir") or "")
    round_id = str(summary_payload.get("round_id") or (Path(source_round_dir).name if source_round_dir else root.name))
    pair_rows = _read_jsonl(root / "real_query_pairs.jsonl")
    attack_rows = _read_jsonl(root / "attack_teacher.jsonl")
    defense_rows = _read_jsonl(root / "defense_teacher.jsonl")
    attacks = _active_feedback_attacks_by_key(attack_rows)
    defenses = _active_feedback_defenses_by_key(defense_rows)
    db = RealMetaDB(path=db_path)
    added = 0
    skipped = 0
    match_results: list[float] = []
    for row in pair_rows:
        attack = _lookup_active_feedback_attack(row, attacks)
        defense = _lookup_active_feedback_defense(row, defenses)
        if attack is None or defense is None:
            skipped += 1
            continue
        lane_results = _active_feedback_lane_results(row)
        if len(lane_results) != attack.format.n_teams or defense.format.n_teams != attack.format.n_teams:
            skipped += 1
            continue
        match_result = float(row.get("attack_success", match_win_probability(lane_results, attack.format.win_required)))
        db.add(
            RealMetaRecord.from_match(
                observation=observe_defense(defense),
                full_defense_if_available=defense,
                attack_plan=attack,
                lane_results=lane_results,
                match_result=match_result,
                rank_segment=rank_segment,
                server=server,
                season=season,
                timestamp=timestamp,
            )
        )
        added += 1
        match_results.append(match_result)
    return RealCalibrationIngestionSummary(
        round_dir=str(root),
        db_path=str(db_path),
        round_id=round_id,
        records_added=added,
        skipped_pairs=skipped,
        mean_match_result=sum(match_results) / len(match_results) if match_results else 0.0,
        season=season,
        server=server,
        source_kind="active_real_query_feedback",
    )


def build_version_drift_report(
    records: tuple[RealMetaRecord, ...] | list[RealMetaRecord],
    *,
    baseline_season: str,
    current_season: str,
    delta_threshold: float = 0.15,
    min_overlap: float = 0.20,
) -> VersionDriftReport:
    baseline = tuple(record for record in records if record.season == baseline_season)
    current = tuple(record for record in records if record.season == current_season)
    baseline_mean = _mean(record.match_result for record in baseline)
    current_mean = _mean(record.match_result for record in current)
    baseline_observations = {record.observation_hash for record in baseline}
    current_observations = {record.observation_hash for record in current}
    overlap_denominator = max(len(current_observations), 1)
    overlap = len(baseline_observations & current_observations) / overlap_denominator
    delta = current_mean - baseline_mean
    enough_records = bool(baseline and current)
    drift_detected = enough_records and (abs(delta) >= float(delta_threshold) or overlap < float(min_overlap))
    return VersionDriftReport(
        baseline_season=baseline_season,
        current_season=current_season,
        baseline_records=len(baseline),
        current_records=len(current),
        baseline_mean_match_result=baseline_mean,
        current_mean_match_result=current_mean,
        match_result_delta=delta,
        observation_overlap=overlap,
        drift_detected=drift_detected,
    )


def build_real_calibration_validation_report(
    *,
    samples_jsonl: Sequence[str | Path],
    calibration_json: str | Path,
    min_samples: int = 100,
    min_brier_improvement: float = 0.0,
    min_ece_improvement: float = 0.0,
    now: float | None = None,
    recency_tau: float = 7.0 * 24.0 * 60.0 * 60.0,
) -> dict[str, Any]:
    sample_paths = tuple(Path(path) for path in samples_jsonl)
    model_path = Path(calibration_json)
    scores, labels, feature_rows = _load_real_calibration_validation_samples(
        sample_paths,
        now=now,
        recency_tau=recency_tau,
    )
    payload = _read_json(model_path)
    model_payload = payload.get("model", payload)
    if not isinstance(model_payload, Mapping):
        raise ValueError("calibration JSON must contain a model object")
    model = RealCalibrationModel.from_dict(model_payload)
    base_payload = payload.get("base_model")
    base_model = RealCalibrationModel.from_dict(base_payload) if isinstance(base_payload, Mapping) else RealCalibrationModel()
    base_predictions = tuple(base_model.calibrate(score, features) for score, features in zip(scores, feature_rows))
    calibrated_predictions = tuple(model.calibrate(score, features) for score, features in zip(scores, feature_rows))
    feature_names = tuple(sorted({str(key) for row in feature_rows for key in row.keys()}))
    raw_brier = brier_score(labels, scores)
    base_brier = brier_score(labels, base_predictions)
    calibrated_brier = brier_score(labels, calibrated_predictions)
    raw_ece = expected_calibration_error(labels, scores)
    base_ece = expected_calibration_error(labels, base_predictions)
    calibrated_ece = expected_calibration_error(labels, calibrated_predictions)
    report = {
        "schema_version": REAL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION,
        "module": "RealCalibrationValidationReport",
        "samples_jsonl": [str(path) for path in sample_paths],
        "calibration_json": str(model_path),
        "samples": len(scores),
        "labels_mean": _mean(labels),
        "raw_prediction_mean": _mean(scores),
        "base_prediction_mean": _mean(base_predictions),
        "calibrated_prediction_mean": _mean(calibrated_predictions),
        "raw_brier": raw_brier,
        "base_brier": base_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": raw_brier - calibrated_brier,
        "base_brier_improvement": base_brier - calibrated_brier,
        "raw_ece": raw_ece,
        "base_ece": base_ece,
        "calibrated_ece": calibrated_ece,
        "ece_improvement": raw_ece - calibrated_ece,
        "base_ece_improvement": base_ece - calibrated_ece,
        "feature_names": list(feature_names),
    }
    red_lines = _real_calibration_validation_red_lines(
        report,
        min_samples=min_samples,
        min_brier_improvement=min_brier_improvement,
        min_ece_improvement=min_ece_improvement,
    )
    report["red_line_violations"] = red_lines
    report["production_ready"] = len(red_lines) == 0
    return report


@dataclass(frozen=True)
class RealCalibrationModel:
    logit_scale: float = 1.0
    bias: float = 0.0
    feature_weights: Mapping[str, float] | None = None

    def calibrate(self, sim_probability: float, features: Mapping[str, float] | None = None) -> float:
        logit = _logit(sim_probability)
        feature_shift = 0.0
        if self.feature_weights and features:
            feature_shift = sum(float(self.feature_weights.get(key, 0.0)) * float(value) for key, value in features.items())
        return _sigmoid(self.logit_scale * logit + self.bias + feature_shift)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "logit_scale": self.logit_scale,
            "bias": self.bias,
            "feature_weights": {} if self.feature_weights is None else dict(sorted(self.feature_weights.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealCalibrationModel":
        return cls(
            logit_scale=float(data.get("logit_scale", 1.0)),
            bias=float(data.get("bias", 0.0)),
            feature_weights={str(key): float(value) for key, value in data.get("feature_weights", {}).items()},
        )

    @classmethod
    def fit_platt(cls, sim_probabilities: tuple[float, ...], labels: tuple[float, ...]) -> "RealCalibrationModel":
        if len(sim_probabilities) != len(labels):
            raise ValueError("sim_probabilities and labels must have the same length")
        if not sim_probabilities:
            return cls()
        best = cls()
        best_score = brier_score(labels, tuple(best.calibrate(score) for score in sim_probabilities))
        # Small deterministic grid is enough for the first engineering
        # calibrator and avoids adding scipy/sklearn dependencies.
        for scale_step in range(5, 81):
            scale = scale_step / 10.0
            for bias_step in range(-80, 81):
                bias = bias_step / 20.0
                candidate = cls(logit_scale=scale, bias=bias)
                calibrated = tuple(candidate.calibrate(score) for score in sim_probabilities)
                score = brier_score(labels, calibrated)
                if score < best_score:
                    best = candidate
                    best_score = score
        return best

    @classmethod
    def fit_feature_calibrator(
        cls,
        sim_probabilities: Sequence[float],
        labels: Sequence[float],
        feature_rows: Sequence[Mapping[str, float]],
        *,
        feature_names: Sequence[str] | None = None,
        base_model: "RealCalibrationModel" | None = None,
        max_passes: int = 8,
    ) -> "RealCalibrationModel":
        scores = tuple(float(value) for value in sim_probabilities)
        targets = tuple(float(value) for value in labels)
        rows = tuple(feature_rows)
        if len(scores) != len(targets) or len(scores) != len(rows):
            raise ValueError("sim_probabilities, labels, and feature_rows must have the same length")
        if not scores:
            return cls()
        if max_passes < 0:
            raise ValueError("max_passes must be non-negative")
        names = (
            tuple(str(name) for name in feature_names)
            if feature_names is not None
            else tuple(sorted({str(key) for row in rows for key in row.keys()}))
        )
        base = base_model if base_model is not None else cls.fit_platt(scores, targets)
        if not names:
            return base
        weights = {name: float((base.feature_weights or {}).get(name, 0.0)) for name in names}
        bias = float(base.bias)
        scale = float(base.logit_scale)
        best_score = _feature_calibration_score(scores, targets, rows, scale=scale, bias=bias, weights=weights, names=names)
        deltas = (-4.0, 4.0, -2.0, 2.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, -0.1, 0.1)
        for _pass in range(max_passes):
            improved = False
            best_value = bias
            for delta in deltas:
                candidate_bias = bias + delta
                score = _feature_calibration_score(
                    scores,
                    targets,
                    rows,
                    scale=scale,
                    bias=candidate_bias,
                    weights=weights,
                    names=names,
                )
                if score + 1e-12 < best_score:
                    best_score = score
                    best_value = candidate_bias
                    improved = True
            bias = best_value
            for name in names:
                current = weights[name]
                best_value = current
                for delta in deltas:
                    candidate_weights = dict(weights)
                    candidate_weights[name] = current + delta
                    score = _feature_calibration_score(
                        scores,
                        targets,
                        rows,
                        scale=scale,
                        bias=bias,
                        weights=candidate_weights,
                        names=names,
                    )
                    if score + 1e-12 < best_score:
                        best_score = score
                        best_value = current + delta
                        improved = True
                weights[name] = best_value
            if not improved:
                break
        active_weights = {name: value for name, value in weights.items() if abs(value) > 1e-12}
        return cls(logit_scale=scale, bias=bias, feature_weights=active_weights)


def build_real_calibration_features(
    record: RealMetaRecord,
    *,
    now: float | None = None,
    recency_tau: float = 7.0 * 24.0 * 60.0 * 60.0,
) -> dict[str, float]:
    visible_slots = tuple(slot for row in record.visible_slots for slot in row if not slot.is_hidden)
    hidden_slots_total = float(len(record.hidden_slots))
    total_slots = float(sum(len(row) for row in record.visible_slots)) or float(
        record.attack_plan.format.n_teams * record.attack_plan.format.team_size
    )
    hidden_by_team: dict[int, int] = {}
    for team_idx, _slot_idx in record.hidden_slots:
        hidden_by_team[int(team_idx)] = hidden_by_team.get(int(team_idx), 0) + 1
    visible_stars = tuple(float(star) for _equip_id, star in record.unique_equip_stars)
    visible_powers = tuple(float(slot.final_power) for slot in visible_slots if slot.final_power is not None)
    visible_ranks = tuple(float(slot.standing_rank) for slot in visible_slots if slot.standing_rank is not None)
    attack_loadouts = tuple(loadout for team in record.attack_plan.teams for loadout in team.slots)
    attack_stars = tuple(float(loadout.unique_equip_star) for loadout in attack_loadouts if loadout.unique_equip_star is not None)
    attack_powers = tuple(float(loadout.final_power) for loadout in attack_loadouts)
    attack_ranks = tuple(float(loadout.standing_rank) for loadout in attack_loadouts)
    hidden_budget = max(float(record.attack_plan.format.max_hidden_total), 1.0)
    features = {
        "hidden_slots_total": hidden_slots_total,
        "hidden_fraction": hidden_slots_total / max(total_slots, 1.0),
        "hidden_budget_fraction": hidden_slots_total / hidden_budget,
        "visible_fraction": float(len(visible_slots)) / max(total_slots, 1.0),
        "max_hidden_slots_per_team": float(max(hidden_by_team.values(), default=0)),
        "mean_hidden_slots_per_team": hidden_slots_total / max(float(record.attack_plan.format.n_teams), 1.0),
        "match_teams": float(record.attack_plan.format.n_teams),
        "match_win_required": float(record.attack_plan.format.win_required or 0),
        "visible_unique_star_mean": _mean(visible_stars),
        "visible_unique_star_min": min(visible_stars) if visible_stars else 0.0,
        "visible_unique_star_max": max(visible_stars) if visible_stars else 0.0,
        "visible_unique_star_known_fraction": float(len(visible_stars)) / max(float(len(visible_slots)), 1.0),
        "visible_power_mean_k": _mean(visible_powers) / 1000.0,
        "visible_standing_rank_mean_100": _mean(visible_ranks) / 100.0,
        "attack_unique_star_mean": _mean(attack_stars),
        "attack_power_mean_k": _mean(attack_powers) / 1000.0,
        "attack_standing_rank_mean_100": _mean(attack_ranks) / 100.0,
        "recency_weight": 1.0 if now is None else time_decay_weight(now=now, timestamp=record.timestamp, tau=recency_tau),
        f"server:{record.server or 'unknown'}": 1.0,
        f"season:{record.season or 'unknown'}": 1.0,
        f"rank_segment:{record.rank_segment or 'unknown'}": 1.0,
    }
    return features


def time_decay_weight(*, now: float, timestamp: float, tau: float) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return math.exp(-max(0.0, float(now) - float(timestamp)) / tau)


def _feature_calibration_score(
    sim_probabilities: tuple[float, ...],
    labels: tuple[float, ...],
    feature_rows: tuple[Mapping[str, float], ...],
    *,
    scale: float,
    bias: float,
    weights: Mapping[str, float],
    names: tuple[str, ...],
) -> float:
    predictions = []
    for sim_probability, row in zip(sim_probabilities, feature_rows):
        shift = sum(float(weights.get(name, 0.0)) * float(row.get(name, 0.0)) for name in names)
        predictions.append(_sigmoid(float(scale) * _logit(sim_probability) + float(bias) + shift))
    return brier_score(labels, tuple(predictions))


def _load_real_calibration_validation_samples(
    paths: Sequence[Path],
    *,
    now: float | None,
    recency_tau: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[dict[str, float], ...]]:
    scores: list[float] = []
    labels: list[float] = []
    feature_rows: list[dict[str, float]] = []
    for path in paths:
        for line_no, row in enumerate(_read_jsonl(path), start=1):
            scores.append(_required_sample_float(row, ("sim_probability", "sim_prob", "prediction", "score"), path, line_no))
            labels.append(_required_sample_float(row, ("label", "match_result", "real_label", "win_rate"), path, line_no))
            feature_rows.append(
                _real_calibration_sample_features(row, path=path, line_no=line_no, now=now, recency_tau=recency_tau)
            )
    return tuple(scores), tuple(labels), tuple(feature_rows)


def _real_calibration_sample_features(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_no: int,
    now: float | None,
    recency_tau: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    record_payload = row.get("real_meta_record", row.get("record"))
    if record_payload is not None:
        if not isinstance(record_payload, Mapping):
            raise ValueError(f"{path}:{line_no}: record must be a JSON object")
        features.update(build_real_calibration_features(RealMetaRecord.from_dict(record_payload), now=now, recency_tau=recency_tau))
    raw_features = row.get("features", {})
    if not isinstance(raw_features, Mapping):
        raise ValueError(f"{path}:{line_no}: features must be a JSON object")
    features.update({str(key): float(value) for key, value in raw_features.items()})
    return features


def _required_sample_float(row: Mapping[str, Any], keys: tuple[str, ...], path: Path, line_no: int) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    raise ValueError(f"{path}:{line_no}: missing one of {', '.join(keys)}")


def _real_calibration_validation_red_lines(
    report: Mapping[str, Any],
    *,
    min_samples: int,
    min_brier_improvement: float,
    min_ece_improvement: float,
) -> list[str]:
    violations: list[str] = []
    samples = int(report.get("samples", 0) or 0)
    if samples <= 0:
        violations.append("no_real_calibration_holdout_samples")
    if samples < int(min_samples):
        violations.append("real_calibration_holdout_samples_low")
    if float(report.get("brier_improvement", 0.0) or 0.0) < float(min_brier_improvement):
        violations.append("real_calibration_brier_not_improved")
    if float(report.get("ece_improvement", 0.0) or 0.0) < float(min_ece_improvement):
        violations.append("real_calibration_ece_not_improved")
    return _dedupe_strings(violations)


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: Any) -> float:
    rows = tuple(float(value) for value in values)
    return sum(rows) / len(rows) if rows else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def _build_real_calibration_samples_from_round_dir(
    root: Path,
    *,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> tuple[list[dict[str, Any]], int]:
    summary_payload = _read_json(root / "summary.json")
    round_id = str(summary_payload.get("round_id") or root.name)
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    pair_rows = _read_jsonl(root / "oracle_pairs.jsonl")
    candidates_by_pair = _candidates_by_pair(candidates)
    defenses_by_id = {
        str(row.get("defense_id")): _defense_plan_from_dict(row["defense_plan"])
        for row in defenses
        if row.get("defense_id") is not None and isinstance(row.get("defense_plan"), Mapping)
    }
    rows: list[dict[str, Any]] = []
    skipped = 0
    for pair in pair_rows:
        key = (str(pair.get("attack_id") or ""), str(pair.get("defense_id") or ""))
        candidate = candidates_by_pair.get(key)
        defense = defenses_by_id.get(key[1])
        if candidate is None or defense is None or not isinstance(candidate.get("attack_plan"), Mapping):
            skipped += 1
            continue
        prediction = _calibration_prediction_from_candidate(candidate)
        if prediction is None:
            skipped += 1
            continue
        attack = _attack_plan_from_dict(candidate["attack_plan"])
        row = _real_calibration_sample_row(
            pair,
            attack=attack,
            defense=defense,
            sim_probability=prediction[0],
            prediction_source=prediction[1],
            round_id=round_id,
            source_kind="league_round_artifact",
            query_id=None,
            rank_segment=rank_segment,
            server=server,
            season=season,
            timestamp=timestamp,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped


def _build_real_calibration_samples_from_feedback_dir(
    root: Path,
    *,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> tuple[list[dict[str, Any]], int]:
    summary_payload = _read_json(root / "summary.json")
    source_round = Path(str(summary_payload.get("round_dir") or ""))
    source_round_summary = _read_json(source_round / "summary.json") if str(source_round) else {}
    round_id = str(summary_payload.get("round_id") or source_round_summary.get("round_id") or (source_round.name if str(source_round) else root.name))
    candidates_by_pair = _candidates_by_pair(_read_jsonl(source_round / "candidates.jsonl")) if str(source_round) else {}
    pair_rows = _read_jsonl(root / "real_query_pairs.jsonl")
    attacks = _active_feedback_attacks_by_key(_read_jsonl(root / "attack_teacher.jsonl"))
    defenses = _active_feedback_defenses_by_key(_read_jsonl(root / "defense_teacher.jsonl"))
    rows: list[dict[str, Any]] = []
    skipped = 0
    for pair in pair_rows:
        key = (str(pair.get("attack_id") or ""), str(pair.get("defense_id") or ""))
        candidate = candidates_by_pair.get(key)
        prediction = _calibration_prediction_from_candidate(candidate or {})
        attack = _lookup_active_feedback_attack(pair, attacks)
        defense = _lookup_active_feedback_defense(pair, defenses)
        if prediction is None or attack is None or defense is None:
            skipped += 1
            continue
        row = _real_calibration_sample_row(
            pair,
            attack=attack,
            defense=defense,
            sim_probability=prediction[0],
            prediction_source=prediction[1],
            round_id=round_id,
            source_kind="active_real_query_feedback",
            query_id=None if pair.get("query_id") is None else str(pair.get("query_id")),
            rank_segment=rank_segment,
            server=server,
            season=season,
            timestamp=timestamp,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped


def _real_calibration_sample_row(
    pair: Mapping[str, Any],
    *,
    attack: AttackPlan,
    defense: DefensePlan,
    sim_probability: float,
    prediction_source: str,
    round_id: str,
    source_kind: str,
    query_id: str | None,
    rank_segment: str,
    server: str,
    season: str,
    timestamp: float,
) -> dict[str, Any] | None:
    lane_results = _active_feedback_lane_results(pair)
    if len(lane_results) != attack.format.n_teams or defense.format.n_teams != attack.format.n_teams:
        return None
    label = float(pair.get("attack_success", match_win_probability(lane_results, attack.format.win_required)))
    record = RealMetaRecord.from_match(
        observation=observe_defense(defense),
        full_defense_if_available=defense,
        attack_plan=attack,
        lane_results=lane_results,
        match_result=label,
        rank_segment=rank_segment,
        server=server,
        season=season,
        timestamp=timestamp,
    )
    features = build_real_calibration_features(record)
    features[f"source_kind:{source_kind}"] = 1.0
    features[f"prediction_source:{prediction_source}"] = 1.0
    row = {
        "round_id": round_id,
        "attack_id": pair.get("attack_id"),
        "defense_id": pair.get("defense_id"),
        "source_kind": source_kind,
        "sim_probability": float(sim_probability),
        "label": label,
        "prediction_source": prediction_source,
        "round_win_rates": lane_results,
        "real_meta_record": record.to_dict(),
        "features": features,
    }
    if query_id is not None:
        row["query_id"] = query_id
    return row


def _candidates_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row.get("attack_id") or ""), str(row.get("defense_id") or "")): row
        for row in rows
        if row.get("attack_id") is not None and row.get("defense_id") is not None
    }


def _calibration_prediction_from_candidate(row: Mapping[str, Any]) -> tuple[float, str] | None:
    for key in ("predicted_score", "surrogate_score", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), key
    return None


def _active_feedback_attacks_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[object, AttackPlan]]:
    exact: dict[tuple[str, str, str], AttackPlan] = {}
    by_pair: dict[tuple[str, str], AttackPlan] = {}
    for row in rows:
        attack_payload = row.get("attack_plan")
        if not isinstance(attack_payload, Mapping):
            continue
        attack = _attack_plan_from_dict(attack_payload)
        query_id = str(row.get("query_id") or "")
        attack_id = str(row.get("attack_id") or "")
        defense_id = str(row.get("defense_id") or "")
        exact[(query_id, attack_id, defense_id)] = attack
        if attack_id or defense_id:
            by_pair[(attack_id, defense_id)] = attack
    return {"exact": exact, "by_pair": by_pair}


def _active_feedback_defenses_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[object, DefensePlan]]:
    exact: dict[tuple[str, str], DefensePlan] = {}
    by_id: dict[str, DefensePlan] = {}
    for row in rows:
        defense_payload = row.get("defense_plan")
        if not isinstance(defense_payload, Mapping):
            continue
        defense = _defense_plan_from_dict(defense_payload)
        query_id = str(row.get("query_id") or "")
        defense_id = str(row.get("defense_id") or "")
        exact[(query_id, defense_id)] = defense
        if defense_id:
            by_id[defense_id] = defense
    return {"exact": exact, "by_id": by_id}


def _lookup_active_feedback_attack(
    row: Mapping[str, Any],
    attacks: Mapping[str, Any],
) -> AttackPlan | None:
    exact = attacks.get("exact", {})
    by_pair = attacks.get("by_pair", {})
    query_id = str(row.get("query_id") or "")
    attack_id = str(row.get("attack_id") or "")
    defense_id = str(row.get("defense_id") or "")
    if isinstance(exact, Mapping):
        attack = exact.get((query_id, attack_id, defense_id))
        if isinstance(attack, AttackPlan):
            return attack
    if isinstance(by_pair, Mapping):
        attack = by_pair.get((attack_id, defense_id))
        if isinstance(attack, AttackPlan):
            return attack
    return None


def _lookup_active_feedback_defense(
    row: Mapping[str, Any],
    defenses: Mapping[str, Any],
) -> DefensePlan | None:
    exact = defenses.get("exact", {})
    by_id = defenses.get("by_id", {})
    query_id = str(row.get("query_id") or "")
    defense_id = str(row.get("defense_id") or "")
    if isinstance(exact, Mapping):
        defense = exact.get((query_id, defense_id))
        if isinstance(defense, DefensePlan):
            return defense
    if isinstance(by_id, Mapping):
        defense = by_id.get(defense_id)
        if isinstance(defense, DefensePlan):
            return defense
    return None


def _active_feedback_lane_results(row: Mapping[str, Any]) -> tuple[float, ...]:
    raw = row.get("round_win_rates", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(float(value) for value in raw)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _pairs(values: Any, *, key_type: type = str) -> tuple[tuple[Any, float], ...]:
    return tuple((key_type(item[0]), float(item[1])) for item in values)


def _visible_slot_signature(slots: Sequence[Sequence[VisibleSlot]]) -> set[tuple[int, int, int, int | None, int | None]]:
    signature: set[tuple[int, int, int, int | None, int | None]] = set()
    for team_idx, row in enumerate(slots, start=1):
        for slot_idx, slot in enumerate(row, start=1):
            if slot.is_hidden or slot.hero_id is None:
                continue
            signature.add((team_idx, slot_idx, int(slot.hero_id), slot.unique_equip_id, slot.unique_equip_star))
    return signature


def _set_overlap(left: set[Any], right: set[Any]) -> float:
    if not left:
        return 1.0 if not right else 0.0
    return len(left & right) / len(left)


def _set_jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _match_format_from_dict(data: Mapping[str, Any]) -> MatchFormat:
    return MatchFormat(
        n_teams=int(data["n_teams"]),
        team_size=int(data.get("team_size", 5)),
        win_required=None if data.get("win_required") is None else int(data["win_required"]),
        max_hidden_per_team=int(data.get("max_hidden_per_team", 2)),
        max_hidden_total=int(data.get("max_hidden_total", 10)),
    )


def _loadout_from_dict(data: Mapping[str, Any]) -> Loadout:
    return Loadout(
        hero_id=int(data["hero_id"]),
        unique_equip_id=None if data.get("unique_equip_id") is None else int(data["unique_equip_id"]),
        unique_equip_star=None if data.get("unique_equip_star") is None else int(data["unique_equip_star"]),
        normal_equip_ids=tuple(int(value) for value in data.get("normal_equip_ids", ())),
        normal_equip_features=_pairs(data.get("normal_equip_features", ())),
        level_features=_pairs(data.get("level_features", ())),
        final_stats=_pairs(data.get("final_stats", ())),
        final_power=float(data.get("final_power", 0.0)),
        standing_rank=float(data.get("standing_rank", 0.0)),
        standing_bucket=str(data.get("standing_bucket", "custom")),
    )


def _team_from_dict(data: Mapping[str, Any]) -> Team:
    return Team(tuple(_loadout_from_dict(item) for item in data["slots"]))


def _attack_plan_from_dict(data: Mapping[str, Any]) -> AttackPlan:
    return AttackPlan(
        format=_match_format_from_dict(data["format"]),
        teams=tuple(_team_from_dict(item) for item in data["teams"]),
        source=str(data["source"]),
        plan_id=None if data.get("plan_id") is None else str(data["plan_id"]),
        version=str(data.get("version", "v4")),
        season=str(data.get("season", "unknown")),
        rank_segment=str(data.get("rank_segment", "unknown")),
    )


def _defense_plan_from_dict(data: Mapping[str, Any]) -> DefensePlan:
    return DefensePlan(
        format=_match_format_from_dict(data["format"]),
        teams=tuple(_team_from_dict(item) for item in data["teams"]),
        mask=tuple(tuple(int(value) for value in row) for row in data["mask"]),
        source=str(data["source"]),
        plan_id=None if data.get("plan_id") is None else str(data["plan_id"]),
        version=str(data.get("version", "v4")),
        season=str(data.get("season", "unknown")),
        rank_segment=str(data.get("rank_segment", "unknown")),
    )


def _visible_slot_from_dict(data: Mapping[str, Any]) -> VisibleSlot:
    return VisibleSlot(
        hero_id=None if data.get("hero_id") is None else int(data["hero_id"]),
        unique_equip_id=None if data.get("unique_equip_id") is None else int(data["unique_equip_id"]),
        unique_equip_star=None if data.get("unique_equip_star") is None else int(data["unique_equip_star"]),
        normal_equip_summary=(
            None if data.get("normal_equip_summary") is None else tuple(int(value) for value in data["normal_equip_summary"])
        ),
        final_power=None if data.get("final_power") is None else float(data["final_power"]),
        standing_rank=None if data.get("standing_rank") is None else float(data["standing_rank"]),
        is_hidden=bool(data["is_hidden"]),
        loadout=None if data.get("loadout") is None else _loadout_from_dict(data["loadout"]),
    )
