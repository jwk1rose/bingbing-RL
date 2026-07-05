from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..domain import AttackPlan, DefensePlan, Loadout, MatchFormat, Team
from ..real_platform.oracle import OracleBatchEvaluator, OracleEvaluationRecord


@dataclass(frozen=True)
class ActiveRealQueryDispatchSummary:
    round_dir: str
    out_dir: str
    queued_queries: int
    dispatchable_queries: int
    skipped_queries: int
    dispatched_pairs: int
    oracle_requests: int
    oracle_result_errors: int
    completion_rate: float
    attack_teacher_rows: int
    defense_teacher_rows: int
    teacher_feedback_complete: bool
    real_query_queue_validated: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def dispatch_active_real_queries(
    round_dir: str | Path,
    out_dir: str | Path,
    *,
    evaluator: OracleBatchEvaluator,
    job_prefix: str,
    base_seed: int,
    max_queries: int | None = None,
) -> ActiveRealQueryDispatchSummary:
    root = Path(round_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    active_queries = [row for row in _read_jsonl(root / "active_queries.jsonl") if row.get("queue") == "real"]
    if max_queries is not None:
        active_queries = active_queries[: max(int(max_queries), 0)]
    candidates = _read_jsonl(root / "candidates.jsonl")
    defenses = _read_jsonl(root / "scored_defenses.jsonl")
    candidate_by_pair = {
        (str(row.get("attack_id")), str(row.get("defense_id"))): row
        for row in candidates
        if row.get("attack_id") is not None and row.get("defense_id") is not None
    }
    defense_by_id = {
        str(row.get("defense_id")): row
        for row in defenses
        if row.get("defense_id") is not None and row.get("defense_plan") is not None
    }
    dispatch_rows: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], AttackPlan, DefensePlan]] = []
    skipped_rows: list[dict[str, Any]] = []
    for query in active_queries:
        key = (str(query.get("attack_id")), str(query.get("defense_id")))
        candidate = candidate_by_pair.get(key)
        defense_row = defense_by_id.get(key[1])
        if candidate is None or defense_row is None:
            skipped_rows.append(
                {
                    "query_id": query.get("query_id"),
                    "attack_id": query.get("attack_id"),
                    "defense_id": query.get("defense_id"),
                    "reason": "missing_candidate_or_defense_artifact",
                }
            )
            continue
        attack_data = candidate.get("attack_plan")
        defense_data = defense_row.get("defense_plan")
        if attack_data is None or defense_data is None:
            skipped_rows.append(
                {
                    "query_id": query.get("query_id"),
                    "attack_id": query.get("attack_id"),
                    "defense_id": query.get("defense_id"),
                    "reason": "missing_attack_or_defense_plan",
                }
            )
            continue
        dispatch_rows.append(
            (
                query,
                candidate,
                defense_row,
                _attack_plan_from_dict(attack_data),
                _defense_plan_from_dict(defense_data),
            )
        )
    records = evaluator.evaluate_pairs(
        [
            (str(query["attack_id"]), attack, str(query["defense_id"]), defense)
            for query, _candidate, _defense_row, attack, defense in dispatch_rows
        ],
        job_prefix=job_prefix,
        base_seed=base_seed,
        metadata={"kind": "masked_team_league_real_query", "round_dir": str(root), "queries": len(dispatch_rows)},
    )
    record_by_pair = {(record.attack_id, record.defense_id): record for record in records}
    pair_rows: list[dict[str, Any]] = []
    attack_teacher_rows: list[dict[str, Any]] = []
    defense_teacher_rows: list[dict[str, Any]] = []
    for query, candidate, defense_row, attack, defense in dispatch_rows:
        record = record_by_pair.get((str(query["attack_id"]), str(query["defense_id"])))
        if record is None:
            continue
        pair_rows.append(_real_query_pair_row(query, record))
        attack_teacher_rows.append(_attack_teacher_row(query, candidate, attack, record))
        defense_teacher_rows.append(_defense_teacher_row(query, defense_row, defense, record))
    _write_jsonl(output / "real_query_pairs.jsonl", pair_rows)
    _write_jsonl(output / "real_query_requests.jsonl", [request for record in records for request in record.requests])
    _write_jsonl(output / "real_query_results.jsonl", [result for record in records for result in record.results])
    _write_jsonl(output / "attack_teacher.jsonl", attack_teacher_rows)
    _write_jsonl(output / "defense_teacher.jsonl", defense_teacher_rows)
    _write_jsonl(output / "skipped_real_queries.jsonl", skipped_rows)
    oracle_result_errors = sum(
        1
        for record in records
        for result in record.results
        if result.get("status") not in {"completed", "cached"}
    )
    oracle_requests = sum(len(record.oracle_request_ids) for record in records)
    completion_rate = 1.0 if oracle_requests == 0 else (oracle_requests - oracle_result_errors) / oracle_requests
    teacher_feedback_complete = len(attack_teacher_rows) == len(records) and len(defense_teacher_rows) == len(records)
    summary = ActiveRealQueryDispatchSummary(
        round_dir=str(root),
        out_dir=str(output),
        queued_queries=len(active_queries),
        dispatchable_queries=len(dispatch_rows),
        skipped_queries=len(skipped_rows),
        dispatched_pairs=len(records),
        oracle_requests=oracle_requests,
        oracle_result_errors=oracle_result_errors,
        completion_rate=completion_rate,
        attack_teacher_rows=len(attack_teacher_rows),
        defense_teacher_rows=len(defense_teacher_rows),
        teacher_feedback_complete=teacher_feedback_complete,
        real_query_queue_validated=bool(active_queries) and len(records) == len(dispatch_rows) and oracle_result_errors == 0 and teacher_feedback_complete,
    )
    (output / "summary.json").write_text(json.dumps(summary.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": "active_real_query_dispatch_validation.v1",
                "module": "ActiveRealQueryDispatch",
                **summary.to_json_dict(),
                "skipped_query_reasons": _reason_counts(skipped_rows),
                "submitted_request_count": oracle_requests,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _real_query_pair_row(query: Mapping[str, Any], record: OracleEvaluationRecord) -> dict[str, Any]:
    return {
        "query_id": query.get("query_id"),
        "query_type": query.get("query_type"),
        "queue": query.get("queue"),
        "score": query.get("score"),
        "attack_id": record.attack_id,
        "defense_id": record.defense_id,
        "attack_hash": record.attack_hash,
        "defense_hash": record.defense_hash,
        "attack_success": record.attack_success,
        "round_win_rates": record.round_win_rates,
        "oracle_request_ids": record.oracle_request_ids,
    }


def _attack_teacher_row(
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    attack: AttackPlan,
    record: OracleEvaluationRecord,
) -> dict[str, Any]:
    target_baseline_break_rate = _float_or(candidate.get("target_baseline_break_rate"), 0.0)
    return {
        "teacher_group_id": f"active_real:{record.defense_id}",
        "query_id": query.get("query_id"),
        "defense_id": record.defense_id,
        "attack_id": record.attack_id,
        "attack_role": candidate.get("attack_role"),
        "rank": candidate.get("rank"),
        "attack_plan": _jsonable(attack),
        "attack_success": record.attack_success,
        "gap_target": float(candidate.get("belief_top1_top2_gap") or 0.0),
        "target_defense_id": candidate.get("target_defense_id", record.defense_id),
        "target_defense_hash": candidate.get("target_defense_hash", candidate.get("defense_hash", record.defense_hash)),
        "target_defense_strength": candidate.get("target_defense_strength"),
        "target_baseline_break_rate": target_baseline_break_rate,
        "exploiter_residual_target": round(float(record.attack_success) - target_baseline_break_rate, 12),
        "role_weight": _float_or(candidate.get("role_weight"), 1.0),
        "source": "active_real_query",
    }


def _defense_teacher_row(
    query: Mapping[str, Any],
    defense_row: Mapping[str, Any],
    defense: DefensePlan,
    record: OracleEvaluationRecord,
) -> dict[str, Any]:
    risk = defense_row.get("defense_risk_report")
    if not isinstance(risk, Mapping):
        risk = {}
    meta_attack_success = _float_or(defense_row.get("meta_attack_success"), _float_or(risk.get("meta_attack_success"), 0.0))
    survival_rate = 1.0 - float(record.attack_success)
    return {
        "teacher_group_id": f"active_real:{record.defense_id}",
        "query_id": query.get("query_id"),
        "defense_id": record.defense_id,
        "defense_role": defense_row.get("defense_role"),
        "defense_plan": _jsonable(defense),
        "break_rate": float(record.attack_success),
        "value_target": survival_rate,
        "survival_rate": survival_rate,
        "meta_attack_success": meta_attack_success,
        "anti_meta_residual_target": round(survival_rate - meta_attack_success, 12),
        "gap_target": float(defense_row.get("gap_target", defense_row.get("ambiguity_score", 0.0)) or 0.0),
        "ambiguity_score": float(defense_row.get("ambiguity_score", 0.0) or 0.0),
        "source": "active_real_query",
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _float_or(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(fallback)


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
        source=str(data.get("source", "artifact")),
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
        source=str(data.get("source", "artifact")),
        plan_id=None if data.get("plan_id") is None else str(data["plan_id"]),
        version=str(data.get("version", "v4")),
        season=str(data.get("season", "unknown")),
        rank_segment=str(data.get("rank_segment", "unknown")),
    )


def _pairs(values: Any) -> tuple[tuple[str, float], ...]:
    return tuple((str(item[0]), float(item[1])) for item in values)
