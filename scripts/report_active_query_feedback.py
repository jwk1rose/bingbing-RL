#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masked_team_league.reports import build_active_query_feedback_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an active/real-query feedback report from league round artifacts.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-matched-query-coverage", type=float, default=1.0)
    parser.add_argument("--max-oracle-result-error-rate", type=float, default=0.0)
    parser.add_argument("--min-real-query-count", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = tuple(
        build_active_query_feedback_report(
            round_dir,
            min_matched_query_coverage=args.min_matched_query_coverage,
            max_oracle_result_error_rate=args.max_oracle_result_error_rate,
            min_real_query_count=args.min_real_query_count,
        )
        for round_dir in args.round_dir
    )
    report = reports[0] if len(reports) == 1 else _merge_reports(
        reports,
        min_matched_query_coverage=args.min_matched_query_coverage,
        max_oracle_result_error_rate=args.max_oracle_result_error_rate,
        min_real_query_count=args.min_real_query_count,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "queries": report["queries"],
                "matched_query_coverage": report["matched_query_coverage"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _merge_reports(
    reports: tuple[dict[str, object], ...],
    *,
    min_matched_query_coverage: float,
    max_oracle_result_error_rate: float,
    min_real_query_count: int,
) -> dict[str, object]:
    queries = sum(int(report.get("queries", 0) or 0) for report in reports)
    matched = sum(int(report.get("matched_queries", 0) or 0) for report in reports)
    results = sum(int(report.get("oracle_result_rows", 0) or 0) for report in reports)
    oracle_errors = sum(int(report.get("oracle_result_errors", 0) or 0) for report in reports)
    real_queries = sum(int(report.get("real_queries", 0) or 0) for report in reports)
    matched_real = sum(int(report.get("matched_real_queries", 0) or 0) for report in reports)
    sim_queries = sum(int(report.get("sim_queries", 0) or 0) for report in reports)
    matched_sim = sum(int(report.get("matched_sim_queries", 0) or 0) for report in reports)
    query_feedback = [row for report in reports for row in (report.get("query_feedback", ()) or ())]
    merged = {
        "schema_version": "active_query_feedback_report.v1",
        "module": "ActiveQueryFeedbackReport",
        "round_dir": ",".join(str(report.get("round_dir", "")) for report in reports),
        "round_dirs": [str(report.get("round_dir", "")) for report in reports],
        "queries": queries,
        "matched_queries": matched,
        "unmatched_queries": queries - matched,
        "matched_query_coverage": 1.0 if queries <= 0 else matched / queries,
        "real_queries": real_queries,
        "matched_real_queries": matched_real,
        "real_query_feedback_coverage": 1.0 if real_queries <= 0 else matched_real / real_queries,
        "sim_queries": sim_queries,
        "matched_sim_queries": matched_sim,
        "sim_query_feedback_coverage": 1.0 if sim_queries <= 0 else matched_sim / sim_queries,
        "oracle_pairs": sum(int(report.get("oracle_pairs", 0) or 0) for report in reports),
        "oracle_result_rows": results,
        "oracle_result_errors": oracle_errors,
        "oracle_result_error_rate": 0.0 if results <= 0.0 else oracle_errors / results,
        "queues": _queue_stats(query_feedback),
        "query_feedback": query_feedback,
    }
    red_lines: list[str] = []
    for report in reports:
        red_lines.extend(str(value) for value in (report.get("red_line_violations", ()) or ()))
    if float(merged["matched_query_coverage"]) < float(min_matched_query_coverage):
        red_lines.append("active_query_feedback_coverage_low")
    if int(merged["real_queries"]) < int(min_real_query_count):
        red_lines.append("real_query_count_low")
    if int(merged["real_queries"]) > int(merged["matched_real_queries"]):
        red_lines.append("real_query_feedback_missing")
    if float(merged["oracle_result_error_rate"]) > float(max_oracle_result_error_rate):
        red_lines.append("oracle_result_errors")
    merged["red_line_violations"] = _dedupe(red_lines)
    return merged


def _queue_stats(rows: list[object]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    queues = sorted({str(row.get("queue", "unknown")) for row in rows if isinstance(row, dict)})
    for queue in queues:
        queue_rows = [row for row in rows if isinstance(row, dict) and str(row.get("queue", "unknown")) == queue]
        successes = [float(row["attack_success"]) for row in queue_rows if isinstance(row.get("attack_success"), (int, float))]
        stats[queue] = {
            "queries": len(queue_rows),
            "matched_queries": len(successes),
            "underdog_queries": sum(1 for row in queue_rows if row.get("query_type") == "underdog"),
            "mean_attack_success": 0.0 if not successes else sum(successes) / len(successes),
        }
    return stats


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
