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

from masked_team_league.reports import build_belief_real_distribution_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate belief-model real-distribution similarity usage in round artifacts.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-real-coverage", type=float, default=0.50)
    parser.add_argument("--min-mean-real-records", type=float, default=1.0)
    parser.add_argument("--min-mean-real-similarity", type=float, default=0.25)
    parser.add_argument("--max-oracle-alignment-mae", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = tuple(
        build_belief_real_distribution_validation_report(
            round_dir,
            min_real_coverage=args.min_real_coverage,
            min_mean_real_records=args.min_mean_real_records,
            min_mean_real_similarity=args.min_mean_real_similarity,
            max_oracle_alignment_mae=args.max_oracle_alignment_mae,
        )
        for round_dir in args.round_dir
    )
    report = reports[0] if len(reports) == 1 else _merge_reports(
        reports,
        min_real_coverage=args.min_real_coverage,
        min_mean_real_records=args.min_mean_real_records,
        min_mean_real_similarity=args.min_mean_real_similarity,
        max_oracle_alignment_mae=args.max_oracle_alignment_mae,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "candidates": report["candidates"],
                "real_distribution_coverage": report["real_distribution_coverage"],
                "oracle_alignment_mae": report["oracle_alignment_mae"],
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
    min_real_coverage: float,
    min_mean_real_records: float,
    min_mean_real_similarity: float,
    max_oracle_alignment_mae: float,
) -> dict[str, object]:
    candidate_rows = [row for report in reports for row in (report.get("candidate_rows", ()) or ())]
    candidates = sum(int(report.get("candidates", 0) or 0) for report in reports)
    real_rows = sum(int(report.get("real_distribution_rows", 0) or 0) for report in reports)
    real_record_total = sum(
        float(report.get("mean_real_record_count", 0.0) or 0.0) * int(report.get("real_distribution_rows", 0) or 0)
        for report in reports
    )
    real_similarity_total = sum(
        float(report.get("mean_real_similarity", 0.0) or 0.0) * int(report.get("real_distribution_rows", 0) or 0)
        for report in reports
    )
    real_match_total = sum(
        float(report.get("mean_real_match_result", 0.0) or 0.0) * int(report.get("real_distribution_rows", 0) or 0)
        for report in reports
    )
    entropy_rows = [
        row for row in candidate_rows if isinstance(row, dict) and row.get("weight_entropy_normalized") is not None
    ]
    entropy_total = sum(float(row.get("weight_entropy_normalized", 0.0) or 0.0) for row in entropy_rows)
    alignment_rows = sum(int(report.get("oracle_alignment_rows", 0) or 0) for report in reports)
    alignment_total = sum(
        float(report.get("oracle_alignment_mae", 0.0) or 0.0) * int(report.get("oracle_alignment_rows", 0) or 0)
        for report in reports
    )
    merged = {
        "schema_version": "belief_real_distribution_validation_report.v1",
        "module": "BeliefRealDistributionValidationReport",
        "round_dir": ",".join(str(report.get("round_dir", "")) for report in reports),
        "round_dirs": [str(report.get("round_dir", "")) for report in reports],
        "candidates": candidates,
        "belief_domain_stats_rows": sum(int(report.get("belief_domain_stats_rows", 0) or 0) for report in reports),
        "real_distribution_rows": real_rows,
        "real_distribution_coverage": 0.0 if candidates <= 0 else real_rows / candidates,
        "exact_real_rows": sum(int(report.get("exact_real_rows", 0) or 0) for report in reports),
        "similar_real_rows": sum(int(report.get("similar_real_rows", 0) or 0) for report in reports),
        "mean_real_record_count": 0.0 if real_rows <= 0 else round(real_record_total / real_rows, 12),
        "mean_real_similarity": 0.0 if real_rows <= 0 else round(real_similarity_total / real_rows, 12),
        "mean_real_match_result": 0.0 if real_rows <= 0 else round(real_match_total / real_rows, 12),
        "mean_weight_entropy_normalized": 0.0 if not entropy_rows else round(entropy_total / len(entropy_rows), 12),
        "oracle_alignment_rows": alignment_rows,
        "oracle_alignment_mae": 0.0 if alignment_rows <= 0 else round(alignment_total / alignment_rows, 12),
        "candidate_rows": candidate_rows,
    }
    red_lines: list[str] = []
    for report in reports:
        red_lines.extend(str(value) for value in (report.get("red_line_violations", ()) or ()))
    if candidates <= 0:
        red_lines.append("no_candidate_rows")
    if int(merged["belief_domain_stats_rows"]) < candidates:
        red_lines.append("belief_domain_stats_missing")
    if float(merged["real_distribution_coverage"]) < float(min_real_coverage):
        red_lines.append("real_distribution_coverage_low")
    if float(merged["mean_real_record_count"]) < float(min_mean_real_records):
        red_lines.append("real_record_count_low")
    if float(merged["mean_real_similarity"]) < float(min_mean_real_similarity):
        red_lines.append("real_similarity_low")
    if alignment_rows > 0 and float(merged["oracle_alignment_mae"]) > float(max_oracle_alignment_mae):
        red_lines.append("real_oracle_alignment_error_high")
    merged["red_line_violations"] = _dedupe(red_lines)
    return merged


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
