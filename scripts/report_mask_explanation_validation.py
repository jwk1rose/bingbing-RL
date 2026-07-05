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

from masked_team_league.reports import build_mask_explanation_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate learned mask/risk explanation coverage in league round artifacts.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-hidden-explanation-coverage", type=float, default=0.95)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = tuple(
        build_mask_explanation_validation_report(
            round_dir,
            min_hidden_explanation_coverage=args.min_hidden_explanation_coverage,
        )
        for round_dir in args.round_dir
    )
    report = reports[0] if len(reports) == 1 else _merge_reports(reports, threshold=args.min_hidden_explanation_coverage)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "defenses": report["defenses"],
                "hidden_explanation_coverage": report["hidden_explanation_coverage"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _merge_reports(reports: tuple[dict[str, object], ...], *, threshold: float) -> dict[str, object]:
    defenses = sum(int(report.get("defenses", 0) or 0) for report in reports)
    total_hidden = sum(int(report.get("total_hidden_slots", 0) or 0) for report in reports)
    explained_hidden = sum(int(report.get("explained_hidden_slots", 0) or 0) for report in reports)
    defense_rows = [row for report in reports for row in (report.get("defense_rows", ()) or ())]
    hidden_counts = [
        float(report.get("mean_hidden_count", 0.0) or 0.0) * int(report.get("defenses", 0) or 0)
        for report in reports
    ]
    learned_score_weights = [
        float(report.get("mean_learned_mask_score", 0.0) or 0.0) * int(report.get("learned_mask_score_rows", 0) or 0)
        for report in reports
    ]
    learned_rows = sum(int(report.get("learned_mask_score_rows", 0) or 0) for report in reports)
    merged = {
        "schema_version": "mask_explanation_validation_report.v1",
        "module": "MaskExplanationValidationReport",
        "round_dir": ",".join(str(report.get("round_dir", "")) for report in reports),
        "round_dirs": [str(report.get("round_dir", "")) for report in reports],
        "defenses": defenses,
        "risk_report_rows": sum(int(report.get("risk_report_rows", 0) or 0) for report in reports),
        "mask_explanation_rows": sum(int(report.get("mask_explanation_rows", 0) or 0) for report in reports),
        "learned_mask_score_rows": learned_rows,
        "counter_attack_risk_rows": sum(int(report.get("counter_attack_risk_rows", 0) or 0) for report in reports),
        "defenses_with_no_hidden_slots": sum(int(report.get("defenses_with_no_hidden_slots", 0) or 0) for report in reports),
        "total_hidden_slots": total_hidden,
        "explained_hidden_slots": explained_hidden,
        "hidden_explanation_coverage": 0.0 if total_hidden <= 0 else explained_hidden / total_hidden,
        "mean_hidden_count": 0.0 if defenses <= 0 else sum(hidden_counts) / defenses,
        "mean_learned_mask_score": 0.0 if learned_rows <= 0 else sum(learned_score_weights) / learned_rows,
        "defense_rows": defense_rows,
    }
    red_lines: list[str] = []
    for report in reports:
        red_lines.extend(str(value) for value in (report.get("red_line_violations", ()) or ()))
    if float(merged["hidden_explanation_coverage"]) < float(threshold):
        red_lines.append("hidden_slot_explanation_coverage_low")
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
