#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_defense_anti_meta_effectiveness_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report defense anti-meta residual feedback quality from defense teacher artifacts.")
    parser.add_argument("--teacher-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-feedback-coverage", type=float, default=0.95)
    parser.add_argument("--min-positive-residual-rate", type=float, default=0.50)
    parser.add_argument("--min-mean-residual", type=float, default=0.0)
    parser.add_argument("--min-trend-delta", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_defense_anti_meta_effectiveness_report(
        args.teacher_jsonl,
        training_root=args.training_root,
        min_feedback_coverage=args.min_feedback_coverage,
        min_positive_residual_rate=args.min_positive_residual_rate,
        min_mean_residual=args.min_mean_residual,
        min_trend_delta=args.min_trend_delta,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "teacher_rows": report["teacher_rows"],
                "anti_meta_feedback_coverage": report["anti_meta_feedback_coverage"],
                "mean_residual": report["anti_meta"]["mean_residual"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
