#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.real_platform.calibration import build_real_calibration_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate real calibration on holdout samples.")
    parser.add_argument("--samples-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-brier-improvement", type=float, default=0.0)
    parser.add_argument("--min-ece-improvement", type=float, default=0.0)
    parser.add_argument("--now", type=float, default=None)
    parser.add_argument("--recency-tau", type=float, default=7.0 * 24.0 * 60.0 * 60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_real_calibration_validation_report(
        samples_jsonl=args.samples_jsonl,
        calibration_json=args.calibration_json,
        min_samples=args.min_samples,
        min_brier_improvement=args.min_brier_improvement,
        min_ece_improvement=args.min_ece_improvement,
        now=args.now,
        recency_tau=args.recency_tau,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "samples": report["samples"],
                "brier_improvement": report["brier_improvement"],
                "ece_improvement": report["ece_improvement"],
                "production_ready": report["production_ready"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
