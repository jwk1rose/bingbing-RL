#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_active_real_query_dispatch_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate active real-query dispatch validation outputs into a scheduler-readable gate."
    )
    parser.add_argument("--validation-json", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--min-dispatched-pairs", type=int, default=1)
    parser.add_argument("--min-completion-rate", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_active_real_query_dispatch_validation_report(
        args.validation_json,
        min_reports=args.min_reports,
        min_dispatched_pairs=args.min_dispatched_pairs,
        min_completion_rate=args.min_completion_rate,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "reports": report["reports"],
                "dispatched_pairs": report["dispatched_pairs"],
                "completion_rate": report["completion_rate"],
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
