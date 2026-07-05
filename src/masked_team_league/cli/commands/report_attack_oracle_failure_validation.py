#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_attack_oracle_failure_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AttackOracle failure annotations and diagnostics.")
    parser.add_argument("--oracle-output-json", type=Path, action="append", default=[])
    parser.add_argument("--round-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-failure-annotation-coverage", type=float, default=1.0)
    parser.add_argument("--min-failure-diagnostic-coverage", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_attack_oracle_failure_validation_report(
        oracle_output_paths=args.oracle_output_json,
        round_dirs=args.round_dir,
        min_failure_annotation_coverage=args.min_failure_annotation_coverage,
        min_failure_diagnostic_coverage=args.min_failure_diagnostic_coverage,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "checked_rows": report["checked_rows"],
                "failure_rows": report["failure_rows"],
                "failure_annotation_coverage": report["failure_annotation_coverage"],
                "failure_diagnostic_coverage": report["failure_diagnostic_coverage"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
