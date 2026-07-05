#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_production_readiness_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate v4 validation reports into a production readiness gate.")
    parser.add_argument("--report-json", type=Path, action="append", default=[])
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--required-schema-version", action="append", default=[])
    parser.add_argument("--min-clean-report-rate", type=float, default=1.0)
    parser.add_argument("--no-require-production-ready", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_paths = list(args.report_json)
    if args.reports_root is not None:
        report_paths.extend(
            path
            for path in sorted(args.reports_root.glob("**/*report*.json"))
            if path.resolve() != args.out_report.resolve()
        )
    report = build_production_readiness_report(
        report_paths,
        required_schema_versions=args.required_schema_version,
        min_clean_report_rate=args.min_clean_report_rate,
        require_production_ready=not args.no_require_production_ready,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "reports": report["reports"],
                "clean_report_rate": report["clean_report_rate"],
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
