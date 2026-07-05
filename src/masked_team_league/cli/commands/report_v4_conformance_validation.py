#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_v4_conformance_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate v4 conformance evidence against production priorities.")
    parser.add_argument("--report-json", type=Path, action="append", default=[])
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_paths = list(args.report_json)
    if args.reports_root is not None:
        root_paths = {
            path.resolve(): path
            for pattern in ("**/*report*.json", "**/*plan*.json")
            for path in args.reports_root.glob(pattern)
            if path.resolve() != args.out_report.resolve()
        }
        report_paths.extend(path for _, path in sorted(root_paths.items(), key=lambda item: str(item[0])))
    report = build_v4_conformance_validation_report(report_paths)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "requirements_total": report["requirements_total"],
                "passed_requirements": report["passed_requirements"],
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
