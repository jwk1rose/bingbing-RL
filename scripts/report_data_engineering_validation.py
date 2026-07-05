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

from masked_team_league.reports import build_data_engineering_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate run metadata, artifact hashes, and core table coverage for league round artifacts."
    )
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-metadata-coverage", type=float, default=1.0)
    parser.add_argument("--min-core-table-coverage", type=float, default=1.0)
    parser.add_argument("--min-artifact-hash-coverage", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_data_engineering_validation_report(
        args.round_dir,
        min_metadata_coverage=args.min_metadata_coverage,
        min_core_table_coverage=args.min_core_table_coverage,
        min_artifact_hash_coverage=args.min_artifact_hash_coverage,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "rounds": report["rounds"],
                "metadata_coverage": report["metadata_coverage"],
                "core_table_coverage": report["core_table_coverage"],
                "artifact_hash_coverage": report["artifact_hash_coverage"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
