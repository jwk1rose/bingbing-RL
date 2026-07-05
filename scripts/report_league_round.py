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

from masked_team_league.reports import build_league_round_report, red_line_violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a v4 daily-style report from one league round artifact directory.")
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--date", default="unknown")
    parser.add_argument("--fail-on-red-line", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_league_round_report(args.round_dir, date=args.date).to_json_dict()
    report["red_line_violations"] = red_line_violations(report)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_report": str(args.out_report), "red_line_violations": report["red_line_violations"]}, ensure_ascii=False))
    if args.fail_on_red_line and report["red_line_violations"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
