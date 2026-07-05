#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masked_team_league.real_calibration import build_real_calibration_samples_from_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build real calibration sample JSONL from league round and active real-query artifacts.")
    parser.add_argument("--round-dir", type=Path, action="append", default=[])
    parser.add_argument("--active-real-feedback-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--rank-segment", default="unknown")
    parser.add_argument("--server", default="oracle_backend")
    parser.add_argument("--season", required=True)
    parser.add_argument("--timestamp", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.round_dir and not args.active_real_feedback_dir:
        raise SystemExit("at least one --round-dir or --active-real-feedback-dir is required")
    summary = build_real_calibration_samples_from_artifacts(
        out_jsonl=args.out_jsonl,
        round_dirs=args.round_dir,
        active_real_feedback_dirs=args.active_real_feedback_dir,
        rank_segment=args.rank_segment,
        server=args.server,
        season=args.season,
        timestamp=time.time() if args.timestamp is None else args.timestamp,
    )
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(summary.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary.to_json_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
