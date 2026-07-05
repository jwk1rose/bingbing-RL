#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from masked_team_league.real_platform.calibration import (
    RealMetaDB,
    build_version_drift_report,
    ingest_active_real_query_feedback,
    ingest_league_round_real_meta,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest league round and active real-query oracle results into RealMetaDB and report version drift.")
    parser.add_argument("--round-dir", type=Path, action="append", default=[])
    parser.add_argument("--active-real-feedback-dir", type=Path, action="append", default=[])
    parser.add_argument("--db-jsonl", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--rank-segment", default="unknown")
    parser.add_argument("--server", default="oracle_backend")
    parser.add_argument("--season", required=True)
    parser.add_argument("--timestamp", type=float, default=None)
    parser.add_argument("--drift-baseline-season", default=None)
    parser.add_argument("--drift-current-season", default=None)
    parser.add_argument("--drift-delta-threshold", type=float, default=0.15)
    parser.add_argument("--drift-min-overlap", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.round_dir and not args.active_real_feedback_dir:
        raise SystemExit("at least one --round-dir or --active-real-feedback-dir is required")
    timestamp = time.time() if args.timestamp is None else args.timestamp
    ingestions = [
        ingest_league_round_real_meta(
            round_dir,
            args.db_jsonl,
            rank_segment=args.rank_segment,
            server=args.server,
            season=args.season,
            timestamp=timestamp,
        )
        for round_dir in args.round_dir
    ]
    ingestions.extend(
        ingest_active_real_query_feedback(
            feedback_dir,
            args.db_jsonl,
            rank_segment=args.rank_segment,
            server=args.server,
            season=args.season,
            timestamp=timestamp,
        )
        for feedback_dir in args.active_real_feedback_dir
    )
    db = RealMetaDB.load(args.db_jsonl)
    drift = None
    if args.drift_baseline_season is not None and args.drift_current_season is not None:
        drift = build_version_drift_report(
            db.all(),
            baseline_season=args.drift_baseline_season,
            current_season=args.drift_current_season,
            delta_threshold=args.drift_delta_threshold,
            min_overlap=args.drift_min_overlap,
        )
    payload = {
        "db_jsonl": str(args.db_jsonl),
        "ingestions": [summary.to_json_dict() for summary in ingestions],
        "total_records": len(db.all()),
        "drift": None if drift is None else drift.to_json_dict(),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out_report), "total_records": len(db.all())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
