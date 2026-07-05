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

from masked_team_league.model_selection import select_best_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the best model checkpoint from a checkpoint registry.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--dataset-hash", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = select_best_checkpoint(
        args.registry,
        metric=args.metric,
        mode=args.mode,
        model_type=args.model_type,
        dataset_hash=args.dataset_hash,
        out_path=args.out_json,
    )
    print(
        json.dumps(
            {
                "checkpoint_id": record.checkpoint_id,
                "model_path": record.model_path,
                "metric": args.metric,
                "value": record.metric(args.metric),
                "selection": str(args.out_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
