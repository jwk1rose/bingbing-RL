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

from masked_team_league.belief_ranker import build_belief_ranker_dataset_from_rounds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build belief-ranker train/holdout JSONL datasets from league round artifacts.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-id", default="belief-ranker-rounds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_belief_ranker_dataset_from_rounds(
        tuple(args.round_dir),
        out_dir=args.out_dir,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        dataset_id=args.dataset_id,
    )
    print(
        json.dumps(
            {
                "train_jsonl": str(result.train_jsonl),
                "holdout_jsonl": str(result.holdout_jsonl),
                "manifest_json": str(result.manifest_json),
                "total_rows": result.total_rows,
                "train_rows": result.train_rows,
                "holdout_rows": result.holdout_rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
