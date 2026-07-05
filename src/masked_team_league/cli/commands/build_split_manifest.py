#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.training.model_selection import build_jsonl_split_manifest, write_split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a JSONL split manifest with row counts and content hashes.")
    parser.add_argument("--split", action="append", required=True, help="Split assignment in the form name=/path/to/file.jsonl")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--version", default="unknown")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--metadata", action="append", default=[], help="Metadata key=value pair")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    splits: dict[str, list[Path]] = {}
    for item in args.split:
        name, path = _split_assignment(item)
        splits.setdefault(name, []).append(path)
    metadata = dict(_split_assignment(item, value_type=str) for item in args.metadata)
    manifest = build_jsonl_split_manifest(splits, dataset_id=args.dataset_id, version=args.version, metadata=metadata)
    write_split_manifest(args.out_json, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.out_json),
                "dataset_id": manifest.dataset_id,
                "dataset_hash": manifest.dataset_hash,
                "split_counts": manifest.split_counts,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _split_assignment(value: str, *, value_type=Path):
    if "=" not in value:
        raise argparse.ArgumentTypeError("assignment must be in key=value form")
    key, raw = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("assignment key must be non-empty")
    if not raw:
        raise argparse.ArgumentTypeError("assignment value must be non-empty")
    return key, value_type(raw)


if __name__ == "__main__":
    raise SystemExit(main())
