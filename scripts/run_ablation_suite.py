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

from masked_team_league.ablation import build_ablation_suite_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a v4 ablation suite report from league round artifact directories.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=ROUND_DIR",
        help="Ablation variant and its round artifact directory. Repeat for baseline and each ablation.",
    )
    parser.add_argument("--baseline", required=True, help="Variant name to use as the baseline.")
    parser.add_argument("--date", default="unknown")
    parser.add_argument("--suite-id", default="v4_ablation_suite")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--require-v4-variants",
        action="store_true",
        help="Return a non-zero exit code when any required v4 ablation variant is missing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    variants = _parse_variants(args.variant)
    report = build_ablation_suite_report(
        variants,
        baseline_variant=args.baseline,
        date=args.date,
        suite_id=args.suite_id,
    ).to_json_dict()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "variants": report["variants"],
                "missing_required_variants": report["missing_required_variants"],
            },
            ensure_ascii=False,
        )
    )
    if args.require_v4_variants and report["missing_required_variants"]:
        return 2
    return 0


def _parse_variants(values: list[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"--variant must use NAME=ROUND_DIR, got {value!r}")
        if name in variants:
            raise SystemExit(f"duplicate ablation variant {name!r}")
        variants[name] = Path(raw_path)
    if not variants:
        raise SystemExit("at least one --variant is required")
    return variants


if __name__ == "__main__":
    raise SystemExit(main())
