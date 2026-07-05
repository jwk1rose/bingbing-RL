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

from masked_team_league.reports import build_underdog_residual_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate underdog residual objective coverage and production quality from league round artifacts."
    )
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-attack-residual-coverage", type=float, default=0.95)
    parser.add_argument("--min-defense-residual-coverage", type=float, default=0.95)
    parser.add_argument("--min-mean-attack-residual-bonus", type=float, default=0.0)
    parser.add_argument("--min-mean-defense-residual-bonus", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_underdog_residual_validation_report(
        args.round_dir,
        min_attack_residual_coverage=args.min_attack_residual_coverage,
        min_defense_residual_coverage=args.min_defense_residual_coverage,
        min_mean_attack_residual_bonus=args.min_mean_attack_residual_bonus,
        min_mean_defense_residual_bonus=args.min_mean_defense_residual_bonus,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "rounds": report["rounds"],
                "attack_underdog_rows": report["attack_underdog_rows"],
                "attack_residual_coverage": report["attack_residual_coverage"],
                "mean_attack_residual_bonus": report["mean_attack_residual_bonus"],
                "defense_underdog_rows": report["defense_underdog_rows"],
                "defense_residual_coverage": report["defense_residual_coverage"],
                "mean_defense_residual_bonus": report["mean_defense_residual_bonus"],
                "red_line_violations": report["red_line_violations"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
