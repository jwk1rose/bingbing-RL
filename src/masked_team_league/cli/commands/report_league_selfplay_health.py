#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.reporting.validation_reports import build_league_selfplay_health_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LeagueManager/PSRO self-play pool, role, payoff, and retention health.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--min-attack-pool", type=int, default=1)
    parser.add_argument("--min-defense-pool", type=int, default=1)
    parser.add_argument("--min-total-clusters", type=int, default=2)
    parser.add_argument("--min-payoff-density", type=float, default=0.0)
    parser.add_argument("--required-attack-role", action="append", default=None)
    parser.add_argument("--required-defense-role", action="append", default=None)
    parser.add_argument("--min-active-pool-fraction", type=float, default=0.0)
    parser.add_argument("--min-new-attack-strength-delta", type=float, default=None)
    parser.add_argument("--min-new-defense-strength-delta", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_league_selfplay_health_report(
        args.round_dir,
        min_attack_pool=args.min_attack_pool,
        min_defense_pool=args.min_defense_pool,
        min_total_clusters=args.min_total_clusters,
        min_payoff_density=args.min_payoff_density,
        required_attack_roles=args.required_attack_role or ("main", "exploiter", "underdog"),
        required_defense_roles=args.required_defense_role or ("main", "exploiter", "underdog"),
        min_active_pool_fraction=args.min_active_pool_fraction,
        min_new_attack_strength_delta=args.min_new_attack_strength_delta,
        min_new_defense_strength_delta=args.min_new_defense_strength_delta,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.out_report),
                "rounds": report["rounds"],
                "attack_pool": report["attack_pool"],
                "defense_pool": report["defense_pool"],
                "payoff_density": report["payoff_density"],
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
