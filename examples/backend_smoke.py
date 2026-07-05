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

from masked_team_league.real_platform.backend import OracleBackendClient, OracleBackendSimulator, is_oracle_backend_ready
from masked_team_league.generation import LegalPlanGenerator
from masked_team_league.domain import MatchFormat
from masked_team_league.real_platform.resources import load_hero_resource_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit one legal generated BO3/BO5 smoke match to an existing oracle backend.")
    parser.add_argument("--backend", default="http://127.0.0.1:18281")
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--teams", type=int, choices=(3, 5), default=3)
    parser.add_argument("--seed", type=int, default=2026070401)
    parser.add_argument("--request-prefix", default="masked-league-smoke")
    parser.add_argument("--unique-star", type=int, choices=(3, 4, 5), default=5)
    parser.add_argument("--season-buff-id", type=int, action="append", default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = load_hero_resource_bundle(args.heroes_json, unique_equip_star=args.unique_star)
    fmt = MatchFormat(args.teams)
    generator = LegalPlanGenerator(bundle.loadouts, seed=args.seed)
    attack = generator.generate_attack_plan(fmt)
    defense = generator.generate_defense_plan(fmt)
    client = OracleBackendClient(args.backend, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
    status = client.status()
    if not is_oracle_backend_ready(status):
        raise RuntimeError(f"oracle backend is not ready: {status}")
    season_buff_ids = args.season_buff_id
    if season_buff_ids is not None and len(season_buff_ids) == 1:
        season_buff_ids = season_buff_ids[0]
    simulator = OracleBackendSimulator(
        client,
        bundle,
        season_buff_ids=season_buff_ids,
        camp_group=args.camp_group,
    )
    score = simulator.run_plan(
        attack,
        defense,
        request_prefix=args.request_prefix,
        base_seed=args.seed,
        metadata={"kind": "masked_team_league_backend_smoke", "teams": args.teams, "seed": args.seed},
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "backend": args.backend,
                "requests": len(score.requests),
                "attack_match_win_rate": score.attack_match_win_rate,
                "round_win_rates": score.round_win_rates,
                "attack_hash": attack.hash(),
                "defense_hash": defense.hash(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
