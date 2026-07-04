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

from masked_team_league.backend import OracleBackendClient, is_oracle_backend_ready
from masked_team_league.cache import SimulationCache
from masked_team_league.real_oracle import OracleBatchEvaluator
from masked_team_league.resources import load_hero_resource_bundle, load_peak_arena_camp_hero_ids, load_unique_legend_equip_ids
from masked_team_league.resources import load_decoded_runtime_rules
from masked_team_league.round_runner import LeagueRoundConfig, LeagueRoundRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one masked-team league round against a ready oracle backend.")
    parser.add_argument("--backend", default="http://127.0.0.1:18281")
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--legend-equip-lua", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--round-id", default="round_0001")
    parser.add_argument("--teams", type=int, choices=(3, 5), default=3)
    parser.add_argument("--defenses", type=int, default=20)
    parser.add_argument("--attacks-per-defense", type=int, default=200)
    parser.add_argument("--oracle-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026070401)
    parser.add_argument("--season-buff-id", type=int, action="append", default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    unique_legend_ids = None
    hero_ids = None
    runtime_rules = None
    if args.decoded_dir is not None:
        hero_ids = load_peak_arena_camp_hero_ids(args.decoded_dir, camp_group=args.camp_group)
        runtime_rules = load_decoded_runtime_rules(args.decoded_dir)
        unique_legend_ids = runtime_rules.unique_legend_equip_ids
    if args.legend_equip_lua is not None:
        unique_legend_ids = load_unique_legend_equip_ids(args.legend_equip_lua)
    resources = load_hero_resource_bundle(
        args.heroes_json,
        hero_ids=hero_ids,
        unique_legend_equip_ids=unique_legend_ids,
        runtime_rules=runtime_rules,
    )
    client = OracleBackendClient(args.backend, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
    status = client.status()
    if not is_oracle_backend_ready(status):
        raise RuntimeError(f"oracle backend is not ready: {status}")
    season_buff_ids = args.season_buff_id
    if season_buff_ids is not None and len(season_buff_ids) == 1:
        season_buff_ids = season_buff_ids[0]
    evaluator = OracleBatchEvaluator(
        client,
        resources,
        cache=SimulationCache(),
        season_buff_ids=season_buff_ids,
        camp_group=args.camp_group,
    )
    runner = LeagueRoundRunner(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=LeagueRoundConfig(
            teams=args.teams,
            defenses=args.defenses,
            attacks_per_defense=args.attacks_per_defense,
            oracle_top_k=args.oracle_top_k,
            seed=args.seed,
            round_id=args.round_id,
        ),
    )
    summary = runner.run(args.out_dir)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
