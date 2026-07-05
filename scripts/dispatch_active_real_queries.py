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

from masked_team_league.active_feedback import dispatch_active_real_queries
from masked_team_league.backend import OracleBackendClient, is_oracle_backend_ready
from masked_team_league.cache import SimulationCache
from masked_team_league.real_oracle import OracleBatchEvaluator
from masked_team_league.resources import (
    DEFAULT_ORACLE_EXCLUDED_HERO_IDS,
    load_decoded_runtime_rules,
    load_hero_resource_bundle,
    load_peak_arena_camp_hero_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch active real-query queue entries and emit teacher feedback JSONL.")
    parser.add_argument("--backend", default="http://127.0.0.1:18281")
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--job-prefix", default="active_real_query")
    parser.add_argument("--base-seed", type=int, default=2026070501)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--season-buff-id", type=int, action="append", default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--exclude-hero-id", type=int, action="append", default=[])
    parser.add_argument("--include-oracle-unstable-heroes", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hero_ids = None
    runtime_rules = None
    if args.decoded_dir is not None:
        hero_ids = load_peak_arena_camp_hero_ids(args.decoded_dir, camp_group=args.camp_group)
        runtime_rules = load_decoded_runtime_rules(args.decoded_dir)
    excluded_hero_ids = tuple(args.exclude_hero_id)
    if not args.include_oracle_unstable_heroes:
        excluded_hero_ids = tuple(sorted(set(DEFAULT_ORACLE_EXCLUDED_HERO_IDS) | set(excluded_hero_ids)))
    resources = load_hero_resource_bundle(
        args.heroes_json,
        hero_ids=hero_ids,
        excluded_hero_ids=excluded_hero_ids,
        unique_legend_equip_ids=runtime_rules.unique_legend_equip_ids if runtime_rules else None,
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
    summary = dispatch_active_real_queries(
        args.round_dir,
        args.out_dir,
        evaluator=evaluator,
        job_prefix=args.job_prefix,
        base_seed=args.base_seed,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
