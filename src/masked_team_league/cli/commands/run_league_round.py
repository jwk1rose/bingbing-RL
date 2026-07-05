#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_team_league.real_platform.backend import OracleBackendClient, is_oracle_backend_ready
from masked_team_league.scoring import SimulationCache
from masked_team_league.real_platform.oracle import OracleBatchEvaluator
from masked_team_league.real_platform.resources import DEFAULT_ORACLE_EXCLUDED_HERO_IDS
from masked_team_league.real_platform.resources import load_hero_resource_bundle, load_peak_arena_camp_hero_ids, load_unique_legend_equip_ids
from masked_team_league.real_platform.resources import load_decoded_runtime_rules
from masked_team_league.league.round_runner import LeagueRoundConfig, LeagueRoundRunner


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
    parser.add_argument("--defense-roster-candidates", type=int, default=8)
    parser.add_argument("--defense-masks-per-roster", type=int, default=2)
    parser.add_argument("--defense-max-masks-per-roster", type=int, default=128)
    parser.add_argument("--attack-role", action="append", choices=("main", "exploiter", "underdog"), default=None)
    parser.add_argument("--defense-role", action="append", choices=("main", "exploiter", "underdog"), default=None)
    parser.add_argument("--underdog-power-ratio", type=float, default=0.9)
    parser.add_argument("--underdog-residual-weight", type=float, default=0.25)
    parser.add_argument("--active-sim-keep", type=int, default=32)
    parser.add_argument("--active-real-keep", type=int, default=0)
    parser.add_argument("--attack-pool-max-active", type=int, default=None)
    parser.add_argument("--defense-pool-max-active", type=int, default=None)
    parser.add_argument("--historical-keep", type=int, default=4)
    parser.add_argument("--attack-proposal-checkpoint", type=Path, default=None)
    parser.add_argument("--attack-proposal-beam-size", type=int, default=8)
    parser.add_argument("--attack-proposal-device", default=None)
    parser.add_argument("--defense-proposal-checkpoint", type=Path, default=None)
    parser.add_argument("--defense-proposal-beam-size", type=int, default=8)
    parser.add_argument("--defense-proposal-device", default=None)
    parser.add_argument("--belief-ranker-checkpoint", type=Path, default=None)
    parser.add_argument("--belief-ranker-registry", type=Path, default=None)
    parser.add_argument("--belief-ranker-metric", default="holdout_top1_accuracy")
    parser.add_argument("--belief-ranker-metric-mode", choices=("min", "max"), default="max")
    parser.add_argument("--belief-ranker-dataset-hash", default=None)
    parser.add_argument("--belief-ranker-weight", type=float, default=1.0)
    parser.add_argument("--belief-ranker-device", default=None)
    parser.add_argument("--real-meta-db-jsonl", type=Path, default=None)
    parser.add_argument("--disable-position-features", action="store_true")
    parser.add_argument("--disable-equipment-star-features", action="store_true")
    parser.add_argument("--disable-future-feasibility-mask", action="store_true")
    parser.add_argument("--disable-real-calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=2026070401)
    parser.add_argument("--season-buff-id", type=int, action="append", default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--exclude-hero-id", type=int, action="append", default=[])
    parser.add_argument("--include-oracle-unstable-heroes", action="store_true")
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
    excluded_hero_ids = tuple(args.exclude_hero_id)
    if not args.include_oracle_unstable_heroes:
        excluded_hero_ids = tuple(sorted(set(DEFAULT_ORACLE_EXCLUDED_HERO_IDS) | set(excluded_hero_ids)))
    resources = load_hero_resource_bundle(
        args.heroes_json,
        hero_ids=hero_ids,
        excluded_hero_ids=excluded_hero_ids,
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
            defense_roster_candidates=args.defense_roster_candidates,
            defense_masks_per_roster=args.defense_masks_per_roster,
            defense_max_masks_per_roster=args.defense_max_masks_per_roster,
            attack_roles=tuple(args.attack_role) if args.attack_role else ("main", "exploiter", "underdog"),
            defense_roles=tuple(args.defense_role) if args.defense_role else ("main", "exploiter", "underdog"),
            underdog_power_ratio=args.underdog_power_ratio,
            underdog_residual_weight=args.underdog_residual_weight,
            active_sim_keep=args.active_sim_keep,
            active_real_keep=args.active_real_keep,
            attack_pool_max_active=args.attack_pool_max_active,
            defense_pool_max_active=args.defense_pool_max_active,
            historical_keep=args.historical_keep,
            attack_proposal_checkpoint=args.attack_proposal_checkpoint,
            attack_proposal_beam_size=args.attack_proposal_beam_size,
            attack_proposal_device=args.attack_proposal_device,
            defense_proposal_checkpoint=args.defense_proposal_checkpoint,
            defense_proposal_beam_size=args.defense_proposal_beam_size,
            defense_proposal_device=args.defense_proposal_device,
            belief_ranker_checkpoint=args.belief_ranker_checkpoint,
            belief_ranker_registry=args.belief_ranker_registry,
            belief_ranker_metric=args.belief_ranker_metric,
            belief_ranker_metric_mode=args.belief_ranker_metric_mode,
            belief_ranker_dataset_hash=args.belief_ranker_dataset_hash,
            belief_ranker_weight=args.belief_ranker_weight,
            belief_ranker_device=args.belief_ranker_device,
            real_meta_db_jsonl=args.real_meta_db_jsonl,
            use_position_features=not args.disable_position_features,
            use_equipment_star_features=not args.disable_equipment_star_features,
            use_future_feasibility_mask=not args.disable_future_feasibility_mask,
            use_real_calibration=not args.disable_real_calibration,
        ),
    )
    summary = runner.run(args.out_dir)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
