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
from masked_team_league.resources import (
    DEFAULT_ORACLE_EXCLUDED_HERO_IDS,
    load_decoded_runtime_rules,
    load_hero_resource_bundle,
    load_peak_arena_camp_hero_ids,
)
from masked_team_league.round_runner import LeagueRoundConfig
from masked_team_league.selfplay import SelfPlayOrchestrator, SelfPlayOrchestratorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-round masked-team self-play with proposal feedback.")
    parser.add_argument("--backend", default="http://127.0.0.1:18281")
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--teams", type=int, choices=(3, 5), default=3)
    parser.add_argument("--defenses", type=int, default=20)
    parser.add_argument("--attacks-per-defense", type=int, default=200)
    parser.add_argument("--oracle-top-k", type=int, default=20)
    parser.add_argument("--defense-roster-candidates", type=int, default=8)
    parser.add_argument("--defense-masks-per-roster", type=int, default=2)
    parser.add_argument("--defense-max-masks-per-roster", type=int, default=128)
    parser.add_argument("--attack-pool-max-active", type=int, default=None)
    parser.add_argument("--defense-pool-max-active", type=int, default=None)
    parser.add_argument("--historical-keep", type=int, default=4)
    parser.add_argument("--belief-ranker-checkpoint", type=Path, default=None)
    parser.add_argument("--belief-ranker-registry", type=Path, default=None)
    parser.add_argument("--belief-ranker-metric", default="holdout_top1_accuracy")
    parser.add_argument("--belief-ranker-metric-mode", choices=("min", "max"), default="max")
    parser.add_argument("--belief-ranker-dataset-hash", default=None)
    parser.add_argument("--belief-ranker-weight", type=float, default=1.0)
    parser.add_argument("--belief-ranker-device", default=None)
    parser.add_argument("--proposal-epochs", type=int, default=1)
    parser.add_argument("--proposal-lr", type=float, default=1e-3)
    parser.add_argument("--proposal-device", default=None)
    parser.add_argument("--proposal-model-dim", type=int, default=256)
    parser.add_argument("--proposal-heads", type=int, default=8)
    parser.add_argument("--proposal-layers", type=int, default=2)
    parser.add_argument("--attack-proposal-beam-size", type=int, default=8)
    parser.add_argument("--defense-proposal-beam-size", type=int, default=8)
    parser.add_argument("--candidate-weight-temperature", type=float, default=0.25)
    parser.add_argument("--min-candidate-weight", type=float, default=0.05)
    parser.add_argument("--no-dispatch-real-queries", action="store_true")
    parser.add_argument("--validate-after-each-round", action="store_true")
    parser.add_argument("--stop-when-validation-ready", action="store_true")
    parser.add_argument("--validation-min-rounds", type=int, default=2)
    parser.add_argument("--validation-min-oracle-requests", type=int, default=1)
    parser.add_argument("--validation-no-require-latest-checkpoints", action="store_true")
    parser.add_argument("--validation-min-attack-target-coverage", type=float, default=0.95)
    parser.add_argument("--validation-min-attack-positive-residual-rate", type=float, default=0.50)
    parser.add_argument("--validation-min-attack-trend-delta", type=float, default=None)
    parser.add_argument("--validation-min-defense-feedback-coverage", type=float, default=0.95)
    parser.add_argument("--validation-min-defense-positive-residual-rate", type=float, default=0.50)
    parser.add_argument("--validation-min-defense-mean-residual", type=float, default=0.0)
    parser.add_argument("--validation-min-defense-trend-delta", type=float, default=None)
    parser.add_argument("--underdog-power-ratio", type=float, default=0.9)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--exclude-hero-id", type=int, action="append", default=[])
    parser.add_argument("--include-oracle-unstable-heroes", action="store_true")
    parser.add_argument("--seed", type=int, default=2026070401)
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
    evaluator = OracleBatchEvaluator(client, resources, cache=SimulationCache(), camp_group=args.camp_group)
    summary = SelfPlayOrchestrator(
        loadout_pool=resources.loadouts,
        evaluator=evaluator,
        config=SelfPlayOrchestratorConfig(
            rounds=args.rounds,
            start_round=args.start_round,
            root_dir=args.root_dir,
            training_dir=args.training_dir,
            round_config=LeagueRoundConfig(
                teams=args.teams,
                defenses=args.defenses,
                attacks_per_defense=args.attacks_per_defense,
                oracle_top_k=args.oracle_top_k,
                seed=args.seed,
                defense_roster_candidates=args.defense_roster_candidates,
                defense_masks_per_roster=args.defense_masks_per_roster,
                defense_max_masks_per_roster=args.defense_max_masks_per_roster,
                attack_pool_max_active=args.attack_pool_max_active,
                defense_pool_max_active=args.defense_pool_max_active,
                historical_keep=args.historical_keep,
                belief_ranker_checkpoint=args.belief_ranker_checkpoint,
                belief_ranker_registry=args.belief_ranker_registry,
                belief_ranker_metric=args.belief_ranker_metric,
                belief_ranker_metric_mode=args.belief_ranker_metric_mode,
                belief_ranker_dataset_hash=args.belief_ranker_dataset_hash,
                belief_ranker_weight=args.belief_ranker_weight,
                belief_ranker_device=args.belief_ranker_device,
                underdog_power_ratio=args.underdog_power_ratio,
            ),
            proposal_epochs=args.proposal_epochs,
            proposal_lr=args.proposal_lr,
            proposal_device=args.proposal_device,
            proposal_model_dim=args.proposal_model_dim,
            proposal_heads=args.proposal_heads,
            proposal_layers=args.proposal_layers,
            attack_proposal_beam_size=args.attack_proposal_beam_size,
            defense_proposal_beam_size=args.defense_proposal_beam_size,
            candidate_weight_temperature=args.candidate_weight_temperature,
            min_candidate_weight=args.min_candidate_weight,
            dispatch_real_queries=not args.no_dispatch_real_queries,
            validate_after_each_round=args.validate_after_each_round,
            stop_when_validation_ready=args.stop_when_validation_ready,
            validation_min_rounds=args.validation_min_rounds,
            validation_min_oracle_requests=args.validation_min_oracle_requests,
            validation_require_latest_checkpoints=not args.validation_no_require_latest_checkpoints,
            validation_min_attack_target_coverage=args.validation_min_attack_target_coverage,
            validation_min_attack_positive_residual_rate=args.validation_min_attack_positive_residual_rate,
            validation_min_attack_trend_delta=args.validation_min_attack_trend_delta,
            validation_min_defense_feedback_coverage=args.validation_min_defense_feedback_coverage,
            validation_min_defense_positive_residual_rate=args.validation_min_defense_positive_residual_rate,
            validation_min_defense_mean_residual=args.validation_min_defense_mean_residual,
            validation_min_defense_trend_delta=args.validation_min_defense_trend_delta,
        ),
    ).run()
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    return 0


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
