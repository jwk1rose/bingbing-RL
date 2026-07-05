#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masked_team_league.constraints import ConstraintEngine
from masked_team_league.proposal_networks import AttackGenerationNetwork, ProposalNetworkConfig
from masked_team_league.proposal_training import (
    load_attack_teacher_samples_jsonl,
    save_proposal_network_checkpoint,
    train_proposal_network,
)
from masked_team_league.resources import load_decoded_runtime_rules, load_hero_resource_bundle, load_peak_arena_camp_hero_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AttackGenerationNetwork from attack teacher JSONL artifacts.")
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-weight-temperature", type=float, default=0.25)
    parser.add_argument("--min-candidate-weight", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-slots", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hero_ids = None
    runtime_rules = None
    if args.decoded_dir is not None:
        hero_ids = load_peak_arena_camp_hero_ids(args.decoded_dir, camp_group=args.camp_group)
        runtime_rules = load_decoded_runtime_rules(args.decoded_dir)
    resources = load_hero_resource_bundle(
        args.heroes_json,
        hero_ids=hero_ids,
        unique_legend_equip_ids=runtime_rules.unique_legend_equip_ids if runtime_rules else None,
        runtime_rules=runtime_rules,
    )
    engine = ConstraintEngine(resources.loadouts)
    samples = load_attack_teacher_samples_jsonl(
        args.teacher_jsonl,
        loadout_pool=resources.loadouts,
        constraint_engine=engine,
        candidate_weight_temperature=args.candidate_weight_temperature,
        min_candidate_weight=args.min_candidate_weight,
    )
    config = ProposalNetworkConfig(
        loadout_count=len(resources.loadouts),
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        max_slots=args.max_slots,
    )
    network = AttackGenerationNetwork(config)
    history = train_proposal_network(
        network,
        samples,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    record = save_proposal_network_checkpoint(
        args.out_checkpoint,
        network,
        history,
        registry_path=args.registry,
        checkpoint_id=args.out_checkpoint.stem,
        dataset_hash=_file_sha256(args.teacher_jsonl),
        metadata={"teacher_jsonl": str(args.teacher_jsonl), "samples": len(samples)},
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.out_checkpoint),
                "metrics": record.metrics_path,
                "samples": len(samples),
                "train_loss": record.metrics.get("train_loss", 0.0),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
