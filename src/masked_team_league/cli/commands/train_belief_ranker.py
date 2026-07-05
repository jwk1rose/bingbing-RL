#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from masked_team_league.belief.ranker import (
    TorchBeliefRankerAdapter,
    evaluate_belief_ranker,
    load_belief_ranker_samples_jsonl,
    save_belief_ranker_checkpoint,
    train_belief_ranker,
)
from masked_team_league.constraints import ConstraintEngine
from masked_team_league.real_platform.resources import load_decoded_runtime_rules, load_hero_resource_bundle, load_peak_arena_camp_hero_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the neural BeliefModel ranker from defense-plan JSONL artifacts.")
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--holdout-jsonl", type=Path, default=None)
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--negative-candidates", type=int, default=31)
    parser.add_argument("--max-completions", type=int, default=128)
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
    samples = load_belief_ranker_samples_jsonl(
        args.samples_jsonl,
        loadout_pool=resources.loadouts,
        constraint_engine=engine,
        negative_candidates=args.negative_candidates,
        max_completions=args.max_completions,
    )
    holdout_samples = (
        load_belief_ranker_samples_jsonl(
            args.holdout_jsonl,
            loadout_pool=resources.loadouts,
            constraint_engine=engine,
            negative_candidates=args.negative_candidates,
            max_completions=args.max_completions,
        )
        if args.holdout_jsonl is not None
        else ()
    )
    adapter = TorchBeliefRankerAdapter.from_loadouts(resources.loadouts, model_dim=args.model_dim, device=args.device)
    history = train_belief_ranker(
        adapter.model,
        adapter.vocab,
        samples,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    train_metrics = evaluate_belief_ranker(adapter.model, adapter.vocab, samples, device=args.device)
    holdout_metrics = (
        evaluate_belief_ranker(adapter.model, adapter.vocab, holdout_samples, device=args.device) if holdout_samples else None
    )
    metrics = {f"train_{key}": value for key, value in train_metrics.items()}
    if holdout_metrics is not None:
        metrics.update({f"holdout_{key}": value for key, value in holdout_metrics.items()})
    record = save_belief_ranker_checkpoint(
        args.out_checkpoint,
        adapter.model,
        adapter.vocab,
        history,
        metrics=metrics,
        registry_path=args.registry,
        checkpoint_id=args.out_checkpoint.stem,
        dataset_hash=_file_sha256(args.samples_jsonl),
        metadata={
            "samples_jsonl": str(args.samples_jsonl),
            "holdout_jsonl": None if args.holdout_jsonl is None else str(args.holdout_jsonl),
            "samples": len(samples),
            "holdout_samples": len(holdout_samples),
            "negative_candidates": args.negative_candidates,
            "max_completions": args.max_completions,
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.out_checkpoint),
                "metrics": record.metrics_path,
                "samples": len(samples),
                "holdout_samples": len(holdout_samples),
                "train_loss": record.metrics.get("train_loss", 0.0),
                "train_top1_accuracy": record.metrics.get("train_top1_accuracy", 0.0),
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
