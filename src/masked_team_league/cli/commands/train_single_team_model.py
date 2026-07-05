#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from masked_team_league.training.checkpoints import CheckpointRegistry, ModelCheckpointRecord
from masked_team_league.reporting.metrics import DailyTrainingReport
from masked_team_league.real_platform.resources import load_decoded_runtime_rules, load_hero_resource_bundle, load_peak_arena_camp_hero_ids
from masked_team_league.training.single_team_model import LoadoutVocab, SingleTeamWinrateModel, SingleTeamWinrateModelConfig, save_single_team_model
from masked_team_league.training import (
    build_holdout_calibration_report,
    evaluate_single_team_model,
    fit_single_team_calibrator,
    load_single_team_matchup_samples_jsonl,
    train_single_team_winrate_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the position-aware SingleTeamWinrateModel from JSONL matchup samples.")
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--holdout-jsonl", type=Path, default=None)
    parser.add_argument("--heroes-json", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, default=None)
    parser.add_argument("--camp-group", type=int, default=3)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
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
    samples = load_single_team_matchup_samples_jsonl(args.samples_jsonl, resources.by_hero_id)
    holdout_samples = (
        load_single_team_matchup_samples_jsonl(args.holdout_jsonl, resources.by_hero_id) if args.holdout_jsonl is not None else ()
    )
    vocab = LoadoutVocab.from_loadouts(resources.loadouts)
    config = SingleTeamWinrateModelConfig(model_dim=args.model_dim, heads=args.heads, layers=args.layers)
    model = SingleTeamWinrateModel(vocab, config)
    history = train_single_team_winrate_model(
        model,
        vocab,
        samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    calibrator = fit_single_team_calibrator(model, vocab, samples, device=args.device) if args.calibrate else None
    metrics = evaluate_single_team_model(model, vocab, samples, calibrator=calibrator, device=args.device)
    holdout_metrics = (
        evaluate_single_team_model(model, vocab, holdout_samples, calibrator=calibrator, device=args.device) if holdout_samples else None
    )
    save_single_team_model(args.out_model, model, vocab)
    report = DailyTrainingReport(
        date="unknown",
        sim_games=sum(sample.games for sample in samples),
        real_matches=0,
        single_model={"auc": metrics["auc"], "brier": metrics["brier"], "ece": metrics["ece"]},
        attack_oracle={"top1": 0.0, "top5_hit": 0.0},
        defense_oracle={"attack_success": 0.0, "ambiguity": 0.0},
        league={"attack_pool": 0, "defense_pool": 0, "clusters": 0},
        underdog={"samples": 0, "success_rate": 0.0},
        active_queries=[],
        failure_cases=[],
    ).to_json_dict()
    report["train_losses"] = list(history.train_losses)
    report["single_model"].update(metrics)
    if holdout_metrics is not None:
        report["holdout_single_model"] = holdout_metrics
        report["holdout_calibration"] = build_holdout_calibration_report(holdout_metrics).to_json_dict()
    if calibrator is not None:
        report["calibration"] = {"logit_scale": calibrator.logit_scale, "bias": calibrator.bias}
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.registry is not None:
        registry_metrics = _registry_metrics(holdout_metrics or metrics, prefix="holdout_" if holdout_metrics is not None else "")
        CheckpointRegistry(args.registry).add(
            ModelCheckpointRecord(
                checkpoint_id=args.out_model.stem,
                model_type="single_team_value",
                model_path=str(args.out_model),
                metrics_path=str(args.out_metrics),
                created_at=time.time(),
                dataset_hash=_file_sha256(args.samples_jsonl),
                metrics=registry_metrics,
            )
        )
    print(json.dumps({"model": str(args.out_model), "metrics": str(args.out_metrics), "samples": len(samples)}, ensure_ascii=False))
    return 0


def _registry_metrics(metrics: dict[str, float], *, prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
