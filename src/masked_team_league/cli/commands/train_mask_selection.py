#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from masked_team_league.generation.proposal_networks import MaskSelectionNetwork
from masked_team_league.generation.proposal_training import (
    MASK_SLOT_FEATURE_NAMES,
    load_mask_training_samples_jsonl,
    save_mask_selection_checkpoint,
    train_mask_selection_network,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MaskSelectionNetwork from scored defense artifacts.")
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = load_mask_training_samples_jsonl(args.teacher_jsonl)
    network = MaskSelectionNetwork(feature_dim=len(MASK_SLOT_FEATURE_NAMES), hidden_dim=args.hidden_dim)
    history = train_mask_selection_network(
        network,
        samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ranking_weight=args.ranking_weight,
        device=args.device,
        seed=args.seed,
    )
    record = save_mask_selection_checkpoint(
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
