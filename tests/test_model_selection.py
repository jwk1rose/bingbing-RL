from __future__ import annotations

import json
import subprocess
import sys

from masked_team_league.checkpointing import CheckpointRegistry, ModelCheckpointRecord
from masked_team_league.model_selection import (
    build_jsonl_split_manifest,
    load_split_manifest,
    select_best_checkpoint,
    write_split_manifest,
)


def test_build_jsonl_split_manifest_records_counts_hashes_and_dataset_hash(tmp_path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    valid.write_text('{"a":3}\n', encoding="utf-8")

    manifest = build_jsonl_split_manifest(
        {"train": (train,), "valid": (valid,)},
        dataset_id="single-team-v1",
        version="2026-07-05",
        metadata={"source": "unit"},
    )
    path = tmp_path / "split_manifest.json"
    write_split_manifest(path, manifest)
    reloaded = load_split_manifest(path)

    assert manifest.split_counts == {"train": 2, "valid": 1}
    assert len(manifest.dataset_hash) == 64
    assert manifest.files[0].sha256
    assert reloaded == manifest


def test_select_best_checkpoint_filters_registry_and_writes_selection(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = CheckpointRegistry(registry_path)
    weak = ModelCheckpointRecord(
        checkpoint_id="value-r0001",
        model_type="single_team_value",
        model_path="models/value_r0001.pt",
        metrics_path="models/value_r0001.metrics.json",
        created_at=10.0,
        dataset_hash="dataset-a",
        metrics={"holdout_auc": 0.70, "holdout_brier": 0.30},
    )
    strong = ModelCheckpointRecord(
        checkpoint_id="value-r0002",
        model_type="single_team_value",
        model_path="models/value_r0002.pt",
        metrics_path="models/value_r0002.metrics.json",
        created_at=20.0,
        dataset_hash="dataset-a",
        metrics={"holdout_auc": 0.82, "holdout_brier": 0.24},
    )
    unrelated = ModelCheckpointRecord(
        checkpoint_id="belief-r0001",
        model_type="belief_ranker",
        model_path="models/belief.pt",
        metrics_path="models/belief.metrics.json",
        created_at=30.0,
        dataset_hash="dataset-a",
        metrics={"holdout_auc": 0.99},
    )
    registry.add(weak)
    registry.add(strong)
    registry.add(unrelated)

    selected = select_best_checkpoint(
        registry_path,
        metric="holdout_auc",
        mode="max",
        model_type="single_team_value",
        dataset_hash="dataset-a",
        out_path=tmp_path / "selected.json",
    )
    payload = json.loads((tmp_path / "selected.json").read_text(encoding="utf-8"))

    assert selected == strong
    assert payload["checkpoint"]["checkpoint_id"] == "value-r0002"
    assert payload["selection"]["metric"] == "holdout_auc"
    assert payload["selection"]["mode"] == "max"


def test_select_model_checkpoint_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/select_model_checkpoint.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--metric" in result.stdout
    assert "--out-json" in result.stdout


def test_build_split_manifest_script_has_help():
    result = subprocess.run(
        [sys.executable, "scripts/build_split_manifest.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "--dataset-id" in result.stdout
    assert "--out-json" in result.stdout
