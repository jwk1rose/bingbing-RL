from __future__ import annotations

from masked_team_league.training.checkpoints import CheckpointRegistry, ModelCheckpointRecord


def test_checkpoint_registry_persists_and_selects_best_metric(tmp_path):
    path = tmp_path / "registry.json"
    registry = CheckpointRegistry(path)
    weak = ModelCheckpointRecord(
        checkpoint_id="value-r0001",
        model_type="single_team_value",
        model_path="models/value_r0001.pt",
        metrics_path="models/value_r0001.metrics.json",
        created_at=10.0,
        dataset_hash="dataset-a",
        metrics={"holdout_brier": 0.30, "holdout_auc": 0.70},
    )
    strong = ModelCheckpointRecord(
        checkpoint_id="value-r0002",
        model_type="single_team_value",
        model_path="models/value_r0002.pt",
        metrics_path="models/value_r0002.metrics.json",
        created_at=20.0,
        dataset_hash="dataset-b",
        metrics={"holdout_brier": 0.20, "holdout_auc": 0.80},
    )

    registry.add(weak)
    registry.add(strong)
    reloaded = CheckpointRegistry(path)

    assert reloaded.latest("single_team_value") == strong
    assert reloaded.best(metric="holdout_brier", mode="min") == strong
    assert reloaded.best(metric="holdout_auc", mode="max") == strong
    assert reloaded.all() == (weak, strong)
