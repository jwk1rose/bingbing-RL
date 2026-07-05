from __future__ import annotations

import json
from pathlib import Path

from masked_team_league.domain import ResultMetadata
from masked_team_league.data_engineering.run_metadata import (
    RUN_METADATA_SCHEMA_VERSION,
    RunArtifactRef,
    RunMetadataManifest,
    hash_generation_config,
    load_run_metadata_manifest,
    write_run_metadata_manifest,
)


def test_run_metadata_manifest_round_trips_and_hashes_artifacts(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"sample":1}\n', encoding="utf-8")
    artifact = RunArtifactRef.from_path(train_jsonl, kind="single_matchup_jsonl", role="input")
    metadata = ResultMetadata(
        model_version="model-r1",
        data_version="data-r1",
        simulator_version="sim-r1",
        league_iteration=7,
        random_seed=123,
        generation_config_hash=hash_generation_config({"top_k": 20, "seed": 123}),
        calibration_version="cal-r1",
    )
    manifest = RunMetadataManifest.from_result_metadata(
        run_id="unit-run",
        metadata=metadata,
        created_at=1000.0,
        code_version="git-sha-unit",
        input_artifacts=(artifact,),
        output_artifacts=(),
        metrics={"brier": 0.123},
        extra={"stage": "unit"},
    )
    out_path = tmp_path / "run_metadata.json"

    write_run_metadata_manifest(manifest, out_path)
    loaded = load_run_metadata_manifest(out_path)
    payload = loaded.to_json_dict()

    assert loaded == manifest
    assert payload["schema_version"] == RUN_METADATA_SCHEMA_VERSION
    assert payload["model_version"] == "model-r1"
    assert payload["data_version"] == "data-r1"
    assert payload["simulator_version"] == "sim-r1"
    assert payload["random_seed"] == 123
    assert payload["input_artifacts"][0]["sha256"]
    assert payload["input_artifacts"][0]["size_bytes"] == train_jsonl.stat().st_size
    json.dumps(payload, sort_keys=True)


def test_run_metadata_schema_doc_lists_required_reproducibility_fields():
    text = Path("docs/run_metadata_schema.md").read_text(encoding="utf-8")

    assert "run_metadata.v1" in text
    assert "model_version" in text
    assert "data_version" in text
    assert "simulator_version" in text
    assert "generation_config_hash" in text
    assert "calibration_version" in text
